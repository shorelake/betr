"""Checkpoint-based BETR parameter, FLOP and synchronized inference benchmark."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
from tools.betr_common import load_betr
from util.misc import NestedTensor


def deformable_attention_flops(inputs, outputs):
    # value: B,S,H,C; sampling: B,Q,H,L,P,2.
    # Four bilinear samples (4 multiplies + 3 adds), then weight multiply
    # and accumulate: approximately 9 operations per sampled channel.
    value, sampling = get_shape(inputs[0]), get_shape(inputs[3])
    b, q, heads, levels, points, _ = sampling
    return {'ms_deform_attn_estimate': b*q*heads*levels*points*value[-1]*9}


class Forward(torch.nn.Module):
    def __init__(self, model, postprocessor=None):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images, mask):
        result = self.model([NestedTensor(images, mask)])
        if self.postprocessor is not None:
            sizes = images.new_tensor([images.shape[-2], images.shape[-1]]).expand(images.shape[0], 2)
            result = self.postprocessor(result, sizes)
            return tuple(x[k] for x in result for k in ('scores', 'labels', 'boxes'))
        return result['pred_logits'], result['pred_boxes']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--coco-path', help='Use the first COCO validation image, as in ViDT; overrides fixed height/width')
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--width', type=int, default=1300)
    parser.add_argument('--batch-size', type=int, default=1)
    # Match ViDT's fps_calculator.py: total iterations include warmup, and
    # only iterations after warm-iters contribute to the reported mean.
    parser.add_argument('--num-iters', '--iterations', dest='num_iters', type=int, default=300,
                        help='total forward iterations, including warmup (ViDT default: 300)')
    parser.add_argument('--warm-iters', '--warmup', dest='warm_iters', type=int, default=5,
                        help='initial iterations discarded from timing (ViDT default: 5)')
    parser.add_argument('--postprocess', action='store_true', help='Include bbox decoding/top-k in timed forward')
    parser.add_argument('--skip-flops', action='store_true')
    parser.add_argument('--output', default='workdir/benchmark/results.json')
    parser.add_argument('--enc-layers', type=int)
    parser.add_argument('--backbone')
    cli = parser.parse_args()
    if min(cli.height, cli.width, cli.batch_size, cli.num_iters) < 1 or cli.warm_iters < 0 or cli.warm_iters >= cli.num_iters:
        parser.error('Dimensions, batch and num-iters must be positive; 0 <= warm-iters < num-iters')
    device = torch.device(cli.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.manual_seed(42)
    model, post, args, metadata = load_betr(cli.checkpoint, cli.device, cli.enc_layers, cli.backbone)
    runner = Forward(model, post['bbox'] if cli.postprocess else None).eval()
    images = torch.randn(cli.batch_size, 3, cli.height, cli.width, device=device)
    mask = torch.zeros(cli.batch_size, cli.height, cli.width, dtype=torch.bool, device=device)
    if cli.coco_path:
        from datasets import build_dataset
        from util.misc import nested_tensor_from_tensor_list
        args.coco_path = cli.coco_path
        args.dataset_file = args.dataset = 'coco'
        dataset = build_dataset('val', args)
        samples = nested_tensor_from_tensor_list([dataset[0][0].to(device) for _ in range(cli.batch_size)])
        images, mask = samples.decompose()
    result = dict(metadata=metadata, options=vars(cli), precision='float32',
                  parameters_total=sum(p.numel() for p in model.parameters()),
                  parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  parameters_backbone=sum(p.numel() for p in model.backbone.parameters()),
                  timing_scope='Resident input tensor -> forward' + (' + bbox postprocess' if cli.postprocess else '') + '; excludes IO, preprocessing and host/device transfer; existing auxiliary branches are not pruned')
    result['input_shape'] = list(images.shape)
    result['input_source'] = 'COCO val first image' if cli.coco_path else 'fixed synthetic tensor'
    result['timed_iterations'] = cli.num_iters - cli.warm_iters
    result['timing_protocol'] = 'ViDT: no_grad, synchronized per-forward wall time, discard initial warm iterations'
    result['parameters_millions'] = result['parameters_total']/1e6
    # ViDT uses no_grad (rather than inference_mode); keep this identical for
    # a comparable forward-speed measurement.
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for iteration in range(cli.num_iters):
            if device.type == 'cuda': torch.cuda.synchronize(device)
            start = time.perf_counter()
            runner(images, mask)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-start
            if iteration >= cli.warm_iters:
                durations.append(elapsed)
        result.update(fps=cli.batch_size/float(np.mean(durations)),
                      batch_latency_ms_mean=float(np.mean(durations)*1000),
                      batch_latency_ms_p50=float(np.percentile(durations,50)*1000),
                      batch_latency_ms_p95=float(np.percentile(durations,95)*1000),
                      amortized_ms_per_image=float(np.mean(durations)*1000/cli.batch_size),
                      peak_allocated_memory_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
        if not cli.skip_flops:
            # Trace full output dict so training-only attention heads executed by
            # the current forward are included, rather than eliminated as dead outputs.
            class TraceForward(torch.nn.Module):
                def __init__(self, detector):
                    super().__init__(); self.detector=detector
                def forward(self, x, m):
                    outputs = self.detector([NestedTensor(x,m)])
                    tensors = [outputs['pred_logits'], outputs['pred_boxes']]
                    for item in outputs.get('aux_outputs', []):
                        tensors.extend([item['pred_logits'], item['pred_boxes']])
                    enc = outputs.get('enc_outputs', {})
                    for key in ('pred_logits','pred_boxes','pred_filters','pred_mask','sampling_locations_dec','attn_weights_dec'):
                        if enc.get(key) is not None: tensors.append(enc[key])
                    return tuple(tensors)
            analysis = FlopCountAnalysis(TraceForward(model), (images,mask))
            analysis.set_op_handle('prim::PythonOp.MSDeformAttnFunction',deformable_attention_flops)
            analysis.tracer_warnings('none').uncalled_modules_warnings(False)
            total = analysis.total()
            unsupported = dict(analysis.unsupported_ops())
            result['flops'] = dict(total_batch=total, gflops_per_image=total/cli.batch_size/1e9,
                                  by_operator=dict(analysis.by_operator()), unsupported_operators=unsupported,
                                  partial=bool(unsupported),
                                  convention='fvcore: one fused multiply-add=1 FLOP; custom deformable sampling estimated at 9 ops/sample/channel; coordinate arithmetic omitted. Forward only, postprocess excluded regardless of timing flag.')
    path = Path(cli.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(f"Parameters: {result['parameters_millions']:.3f} M (all registered parameters)")
    print(f"FPS: {result['fps']:.2f}; batch latency: {result['batch_latency_ms_mean']:.2f} ms; batch={cli.batch_size}")
    if 'flops' in result:
        print(f"GFLOPs/image: {result['flops']['gflops_per_image']:.3f}; partial={result['flops']['partial']}")
    print(f'Report: {path}')


if __name__ == '__main__':
    main()
