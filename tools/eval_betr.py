"""Evaluate a BETR checkpoint on COCO val2017 with official pycocotools."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from pycocotools.cocoeval import COCOeval

from datasets import build_dataset
from util.misc import collate_fn
from tools.betr_common import add_checkpoint_options, load_betr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_checkpoint_options(parser)
    parser.add_argument('--output-dir', default='workdir/betr_eval')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; 0 evaluates all images')
    cli = parser.parse_args()
    if cli.limit < 0 or cli.batch_size < 1:
        parser.error('limit must be nonnegative and batch-size positive')
    model, post, args, metadata = load_betr(cli.checkpoint, cli.device, cli.enc_layers, cli.backbone)
    args.coco_path = cli.coco_path
    args.dataset_file = args.dataset = 'coco'
    dataset = build_dataset('val', args)
    coco = dataset.coco
    ids = dataset.ids[:cli.limit] if cli.limit else dataset.ids
    selected = Subset(dataset, range(len(ids)))
    loader = DataLoader(selected, batch_size=cli.batch_size, num_workers=cli.workers, collate_fn=collate_fn)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    elapsed = 0.0
    timed_images = 0
    with torch.inference_mode():
        for step, (samples, targets) in enumerate(loader):
            samples = samples.to(cli.device)
            if cli.device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model([samples])
            if cli.device.startswith('cuda'):
                torch.cuda.synchronize()
            duration = time.perf_counter() - start
            if step >= 10:
                elapsed += duration
                timed_images += len(targets)
            sizes = torch.stack([t['orig_size'] for t in targets]).to(cli.device)
            for target, pred in zip(targets, post['bbox'](result, sizes)):
                boxes = pred['boxes'].cpu()
                boxes[:, 2:] -= boxes[:, :2]
                for box, score, label in zip(boxes.tolist(), pred['scores'].tolist(), pred['labels'].tolist()):
                    predictions.append({'image_id': int(target['image_id']), 'category_id': label,
                                        'bbox': box, 'score': score})
            if step % 100 == 0:
                print(f'Evaluated {min((step + 1) * cli.batch_size, len(ids))}/{len(ids)} images', flush=True)
    (output / 'predictions.json').write_text(json.dumps(predictions))
    evaluator = COCOeval(coco, coco.loadRes(predictions), 'bbox')
    evaluator.params.imgIds = list(ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    metadata.update({'metrics_percent': dict(zip(names, (evaluator.stats * 100).tolist())),
                     'num_images': len(ids), 'subset_only': bool(cli.limit),
                     'validation_resize': 'short side 800, max side 1333',
                     'batch_size': cli.batch_size, 'timing_warmup_batches': 10,
                     'model_only_images_per_second': timed_images / elapsed if elapsed else None,
                     'timing_note': 'Excludes loading, transforms, postprocessing and COCOeval; not paper V100 FPS.'})
    (output / 'metrics.json').write_text(json.dumps(metadata, indent=2))
    print(f'Results: {output / "metrics.json"}')


if __name__ == '__main__':
    main()
