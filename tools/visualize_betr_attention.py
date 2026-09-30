"""Export figure 3.6 panels and comparable BETR attention maps without training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

from datasets.coco import make_coco_transforms
from util.misc import nested_tensor_from_tensor_list
from tools.betr_common import load_betr, a2f_maps


def project(flat, shapes, size, binary=False):
    maps = []
    start = 0
    for h, w in shapes.tolist():
        field = flat[start:start+h*w].reshape(1, 1, h, w)
        if binary:
            field = F.interpolate(field, size=size, mode='nearest')
        else:
            field = F.interpolate(field, size=size, mode='bilinear', align_corners=False)
        maps.append(field[0, 0])
        start += h*w
    return torch.stack(maps).mean(0).cpu().numpy()


def render(cli):
    if not 0 < cli.ratio <= 1:
        raise ValueError('ratio must be in (0, 1]')
    annotations = Path(cli.coco_path) / 'annotations/instances_val2017.json'
    coco = COCO(str(annotations))
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels, checkpoints = [], []
    for spec in cli.model:
        label, sep, path = spec.partition('=')
        if not sep or not label or Path(label).name != label:
            raise ValueError('--model must be a simple LABEL=CHECKPOINT')
        labels.append(label)
        checkpoints.append(path)
    if len(set(labels)) != len(labels):
        raise ValueError('Model labels must be unique')
    images = {}
    for image_id in cli.image_ids:
        info = coco.loadImgs(image_id)[0]
        original = Image.open(Path(cli.coco_path) / 'val2017' / info['file_name']).convert('RGB')
        tensor, _ = make_coco_transforms('val')(original, None)
        images[image_id] = (original, tensor, {})
    for label, checkpoint in zip(labels, checkpoints):
        model, _, args, metadata = load_betr(checkpoint, cli.device, cli.enc_layers, cli.backbone)
        metadata.update({'ratio_for_display': cli.ratio, 'image_ids': cli.image_ids,
                         'aggregation': 'all decoder queries, layers, heads; mean resized pyramid levels',
                         'projection': 'repository util.dam, same as training A2F target',
                         'normalization': 'cross attention normalized jointly across models per image; probabilities fixed 0..1'})
        (output / f'{label}_metadata.json').write_text(json.dumps(metadata, indent=2))
        with torch.inference_mode():
            for image_id, (original, tensor, results) in images.items():
                enc = model([nested_tensor_from_tensor_list([tensor.to(cli.device)])])['enc_outputs']
                attention, binary = a2f_maps(enc, cli.ratio)
                foreground = enc['pred_logits'].sigmoid().amax(-1)
                if enc['pred_filters'] is not None:
                    foreground = foreground * enc['pred_filters'].sigmoid().squeeze(-1)
                shapes = enc['spatial_shapes']
                size = (original.height, original.width)
                maps = {'cross_attention': project(attention[0], shapes, size),
                        'a2f_target': project(binary[0], shapes, size, binary=True),
                        'foreground_score': project(foreground[0], shapes, size)}
                if enc['pred_mask'] is not None:
                    maps['a2f_prediction'] = project(enc['pred_mask'][0, :, 0].sigmoid(), shapes, size)
                folder = output / str(image_id)
                folder.mkdir(exist_ok=True)
                raw = {'spatial_shapes': shapes.cpu().numpy(),
                       'cross_attention_flat': attention[0].cpu().numpy(),
                       'a2f_target_flat': binary[0].cpu().numpy(),
                       'foreground_score_flat': foreground[0].cpu().numpy()}
                if enc['pred_mask'] is not None:
                    raw['a2f_prediction_flat'] = enc['pred_mask'][0, :, 0].sigmoid().cpu().numpy()
                np.savez_compressed(folder / f'{label}_raw.npz', **raw)
                results[label] = maps
        del model
        if cli.device.startswith('cuda'):
            torch.cuda.empty_cache()
    for image_id, (original, tensor, results) in images.items():
        folder = output / str(image_id)
        boxes = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        cross_max = max(float(m['cross_attention'].max()) for m in results.values()) or 1.
        columns = [('cross_attention', 'Decoder cross attention'), ('a2f_target', 'Top-k teacher target'),
                   ('foreground_score', 'Early class score'), ('a2f_prediction', 'A2F predictor')]
        fig, axes = plt.subplots(len(labels), 5, figsize=(18, 4*len(labels)), squeeze=False)
        for row, label in enumerate(labels):
            ax = axes[row, 0]
            ax.imshow(original)
            for ann in boxes:
                x, y, w, h = ann['bbox']
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1))
            ax.set_title(f'{label}: image + GT')
            ax.axis('off')
            for col, (key, title) in enumerate(columns, start=1):
                ax = axes[row, col]
                ax.axis('off')
                ax.set_title(title)
                if key not in results[label]:
                    ax.text(.5, .5, 'No A2F prediction head', ha='center', va='center')
                    continue
                field = results[label][key]
                vmax = cross_max if key == 'cross_attention' else 1.
                ax.imshow(field, cmap='gray', vmin=0, vmax=vmax)
                plt.imsave(folder / f'{label}_{key}.png', field, cmap='gray', vmin=0, vmax=vmax)
                # Preserve every native pyramid level, including the binary target.
                raw = np.load(folder / f'{label}_raw.npz')
                start = 0
                for level, (h, w) in enumerate(raw['spatial_shapes']):
                    native = raw[key + '_flat'][start:start+h*w].reshape(h, w)
                    plt.imsave(folder / f'{label}_{key}_level{level}.png', native,
                               cmap='gray', vmin=0, vmax=vmax)
                    start += h*w
        fig.tight_layout()
        fig.savefig(folder / 'comparison.png', dpi=180)
        fig.savefig(folder / 'comparison.pdf')
        plt.close(fig)
        # Figure 3.6 uses a binary teacher map above its predicted response.
        for label in labels:
            maps = results[label]
            if 'a2f_prediction' not in maps:
                continue
            fig = plt.figure(figsize=(10, 6))
            grid = fig.add_gridspec(2, 2)
            ax = fig.add_subplot(grid[:, 0])
            ax.imshow(original)
            for ann in boxes:
                x, y, w, h = ann['bbox']
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red'))
            ax.set_title('Input + ground truth')
            ax.axis('off')
            for row, key in enumerate(('a2f_target', 'a2f_prediction')):
                ax = fig.add_subplot(grid[row, 1])
                # Native finest level avoids averaging binary values across scales.
                with np.load(folder / f'{label}_raw.npz') as raw:
                    h, w = raw['spatial_shapes'][0]
                    field = raw[key + '_flat'][:h*w].reshape(h, w)
                ax.imshow(field, cmap='gray', vmin=0, vmax=1)
                ax.set_title('Binary cross attention' if row == 0 else 'Early A2F prediction')
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(folder / f'{label}_figure3_6.png', dpi=180)
            fig.savefig(folder / f'{label}_figure3_6.pdf')
            plt.close(fig)
    print(f'Figures and raw maps: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', required=True, help='LABEL=CHECKPOINT; repeat for comparison')
    parser.add_argument('--image-ids', type=int, nargs='+', required=True)
    parser.add_argument('--coco-path', default='../det_data/coco')
    parser.add_argument('--output-dir', default='workdir/attention')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ratio', type=float, default=.2)
    parser.add_argument('--enc-layers', type=int)
    parser.add_argument('--backbone', choices=['swin_nano', 'swin_tiny', 'swin_small', 'swin_base'])
    render(parser.parse_args())


if __name__ == '__main__':
    main()
