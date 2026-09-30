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


def resize_map(field, image):
    """Nearest-neighbor display preserves binary labels and feature-cell boundaries."""
    return np.asarray(Image.fromarray(np.asarray(field, dtype=np.float32)).resize(
        image.size, resample=Image.Resampling.NEAREST))


def extra_panels(folder, labels, image, anns):
    raws = {}
    for label in labels:
        with np.load(folder / f'{label}_raw.npz') as data:
            raws[label] = {k: data[k] for k in data.files}
    def level0(raw, key):
        h, w = raw['spatial_shapes'][0]
        return resize_map(raw[key][:h*w].reshape(h,w), image)
    def original(ax):
        ax.imshow(image)
        for ann in anns:
            x,y,w,h = ann['bbox']
            ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=.8))
        ax.axis('off')
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(folder / f'{name}.{ext}',dpi=160)
        plt.close(fig)
    keys = ['cross_attention_flat','a2f_target_flat','foreground_score_flat','a2f_prediction_flat']
    titles = ['Cross attention L0','Binary teacher L0','Early max class score L0','A2F predictor L0']
    vmax = max(float(level0(r,keys[0]).max()) for r in raws.values()) or 1.
    fig,axes = plt.subplots(len(labels),5,figsize=(18,3.6*len(labels)),squeeze=False)
    for row,label in enumerate(labels):
        original(axes[row,0]); axes[row,0].set_title(label)
        for col,(key,title) in enumerate(zip(keys,titles),1):
            ax=axes[row,col]; ax.set_title(title); ax.axis('off')
            if key in raws[label]:
                ax.imshow(level0(raws[label],key),cmap='gray',vmin=0,vmax=vmax if col==1 else 1,interpolation='nearest')
            else: ax.text(.5,.5,'Head absent',ha='center')
    save(fig,'comparison_level0')
    for label,raw in raws.items():
        if 'a2f_prediction_flat' not in raw: continue
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        original(axes[0]); axes[0].set_title(label+' / GT')
        for ax,key,title in zip(axes[1:],keys[1::2],['Binary teacher L0','A2F prediction L0']):
            ax.imshow(level0(raw,key),cmap='gray',vmin=0,vmax=1,interpolation='nearest'); ax.set_title(title); ax.axis('off')
        save(fig,label+'_panel_A_level0')
    # One panel per GT category: compare the same class, not different dedicated heads.
    for category in sorted({a['category_id'] for a in anns}):
        fig,axes=plt.subplots(len(labels),4,figsize=(15,3.6*len(labels)),squeeze=False)
        maps={label:level0(dict(raw, score=raw['class_probability'][:,category]),'score') for label,raw in raws.items()}
        upper=max(float(a.max()) for a in maps.values()) or 1.
        for row,label in enumerate(labels):
            raw=raws[label]; original(axes[row,0]); axes[row,0].set_title(label)
            for col,limit,title in [(1,1.,'Class probability [0,1]'),(2,upper,'Class probability / shared max')]:
                axes[row,col].imshow(maps[label],cmap='gray',vmin=0,vmax=limit,interpolation='nearest')
                axes[row,col].set_title(f'{title}; class {category}'); axes[row,col].axis('off')
            ax=axes[row,3]; original(ax)
            boxes=raw['proposal_boxes']; centers=boxes[:,:2]*np.array(image.size)
            ax.scatter(centers[:,0],centers[:,1],s=6,c='cyan',alpha=.6)
            # Top 30 boxes for readability; all selected centers remain visible.
            for cx,cy,w,h in boxes[:30]:
                ax.add_patch(plt.Rectangle(((cx-w/2)*image.width,(cy-h/2)*image.height),w*image.width,h*image.height,fill=False,edgecolor='cyan',linewidth=.4,alpha=.5))
            ax.set_xlim(0,image.width); ax.set_ylim(image.height,0)
            ax.set_title(f'Top-{len(boxes)} centers / top-30 boxes')
        save(fig,f'panel_B_class{category}_level0')


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
                       'foreground_score_flat': foreground[0].cpu().numpy(),
                       'class_probability': enc['pred_logits'][0].sigmoid().cpu().numpy(),
                       'valid': (~enc['mask_flatten'][0]).cpu().numpy(),
                       'proposal_indices': enc['topk_proposal'][0].cpu().numpy(),
                       'proposal_boxes': enc['pred_boxes'][0, enc['topk_proposal'][0]].cpu().numpy()}

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
        columns = [('cross_attention', 'Decoder cross attention'), ('a2f_target', 'Teacher: mean across levels (not binary)'),
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
                    plt.imsave(folder / f'{label}_{key}_level{level}.png', resize_map(native, original),
                               cmap='gray', vmin=0, vmax=vmax)
                    start += h*w
        fig.tight_layout()
        fig.savefig(folder / 'comparison.png', dpi=180)
        fig.savefig(folder / 'comparison.pdf')
        plt.close(fig)
        extra_panels(folder, labels, original, boxes)
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
                ax.imshow(resize_map(field, original), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
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
