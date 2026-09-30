"""Instance-query attention, box/contour statistics and paired image bootstrap."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch

from datasets.coco import ConvertCocoPolysToMask, make_coco_transforms
from pycocotools.coco import COCO
from dev_models.matcher import build_matcher, build_dense_aux_matcher
from tools.betr_common import load_betr, a2f_maps
from util.dam import attn_map_to_flat_grid
from util.misc import nested_tensor_from_tensor_list


def top_mask(values, valid, ratio):
    """Repository convention: floor(valid_count * ratio) + 1, capped."""
    result = torch.zeros_like(valid)
    n = int(valid.sum())
    if n:
        indices = values.masked_fill(~valid, -float('inf')).topk(min(int(n*ratio)+1, n)).indices
        result[indices] = True
    return result


def region_metrics(values, binary, center, edge, min_points):
    nc, ne = int(center.sum()), int(edge.sum())
    if min(nc, ne) < min_points:
        return None
    c, e = float(values[center].mean()), float(values[edge].mean())
    cc, ec = float(binary[center].mean()), float(binary[edge].mean())
    return dict(center_points=nc, edge_points=ne, center_mean=c, edge_mean=e,
                edge_minus_center=e-c, edge_center_ratio=e/c if c > 1e-12 else None,
                center_coverage=cc, edge_coverage=ec, coverage_difference=ec-cc)


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, seed=42, repeats=1000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
    return dict(mean=float(values.mean()), ci95=np.percentile(boot, [2.5, 97.5]).tolist(), n_images=len(values))


def summarize(rows, out):
    # Average instances within each image, then resample images (not correlated instances).
    summary = []
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('model','level','region','map','size'))
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        for metric in ('center_mean','edge_mean','edge_minus_center','coverage_difference','edge_center_ratio'):
            per_image = {}
            for row in group:
                if row[metric] is not None and np.isfinite(row[metric]):
                    per_image.setdefault(row['image_id'], []).append(row[metric])
            if per_image:
                summary.append(dict(zip(('model','level','region','map','size'),key), metric=metric,
                                    **bootstrap([np.mean(v) for v in per_image.values()])))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    paired = []
    labels = sorted({r['model'] for r in rows})
    # Match the exact same annotation and region; then bootstrap image differences.
    for left_idx, left in enumerate(labels):
        for right in labels[left_idx+1:]:
            lookup = {(r['image_id'],r['annotation_id'],r['level'],r['region'],r['map'],r['size']):r
                      for r in rows if r['model']==left}
            differences = {}
            for row in rows:
                if row['model'] != right:
                    continue
                key = tuple(row[k] for k in ('image_id','annotation_id','level','region','map','size'))
                if key not in lookup:
                    continue
                for metric in ('edge_minus_center','coverage_difference'):
                    group = (key[2:], metric)
                    differences.setdefault(group, {}).setdefault(key[0], []).append(row[metric]-lookup[key][metric])
            for (key, metric), images in differences.items():
                paired.append(dict(left=left,right=right, difference='right minus left',level=key[0],region=key[1],map=key[2],size=key[3],metric=metric,
                                   **bootstrap([np.mean(v) for v in images.values()])))
    (out/'paired_differences.json').write_text(json.dumps(paired, indent=2))


def load_image(coco, root, iid, device):
    info = coco.loadImgs([iid])[0]
    image = Image.open(Path(root)/'val2017'/info['file_name']).convert('RGB')
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid],iscrowd=False))
    # Exactly reproduce converter filtering, retaining annotation IDs.
    anns = [a for a in anns if min(a['bbox'][0]+a['bbox'][2],image.width)>max(a['bbox'][0],0)
            and min(a['bbox'][1]+a['bbox'][3],image.height)>max(a['bbox'][1],0)]
    _, target = ConvertCocoPolysToMask()(image, {'image_id':iid,'annotations':anns})
    tensor, target = make_coco_transforms('val')(image, target)
    target = {k:v.to(device) for k,v in target.items()}
    return image, tensor.to(device), target, anns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',action='append',required=True,help='LABEL=CHECKPOINT (repeat)')
    parser.add_argument('--coco-path',default='../det_data/coco')
    parser.add_argument('--image-ids',type=int,nargs='+')
    parser.add_argument('--sample-size',type=int,default=100,help='Seeded random sample; 0=all val images')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-dir',default='workdir/betr_mechanism')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--ratio',type=float,default=.2)
    parser.add_argument('--center',type=float,default=.5)
    parser.add_argument('--edge',type=float,default=.8)
    parser.add_argument('--contour-width',type=float,default=.1,help='Fraction of sqrt(mask area), minimum 1 original pixel')
    parser.add_argument('--min-points',type=int,default=4)
    parser.add_argument('--no-contours',action='store_true')
    parser.add_argument('--plot-images',type=int,default=3)
    parser.add_argument('--plot-instances',type=int,default=2)
    parser.add_argument('--decoder-layer',type=int,default=-1,help='-1 sums all decoder layers; otherwise zero-based')
    cli = parser.parse_args()
    if not (0<cli.ratio<=1 and 0<cli.center<cli.edge<1 and cli.min_points>0 and cli.contour_width>0 and cli.sample_size>=0):
        parser.error('Invalid ratio, regions, sample size, or minimum points')
    out = Path(cli.output_dir); out.mkdir(parents=True,exist_ok=True)
    coco = COCO(str(Path(cli.coco_path)/'annotations/instances_val2017.json'))
    ids = cli.image_ids or sorted(np.random.default_rng(cli.seed).choice(sorted(coco.imgs),size=min(cli.sample_size or len(coco.imgs),len(coco.imgs)),replace=False).tolist())
    (out/'image_ids.json').write_text(json.dumps(ids))
    rows, curves, alignments, skipped, metadata, panels = [], [], [], [], {}, {}
    for spec in cli.model:
        label, path = spec.split('=',1)
        if not label or label in metadata or Path(label).name!=label:
            parser.error('Labels must be unique simple filenames')
        model, _, args, meta = load_betr(path,cli.device)
        metadata[label]=meta
        matcher, gt_matcher = build_matcher(args), build_dense_aux_matcher(args)
        with torch.inference_mode():
            for image_index, iid in enumerate(ids):
                image, tensor, target, anns = load_image(coco,cli.coco_path,iid,cli.device)
                outputs = model([nested_tensor_from_tensor_list([tensor])])
                enc=outputs['enc_outputs']; shapes=enc['spatial_shapes'].tolist()
                valid=~enc['mask_flatten'][0]
                _, teacher=a2f_maps(enc,cli.ratio)
                qids,tids=matcher(outputs,[target])[0]
                pairs={int(t):int(q) for q,t in zip(qids,tids)}
                gt_indices,_=gt_matcher(enc,[target])
                foreground=enc['pred_logits'][0].sigmoid()
                if enc['pred_filters'] is not None:
                    foreground=foreground*enc['pred_filters'][0].sigmoid()
                prediction=enc['pred_mask']
                if prediction is not None:
                    pred=prediction[0,:,0].sigmoid()
                    chosen=top_mask(pred,valid,cli.ratio)
                    truth=teacher[0].bool()
                    for level,(h,w) in enumerate(shapes):
                        st=int(enc['level_start_index'][level]); sl=slice(st,st+h*w)
                        inter=int((chosen[sl]&truth[sl]).sum()); union=int((chosen[sl]|truth[sl]).sum())
                        alignments.append(dict(model=label,image_id=iid,level=level,iou=inter/union if union else None,
                                               precision=inter/int(chosen[sl].sum()) if chosen[sl].any() else None,
                                               recall=inter/int(truth[sl].sum()) if truth[sl].any() else None))
                for ti,ann in enumerate(anns):
                    if ti not in pairs:
                        skipped.append(dict(model=label,image_id=iid,annotation_id=ann['id'],reason='unmatched GT')); continue
                    qi=pairs[ti]
                    loc=enc['sampling_locations_dec'][:,:,qi:qi+1]
                    weights=enc['attn_weights_dec'][:,:,qi:qi+1]
                    if cli.decoder_layer>=0:
                        if cli.decoder_layer>=loc.shape[1]: raise ValueError('decoder-layer out of range')
                        loc=loc[:,cli.decoder_layer:cli.decoder_layer+1]; weights=weights[:,cli.decoder_layer:cli.decoder_layer+1]
                    attention=attn_map_to_flat_grid(enc['spatial_shapes'],enc['level_start_index'],loc,weights).sum((1,2))[0]
                    binary=top_mask(attention,valid,cli.ratio)
                    assignment=torch.zeros_like(valid)
                    src,tgt=gt_indices[0]; assignment[src[tgt==ti].to(valid.device)]=True
                    class_score=foreground[:,int(target['labels'][ti])]
                    class_binary=top_mask(class_score,valid,cli.ratio)
                    cx,cy,bw,bh=target['boxes'][ti].cpu().tolist()
                    size='small' if ann['area']<32**2 else 'medium' if ann['area']<96**2 else 'large'
                    if not cli.no_contours:
                        mask=coco.annToMask(ann).astype(bool); distance=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
                        band=max(1.,cli.contour_width*np.sqrt(mask.sum()))
                    for level,(h,w) in enumerate(shapes):
                        st=int(enc['level_start_index'][level]); sl=slice(st,st+h*w)
                        # Use actual feature stride, not padded tensor dimensions, to locate grid centers.
                        stride=enc['strides'][level]; th,tw=target['size'].cpu().tolist()
                        yy,xx=np.mgrid[:h,:w]; x=(xx+.5)*stride/tw; y=(yy+.5)*stride/th
                        r=np.maximum(abs(x-cx)/max(bw/2,1e-12),abs(y-cy)/max(bh/2,1e-12))
                        ok=valid[sl].cpu().numpy().reshape(h,w)&(x<1)&(y<1)
                        regions={'box':(ok&(r<=cli.center),ok&(r>cli.edge)&(r<=1))}
                        if not cli.no_contours:
                            px=np.clip((x*image.width).astype(int),0,image.width-1); py=np.clip((y*image.height).astype(int),0,image.height-1)
                            d=distance[py,px]; inside=mask[py,px]&ok
                            regions['contour']=(inside&(d>band),inside&(d<=band))
                        fields={'query_attention':(attention,binary),'class_response':(class_score,class_binary),
                                'gt_assignment':(assignment.float(),assignment),'a2f_teacher':(teacher[0],teacher[0].bool())}
                        if prediction is not None: fields['a2f_prediction']=(pred,chosen)
                        cache={}
                        for name,(values,bits) in fields.items():
                            a=values[sl].cpu().numpy().reshape(h,w); b=bits[sl].float().cpu().numpy().reshape(h,w); cache[name]=a
                            for region,(center,edge) in regions.items():
                                record=dict(model=label,image_id=iid,annotation_id=ann['id'],query=qi,level=level,size=size,region=region,map=name)
                                metrics=region_metrics(a,b,center,edge,cli.min_points)
                                if metrics is None:
                                    skipped.append(dict(**record,reason='insufficient region grid points')); continue
                                rows.append(dict(**record,**metrics))
                            for bin_id in range(10):
                                select=ok&(r>=bin_id/10)&(r<(bin_id+1)/10)
                                if select.any(): curves.append(dict(model=label,image_id=iid,annotation_id=ann['id'],level=level,size=size,map=name,bin=bin_id,mean=float(a[select].mean()),points=int(select.sum())))
                        if level==0 and image_index<cli.plot_images and ti<cli.plot_instances:
                            panels.setdefault((iid,ann['id']),{})[label]=(image.copy(),ann,cache,binary[sl].cpu().numpy().reshape(h,w),regions['box'],(tw/stride,th/stride))
                if image_index%10==0: print(f'{label}: {image_index+1}/{len(ids)}',flush=True)
        del model
        if cli.device.startswith('cuda'): torch.cuda.empty_cache()
    write_csv(out/'instances.csv',rows); write_csv(out/'radial.csv',curves); write_csv(out/'alignment.csv',alignments); write_csv(out/'excluded.csv',skipped)
    (out/'metadata.json').write_text(json.dumps(dict(options=vars(cli),models=metadata),indent=2))
    summarize(rows,out)
    for (iid,aid),models in panels.items():
        fig,axes=plt.subplots(len(models),6,figsize=(20,3.5*len(models)),squeeze=False)
        vmax=max(float(item[2]['query_attention'].max()) for item in models.values()) or 1.
        for row,(label,(image,ann,cache,bits,regions,grid_extent)) in enumerate(models.items()):
            ax=axes[row,0]; ax.imshow(image); x,y,w,h=ann['bbox']; ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,edgecolor='red')); ax.set_title(f'{label}: GT {aid}')
            for col,(key,title) in enumerate([('query_attention','Matched query attention'),('binary','Query top-k'),('class_response','GT class response'),('gt_assignment','O2M assigned points'),('a2f_prediction','A2F predictor')],1):
                ax=axes[row,col]; ax.set_title(title)
                a=bits if key=='binary' else cache.get(key)
                if a is None: ax.text(.5,.5,'Head absent',ha='center')
                else:
                    # Render boolean and continuous maps on the same explicit pixel grid.
                    a=np.asarray(a,dtype=np.float32)
                    ax.imshow(a,cmap='gray',vmin=0,vmax=vmax if key=='query_attention' else 1,
                              interpolation='nearest',origin='upper')
                    for region,color in zip(regions,['cyan','orange']):
                        if region.any() and not region.all(): ax.contour(region,levels=[.5],colors=[color],linewidths=.5)
                    ax.set_xlim(-.5,grid_extent[0]-.5)
                    ax.set_ylim(grid_extent[1]-.5,-.5)
                    ax.set_aspect('equal')
            for ax in axes[row]: ax.axis('off')
        fig.tight_layout(); fig.savefig(out/f'{iid}_{aid}_comparison.png',dpi=150); fig.savefig(out/f'{iid}_{aid}_comparison.pdf')
        # Second view: common GT crop with 25% context, without renormalizing colors.
        for row,(label,(image,ann,cache,bits,regions,grid_extent)) in enumerate(models.items()):
            x,y,w,h=ann['bbox']
            x0=max(0,x-.25*w); x1=min(image.width,x+1.25*w)
            y0=max(0,y-.25*h); y1=min(image.height,y+1.25*h)
            axes[row,0].set_xlim(x0,x1); axes[row,0].set_ylim(y1,y0)
            for ax in axes[row,1:]:
                ax.set_xlim(x0/image.width*grid_extent[0]-.5,x1/image.width*grid_extent[0]-.5)
                ax.set_ylim(y1/image.height*grid_extent[1]-.5,y0/image.height*grid_extent[1]-.5)
        fig.tight_layout(); fig.savefig(out/f'{iid}_{aid}_zoom.png',dpi=150); fig.savefig(out/f'{iid}_{aid}_zoom.pdf'); plt.close(fig)
    for level in sorted({r['level'] for r in curves}):
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,name in zip(axes,['query_attention','class_response','gt_assignment']):
            for label in metadata:
                y=[]
                for b in range(10):
                    image_values={}
                    for row in curves:
                        if row['model']==label and row['level']==level and row['map']==name and row['bin']==b: image_values.setdefault(row['image_id'],[]).append(row['mean'])
                    y.append(float(np.mean([np.mean(v) for v in image_values.values()])) if image_values else np.nan)
                ax.plot(np.arange(10)/10+.05,y,label=label)
            ax.set(title=name,xlabel='normalized box radius r',ylabel='mean per grid point'); ax.legend()
        fig.tight_layout(); fig.savefig(out/f'radial_level{level}.png',dpi=160); fig.savefig(out/f'radial_level{level}.pdf'); plt.close(fig)
    print(f'Results: {out}')

if __name__=='__main__': main()
