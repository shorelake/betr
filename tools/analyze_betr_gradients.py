"""Measure loss-gradient conflicts on shared backbone parameters, without updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from dev_models.matcher import build_dense_aux_matcher
from tools.betr_common import load_betr
from tools.analyze_betr_attention import load_image, write_csv, bootstrap
from util.misc import nested_tensor_from_tensor_list


def gradient_metrics(left, right, indices):
    dot=0.; nl=0.; nr=0.
    for i in indices:
        a,b=left[i],right[i]
        if a is not None: nl+=float(a.double().square().sum())
        if b is not None: nr+=float(b.double().square().sum())
        if a is not None and b is not None: dot+=float((a.double()*b.double()).sum())
    l,r=np.sqrt(nl),np.sqrt(nr)
    return dict(left_norm=float(l),right_norm=float(r),cosine=dot/(l*r) if l*r>1e-20 else None,
                norm_ratio_right_left=float(r/l) if l>1e-20 else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',required=True,help='LABEL=CHECKPOINT')
    p.add_argument('--coco-path',default='../det_data/coco'); p.add_argument('--image-ids',type=int,nargs='+')
    p.add_argument('--sample-size',type=int,default=20); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--aux-weight',type=float,default=2.)
    p.add_argument('--output-dir',default='workdir/betr_gradients'); p.add_argument('--device',default='cuda')
    cli=p.parse_args()
    if cli.batch_size<1 or cli.sample_size<0 or cli.aux_weight<=0: p.error('Invalid batch/sample size or auxiliary weight')
    out=Path(cli.output_dir); out.mkdir(parents=True,exist_ok=True)
    coco=COCO(str(Path(cli.coco_path)/'annotations/instances_val2017.json'))
    ids=cli.image_ids or sorted(np.random.default_rng(cli.seed).choice(sorted(coco.imgs),min(cli.sample_size or len(coco.imgs),len(coco.imgs)),replace=False).tolist())
    (out/'image_ids.json').write_text(json.dumps(ids)); rows=[]; metadata={}
    for spec in cli.model:
        label,path=spec.split('=',1)
        if label in metadata: raise ValueError('Labels must be unique')
        model,criterion,_,args,meta=load_betr(path,cli.device,return_criterion=True); metadata[label]=meta
        # Isolate main objectives; auxiliary objectives are computed explicitly below.
        criterion.dense_aux_loss=None
        criterion.kd_from_dec=False
        gt_matcher=build_dense_aux_matcher(args)
        named=[(n,p) for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
        params=[p for _,p in named]
        groups={'backbone':list(range(len(params)))}
        for i,(name,_) in enumerate(named):
            stage=name.split('body.',1)[-1].split('.')
            key='stage_'+stage[1] if stage[0]=='layers' and len(stage)>1 else 'stem_or_norm'
            groups.setdefault(key,[]).append(i)
        if not params: raise ValueError('No trainable backbone parameters')
        for start in range(0,len(ids),cli.batch_size):
            tensors=[]; targets=[]; batch_ids=ids[start:start+cli.batch_size]
            for iid in batch_ids:
                _,tensor,target,_=load_image(coco,cli.coco_path,iid,cli.device); tensors.append(tensor); targets.append(target)
            outputs=model([nested_tensor_from_tensor_list(tensors)])
            losses=criterion(outputs,targets)
            decoder=outputs['pred_logits'].sum()*0
            early=decoder
            for name,value in losses.items():
                if name.startswith('loss_ce'): weight=args.dense_cls_loss_coef if name.endswith('_enc') else args.cls_loss_coef
                elif name.startswith('loss_bbox'): weight=args.dense_bbox_loss_coef if name.endswith('_enc') else args.bbox_loss_coef
                elif name.startswith('loss_giou'): weight=args.dense_giou_loss_coef if name.endswith('_enc') else args.giou_loss_coef
                else: continue
                if name.endswith('_enc'): early=early+value*weight
                else: decoder=decoder+value*weight
            enc=outputs['enc_outputs']; indices,_=gt_matcher(enc,targets)
            npos=max(sum(len(x[0]) for x in indices),1)
            gt=criterion.loss_labels(enc,targets,indices,npos,log=False,enc_outputs=True)['loss_ce']*cli.aux_weight
            objectives={'decoder':decoder,'early_o2o':early,'gt_focal':gt}
            if enc['pred_mask'] is not None:
                objectives['a2f']=criterion.loss_mask_prediction(enc)['loss_mask_pred']*cli.aux_weight
            grads={}
            for name,value in objectives.items():
                grad=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                # Offload to CPU to keep four large gradient vectors out of GPU memory.
                grads[name]=[g.detach().cpu() if g is not None else None for g in grad]
            for left,right in [('decoder','gt_focal'),('early_o2o','gt_focal'),('decoder','a2f'),('early_o2o','a2f')]:
                if right not in grads: continue
                for group,indices_group in groups.items():
                    rows.append(dict(model=label,batch=start//cli.batch_size,image_ids=' '.join(map(str,batch_ids)),
                                     group=group,left=left,right=right,left_loss=float(objectives[left].detach()),right_loss=float(objectives[right].detach()),
                                     **gradient_metrics(grads[left],grads[right],indices_group)))
            del outputs,losses,enc,objectives,grads,decoder,early,gt,value,grad
            print(f'{label}: {start+len(batch_ids)}/{len(ids)}',flush=True)
        del model,criterion,params,named
        if cli.device.startswith('cuda'): torch.cuda.empty_cache()
    write_csv(out/'gradients.csv',rows)
    summary=[]
    for key in sorted({(r['model'],r['group'],r['left'],r['right']) for r in rows}):
        selected=[r for r in rows if (r['model'],r['group'],r['left'],r['right'])==key and r['cosine'] is not None]
        if not selected: continue
        record=dict(zip(('model','group','left','right'),key))
        for metric in ['cosine','norm_ratio_right_left']:
            result=bootstrap([r[metric] for r in selected],seed=cli.seed); result['n_batches']=result.pop('n_images'); record[metric]=result
        record['negative_cosine_fraction']=float(np.mean([r['cosine']<0 for r in selected])); summary.append(record)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    from matplotlib import pyplot as plt
    for group in sorted({r['group'] for r in summary}):
        selected=[r for r in summary if r['group']==group]
        fig,axes=plt.subplots(1,2,figsize=(13,max(3,len(selected)*.4)))
        for i,row in enumerate(selected):
            mean=row['cosine']['mean']; lo,hi=row['cosine']['ci95']
            axes[0].errorbar(mean,i,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o')
            axes[1].barh(i,row['negative_cosine_fraction'])
        names=[f"{r['model']} {r['left']} vs {r['right']}" for r in selected]
        for ax in axes: ax.set_yticks(range(len(names)),names)
        axes[0].axvline(0,color='gray'); axes[0].set(xlabel='Gradient cosine (95% bootstrap CI)',xlim=(-1,1))
        axes[1].set(xlabel='Fraction of negative cosine',xlim=(0,1))
        fig.suptitle(group); fig.tight_layout(); fig.savefig(out/f'{group}_conflict.png',dpi=160); fig.savefig(out/f'{group}_conflict.pdf'); plt.close(fig)
    (out/'metadata.json').write_text(json.dumps(dict(options=vars(cli),models=metadata,
        protocol='Eval mode with gradients, deterministic val transform, no optimizer updates; unweighted main groups use checkpoint loss coefficients; auxiliary weight overridden explicitly; KD and legacy O2M IoU excluded. GT focal computed counterfactually on every checkpoint; A2F only if head exists.'),indent=2))
    print(f'Results: {out}')
if __name__=='__main__': main()
