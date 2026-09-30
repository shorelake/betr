import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
from tools.analyze_betr_attention import top_mask, region_metrics, summarize
from tools.analyze_betr_gradients import gradient_metrics


class MechanismTests(unittest.TestCase):
    def test_topk_padding(self):
        mask=top_mask(torch.tensor([1.,4.,3.,100.]),torch.tensor([True,True,True,False]),.2)
        self.assertEqual(mask.tolist(),[False,True,False,False])
        self.assertEqual(int(top_mask(torch.ones(4),torch.zeros(4,dtype=torch.bool),.2).sum()),0)

    def test_area_normalized_density(self):
        v=np.ones(10); bits=np.ones(10)
        center=np.arange(10)<2; edge=np.arange(10)>=2
        result=region_metrics(v,bits,center,edge,1)
        self.assertEqual(result['edge_center_ratio'],1.)
        self.assertEqual(result['coverage_difference'],0.)
        self.assertIsNone(region_metrics(v,bits,center,edge,3))

    def test_gradient_unused_and_opposed(self):
        result=gradient_metrics([torch.tensor([1.,0.]),None],[torch.tensor([-1.,0.]),None],[0,1])
        self.assertEqual(result['cosine'],-1.)
        self.assertEqual(result['norm_ratio_right_left'],1.)
        self.assertIsNone(gradient_metrics([None],[None],[0])['cosine'])

    def test_image_paired_bootstrap(self):
        rows=[]
        for label,value in [('a',1.),('b',2.)]:
            for iid in [1,2]:
                for aid in [1,2]:
                    rows.append(dict(model=label,image_id=iid,annotation_id=aid,level=0,region='box',map='query_attention',size='large',center_mean=1.,edge_mean=value,edge_minus_center=value-1,coverage_difference=value-1,edge_center_ratio=value))
        with tempfile.TemporaryDirectory() as folder:
            summarize(rows,Path(folder))
            paired=json.loads((Path(folder)/'paired_differences.json').read_text())
            self.assertTrue(paired)
            self.assertEqual(paired[0]['mean'],1.)
            self.assertEqual(paired[0]['n_images'],2)

if __name__=='__main__': unittest.main()
