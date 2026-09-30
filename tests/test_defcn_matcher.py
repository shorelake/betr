import unittest
from types import SimpleNamespace
import torch
from dev_models.matcher import build_dense_aux_matcher, DeFCNAuxMatcher


class DeFCNTests(unittest.TestCase):
    def inputs(self):
        outputs = dict(pred_logits=torch.logit(torch.tensor([[[.99,.01],[.1,.01],[.1,.01],[.1,.01]]])),
                       pred_boxes=torch.tensor([[[.5,.5,1.,1.]]*4]),
                       spatial_shapes=torch.tensor([[2,2]]), strides=[8],
                       pred_filters=torch.full((1,4,1), -100.))
        targets = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5,.5,1.,1.]]), size=torch.tensor([16,16]))]
        return outputs, targets

    def test_quality_threshold_and_filter_independence(self):
        m = build_dense_aux_matcher(SimpleNamespace(dense_aux_loss='gt-defcn'))
        self.assertIsInstance(m, DeFCNAuxMatcher)
        o,t = self.inputs()
        self.assertEqual(m(o,t)[0][0][0].tolist(), [0])
        o['pred_filters'].fill_(100)
        self.assertEqual(m(o,t)[0][0][0].tolist(), [0])

    def test_conflict_uses_max_quality(self):
        o,t = self.inputs()
        o['pred_logits'][0,0,1] = torch.logit(torch.tensor(.8))
        t[0]['labels'] = torch.tensor([0,1])
        t[0]['boxes'] = t[0]['boxes'].repeat(2,1)
        src,dst = DeFCNAuxMatcher()(o,t)[0][0]
        self.assertEqual(src.tolist(), [0])
        self.assertEqual(dst.tolist(), [0])

    def test_empty_padding_and_single_candidate(self):
        o,t = self.inputs()
        o['mask_flatten'] = torch.tensor([[False,True,True,True]])
        self.assertEqual(DeFCNAuxMatcher()(o,t)[0][0][0].tolist(), [0])
        o['mask_flatten'].fill_(True)
        self.assertEqual(DeFCNAuxMatcher()(o,t)[0][0][0].numel(), 0)
        t[0]['labels'] = torch.empty(0,dtype=torch.long)
        t[0]['boxes'] = torch.empty(0,4)
        indices,ious = DeFCNAuxMatcher()(o,t)
        self.assertEqual(indices[0][0].numel(), 0)
        self.assertEqual(ious.sum().item(), 0)

    def test_outside_gt_is_rejected(self):
        o,t = self.inputs()
        t[0]['boxes'] = torch.tensor([[.75,.75,.4,.4]])
        o['pred_boxes'][:] = t[0]['boxes']
        self.assertEqual(DeFCNAuxMatcher()(o,t)[0][0][0].numel(), 0)

if __name__ == '__main__':
    unittest.main()
