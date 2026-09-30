import unittest

import torch

from main import get_args_parser, get_model
from tools.betr_common import a2f_maps
from dev_models.matcher import build_dense_matcher, SpatialPriorHungarianMatcher
from util.misc import nested_tensor_from_tensor_list


class ReproductionTests(unittest.TestCase):
    def test_disabled_aux_and_global_matcher(self):
        args = get_args_parser().parse_args(['--dense_aux_loss', 'none', '--spatial_prior_radius', 'inf', '--no_kd_from_dec'])
        self.assertIsNone(args.dense_aux_loss)
        self.assertFalse(args.kd_from_dec)
        matcher = build_dense_matcher(args)
        self.assertIsInstance(matcher, SpatialPriorHungarianMatcher)
        self.assertEqual(matcher.radius, float('inf'))

    def test_target_excludes_padding(self):
        enc = {'spatial_shapes': torch.tensor([[2, 2]]), 'level_start_index': torch.tensor([0]),
               'sampling_locations_dec': torch.tensor([[[[[[[.25, .25], [.75, .75]]]]]]]),
               'attn_weights_dec': torch.ones(1, 1, 1, 1, 1, 2),
               'mask_flatten': torch.tensor([[False, False, False, True]])}
        _, target = a2f_maps(enc, 1.)
        self.assertEqual(target.tolist(), [[1., 1., 1., 0.]])

    def test_center_prior_and_empty_targets(self):
        args = get_args_parser().parse_args(['--spatial_prior_radius', '0'])
        matcher = build_dense_matcher(args)
        outputs = {'pred_logits': torch.zeros(2, 4, 2),
                   'pred_boxes': torch.tensor([[[.25, .25, .3, .3]] * 4] * 2),
                   'pred_filters': None, 'strides': [8], 'spatial_shapes': torch.tensor([[2, 2]])}
        targets = [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[.75, .75, .4, .4]]),
                    'size': torch.tensor([16, 16])},
                   {'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0, 4),
                    'size': torch.tensor([16, 16])}]
        matched = matcher(outputs, targets)
        self.assertEqual(matched[0][0].tolist(), [3])
        self.assertEqual(matched[1][0].numel(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA extension required')
    def test_auxiliary_modes_backward(self):
        for mode in ('none', 'gt', 'gt-defcn', 'dam'):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                args = get_args_parser().parse_args(['--dense_aux_loss', mode, '--enc_layers', '0',
                                                     '--num_queries', '10', '--no_kd_from_dec'])
                args.pretrained_path = None
                args.a2f_ratio = .6
                model, criterion, _ = get_model(args)
                model.cuda().train()
                samples = nested_tensor_from_tensor_list([torch.rand(3, 64, 64, device='cuda') for _ in range(2)])
                targets = [{'labels': torch.tensor([1], device='cuda'),
                            'boxes': torch.tensor([[.5, .5, .5, .5]], device='cuda'),
                            'size': torch.tensor([64, 64], device='cuda')} for _ in range(2)]
                outputs = model([samples, targets])
                losses = criterion(outputs, targets)
                aux = {k for k in losses if k.endswith('_enc_aux')}
                expected = {'none': set(), 'gt': {'loss_ce_enc_aux'}, 'gt-defcn': {'loss_ce_enc_aux'}, 'dam': {'loss_mask_pred_enc_aux'}}
                self.assertEqual(aux, expected[mode])
                loss = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                del model, criterion, outputs, losses, loss, grads


if __name__ == '__main__':
    unittest.main()
