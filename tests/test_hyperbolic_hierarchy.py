import unittest

import geoopt
import torch

from tools.hyperbolic_hierarchy import (
    _lca_depth_hyp,
    hierarchy_triplet_loss_hyp,
    prototype_affinity_hyp,
    sample_triplets_from_affinity,
    update_affinity_ema,
)
from tools.sinkhorn import sinkhorn_balanced_transport


class HyperbolicHierarchyTest(unittest.TestCase):
    def setUp(self):
        self.manifold = geoopt.PoincareBall(1.0)

    def test_lca_depth_matches_inner_collinear_point(self):
        inner = self.manifold.expmap0(torch.tensor([[0.2, 0.0]]))
        outer = self.manifold.expmap0(torch.tensor([[1.0, 0.0]]))

        depth = _lca_depth_hyp(inner, outer, self.manifold)

        torch.testing.assert_close(depth, self.manifold.dist0(inner), atol=1e-6, rtol=1e-6)

    def test_prototype_affinity_prefers_nearby_prototypes(self):
        proto_h = self.manifold.expmap0(
            torch.tensor([[0.2, 0.0], [0.22, 0.0], [-0.8, 0.0]])
        )

        affinity = prototype_affinity_hyp(proto_h, curvature=1.0)

        self.assertGreater(affinity[0, 1].item(), affinity[0, 2].item())
        self.assertEqual(affinity.diag().count_nonzero().item(), 0)
        torch.testing.assert_close(affinity.sum(), torch.tensor(1.0))

    def test_affinity_ema_and_triplet_sampling(self):
        affinity = torch.tensor(
            [[0.0, 0.8, 0.1], [0.8, 0.0, 0.1], [0.1, 0.1, 0.0]]
        )
        updated = update_affinity_ema(None, affinity)
        triplets = sample_triplets_from_affinity(updated, 32)

        self.assertEqual(tuple(triplets.shape), (32, 3))
        self.assertTrue(torch.all(triplets[:, 0] != triplets[:, 1]))
        self.assertTrue(torch.all(triplets[:, 0] != triplets[:, 2]))
        self.assertTrue(torch.all(triplets[:, 1] != triplets[:, 2]))

    def test_hierarchy_loss_has_finite_gradients(self):
        raw = torch.randn(4, 3, requires_grad=True)
        proto_h = self.manifold.expmap0(raw / (1.0 + raw.norm(dim=1, keepdim=True)))
        triplets = torch.tensor([[0, 1, 2], [1, 2, 3]])

        loss = hierarchy_triplet_loss_hyp(proto_h, triplets, curvature=1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_sinkhorn_approximately_balances_columns(self):
        torch.manual_seed(0)
        cost = torch.rand(64, 16)

        assignments = sinkhorn_balanced_transport(cost, n_iters=50, epsilon=0.1)

        torch.testing.assert_close(assignments.sum(dim=1), torch.ones(64), atol=1e-5, rtol=1e-5)
        expected_column_mass = torch.full((16,), 4.0)
        torch.testing.assert_close(
            assignments.sum(dim=0), expected_column_mass, atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
