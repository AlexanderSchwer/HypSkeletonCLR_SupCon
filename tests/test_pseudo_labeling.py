import unittest
import importlib.util
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tools.pseudo_labeling import pseudo_label_mask_from_posteriors

HAS_GEOOPT = importlib.util.find_spec("geoopt") is not None


class RecordingCriterion:
    def __init__(self):
        self.features = None
        self.labels = None
        self.mask = None

    def __call__(self, features, labels=None, mask=None):
        self.features = features
        self.labels = labels
        self.mask = mask
        return features.sum() * 0.0 + 2.0


class ConstantLoss:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, output, target):
        return output.sum() * 0.0 + self.value


class PrototypePseudoLabelingTest(unittest.TestCase):
    def test_confident_same_cluster_samples_become_extra_positives(self):
        posteriors = torch.tensor(
            [
                [0.90, 0.10],
                [0.85, 0.15],
                [0.20, 0.80],
                [0.55, 0.45],
            ]
        )

        mask, labels, confidence, confident = pseudo_label_mask_from_posteriors(
            posteriors,
            confidence_threshold=0.8,
        )

        expected_mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(mask, expected_mask)
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 0]))
        torch.testing.assert_close(confidence, torch.tensor([0.90, 0.85, 0.80, 0.55]))
        torch.testing.assert_close(confident, torch.tensor([True, True, True, False]))

    def test_hard_pseudo_mask_weights_augmentation_and_cluster_pairs(self):
        posteriors = torch.tensor(
            [
                [0.90, 0.10],
                [0.85, 0.15],
                [0.20, 0.80],
            ]
        )

        mask, _, _, _ = pseudo_label_mask_from_posteriors(
            posteriors,
            confidence_threshold=0.8,
            mode="pseudo_hard",
            lambda_aug=0.7,
            lambda_pseudo=0.3,
        )

        expected_mask = torch.tensor(
            [
                [0.7, 0.3, 0.0],
                [0.3, 0.7, 0.0],
                [0.0, 0.0, 0.7],
            ]
        )
        torch.testing.assert_close(mask, expected_mask)

    def test_normalized_hard_pseudo_mask_keeps_fixed_anchor_mass(self):
        posteriors = torch.tensor(
            [
                [0.90, 0.10],
                [0.85, 0.15],
                [0.88, 0.12],
                [0.20, 0.80],
            ]
        )

        mask, _, _, _ = pseudo_label_mask_from_posteriors(
            posteriors,
            confidence_threshold=0.8,
            mode="pseudo_hard",
            lambda_aug=1.0,
            lambda_pseudo=0.6,
            normalize_pseudo_mass=True,
        )

        expected_mask = torch.tensor(
            [
                [1.0, 0.3, 0.3, 0.0],
                [0.3, 1.0, 0.3, 0.0],
                [0.3, 0.3, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(mask, expected_mask)

    def test_soft_pseudo_mask_uses_posterior_overlap(self):
        posteriors = torch.tensor(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.20, 0.80],
                [0.55, 0.45],
            ]
        )

        mask, _, _, _ = pseudo_label_mask_from_posteriors(
            posteriors,
            confidence_threshold=0.8,
            mode="pseudo_soft",
            lambda_aug=1.0,
            lambda_pseudo=0.5,
        )

        expected_mask = torch.tensor(
            [
                [1.00, 0.37, 0.13, 0.00],
                [0.37, 1.00, 0.16, 0.00],
                [0.13, 0.16, 1.00, 0.00],
                [0.00, 0.00, 0.00, 1.00],
            ]
        )
        torch.testing.assert_close(mask, expected_mask)

    def test_normalized_soft_pseudo_mask_keeps_fixed_anchor_mass(self):
        posteriors = torch.tensor(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.20, 0.80],
                [0.55, 0.45],
            ]
        )

        mask, _, _, _ = pseudo_label_mask_from_posteriors(
            posteriors,
            confidence_threshold=0.8,
            mode="pseudo_soft",
            lambda_aug=1.0,
            lambda_pseudo=0.5,
            normalize_pseudo_mass=True,
        )

        off_diagonal_mass = mask - torch.eye(4)
        torch.testing.assert_close(off_diagonal_mass[:3].sum(dim=1), torch.full((3,), 0.5))
        torch.testing.assert_close(off_diagonal_mass[3].sum(), torch.tensor(0.0))

    def test_invalid_posterior_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.rand(4), confidence_threshold=0.8)
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.rand(4, 2), confidence_threshold=1.1)
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.rand(4, 2), mode="unknown")
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.rand(4, 2), lambda_aug=-1.0)
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.rand(4, 2), lambda_pseudo=-1.0)
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.tensor([[0.6, -0.1]]), confidence_threshold=0.8)
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.tensor([[float("nan"), 0.1]]), confidence_threshold=0.8)
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.tensor([[1, 0]]), confidence_threshold=0.8)
        with self.assertRaises(ValueError):
            pseudo_label_mask_from_posteriors(torch.rand(4, 2), normalize_pseudo_mass=1)

    def test_processor_uses_selected_posterior_source_for_pseudo_supcon(self):
        if not HAS_GEOOPT:
            self.skipTest("geoopt is required to import the pretraining processor")
        try:
            from processor.pretrain_skeletonclr import SkeletonCLR_Processor
        except ModuleNotFoundError as exc:
            self.skipTest(f"processor dependency is not installed: {exc}")

        processor = object.__new__(SkeletonCLR_Processor)
        processor.arg = SimpleNamespace(
            contrastive_mode=None,
            contrastive_schedule=[
                {
                    "mode": "pseudo_hard",
                    "start_epoch": 1,
                    "end_epoch": 1,
                    "lambda_aug": 1.0,
                    "lambda_pseudo": 1.0,
                }
            ],
            num_epoch=1,
            lambda_aug=1.0,
            lambda_pseudo=None,
            lambda_pseudo_supcon=None,
            pseudo_supcon_warmup_steps=0,
            pseudo_supcon_ramp_steps=0,
            pseudo_supcon_assignment_source="p_mean",
            pseudo_supcon_confidence_threshold=0.8,
        )
        processor.criterion = RecordingCriterion()
        processor.global_step = 1
        processor.contrastive_schedule = processor._normalize_contrastive_schedule()

        features = torch.randn(3, 2, 4)
        cluster_pack = {
            "p_q": torch.tensor([[0.90, 0.10], [0.80, 0.20], [0.20, 0.80]]),
            "p_k": torch.tensor([[0.90, 0.10], [0.90, 0.10], [0.10, 0.90]]),
        }

        loss, metrics = processor._compute_scheduled_contrastive_loss(
            epoch=1,
            features_sup=features,
            cluster_pack=cluster_pack,
        )

        expected_mask = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(loss, torch.tensor(2.0))
        torch.testing.assert_close(processor.criterion.features, features)
        torch.testing.assert_close(processor.criterion.mask, expected_mask)
        self.assertEqual(metrics["pseudo_supcon_active_clusters"], 2)
        self.assertEqual(metrics["pseudo_supcon_extra_positive_pairs"], 2)

    def test_processor_supervised_mode_uses_dataset_labels(self):
        if not HAS_GEOOPT:
            self.skipTest("geoopt is required to import the pretraining processor")
        try:
            from processor.pretrain_skeletonclr import SkeletonCLR_Processor
        except ModuleNotFoundError as exc:
            self.skipTest(f"processor dependency is not installed: {exc}")

        processor = object.__new__(SkeletonCLR_Processor)
        processor.arg = SimpleNamespace(
            contrastive_mode=None,
            contrastive_schedule=[
                {"mode": "supervised", "start_epoch": 1, "end_epoch": 1}
            ],
            num_epoch=1,
            lambda_aug=1.0,
        )
        processor.criterion = RecordingCriterion()
        processor.contrastive_schedule = processor._normalize_contrastive_schedule()

        features = torch.randn(3, 2, 4)
        labels = torch.tensor([0, 1, 1])

        loss, metrics = processor._compute_scheduled_contrastive_loss(
            epoch=1,
            features_sup=features,
            labels=labels,
        )

        torch.testing.assert_close(loss, torch.tensor(2.0))
        torch.testing.assert_close(processor.criterion.features, features)
        torch.testing.assert_close(processor.criterion.labels, labels)
        self.assertIsNone(processor.criterion.mask)
        self.assertEqual(metrics["contrastive_mode"], "supervised")
        self.assertEqual(metrics["lambda_pseudo_effective"], 0.0)

    def test_schedule_blends_adjacent_modes_during_transition(self):
        if not HAS_GEOOPT:
            self.skipTest("geoopt is required to import the pretraining processor")
        try:
            from processor.pretrain_skeletonclr import SkeletonCLR_Processor
        except ModuleNotFoundError as exc:
            self.skipTest(f"processor dependency is not installed: {exc}")

        processor = object.__new__(SkeletonCLR_Processor)
        processor.arg = SimpleNamespace(
            contrastive_mode=None,
            contrastive_schedule=[
                {"mode": "augmentation", "start_epoch": 1, "end_epoch": 2},
                {
                    "mode": "supervised",
                    "start_epoch": 3,
                    "end_epoch": 5,
                    "transition_epochs": 2,
                },
            ],
            num_epoch=5,
            lambda_aug=1.0,
        )
        processor.loss = ConstantLoss(10.0)
        processor.criterion = RecordingCriterion()
        processor.contrastive_schedule = processor._normalize_contrastive_schedule()

        output = torch.randn(2, 3)
        target = torch.zeros(2, dtype=torch.long)
        features = torch.randn(2, 2, 4)
        labels = torch.tensor([0, 1])

        loss, metrics = processor._compute_scheduled_contrastive_loss(
            epoch=3,
            features_sup=features,
            output=output,
            target=target,
            labels=labels,
        )

        self.assertAlmostEqual(
            loss.item(),
            (2.0 / 3.0) * 10.0 + (1.0 / 3.0) * 2.0,
            places=6,
        )
        self.assertEqual(metrics["contrastive_mode"], "augmentation:0.667+supervised:0.333")
        self.assertAlmostEqual(metrics["contrastive_phase_0_weight"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["contrastive_phase_1_weight"], 1.0 / 3.0)

    def test_schedule_rejects_missing_epoch_coverage(self):
        if not HAS_GEOOPT:
            self.skipTest("geoopt is required to import the pretraining processor")
        try:
            from processor.pretrain_skeletonclr import SkeletonCLR_Processor
        except ModuleNotFoundError as exc:
            self.skipTest(f"processor dependency is not installed: {exc}")

        processor = object.__new__(SkeletonCLR_Processor)
        processor.arg = SimpleNamespace(
            contrastive_mode=None,
            contrastive_schedule=[
                {"mode": "augmentation", "start_epoch": 1, "end_epoch": 1},
                {"mode": "supervised", "start_epoch": 3, "end_epoch": 5},
            ],
            num_epoch=5,
        )

        with self.assertRaises(ValueError):
            processor._normalize_contrastive_schedule()

    def test_processor_rejects_misaligned_pseudo_supcon_batches(self):
        if not HAS_GEOOPT:
            self.skipTest("geoopt is required to import the pretraining processor")
        try:
            from processor.pretrain_skeletonclr import SkeletonCLR_Processor
        except ModuleNotFoundError as exc:
            self.skipTest(f"processor dependency is not installed: {exc}")

        processor = object.__new__(SkeletonCLR_Processor)
        processor.arg = SimpleNamespace(
            contrastive_mode=None,
            contrastive_schedule=[
                {
                    "mode": "pseudo_hard",
                    "start_epoch": 1,
                    "end_epoch": 1,
                    "lambda_aug": 1.0,
                    "lambda_pseudo": 1.0,
                }
            ],
            num_epoch=1,
            lambda_aug=1.0,
            lambda_pseudo=None,
            lambda_pseudo_supcon=None,
            pseudo_supcon_warmup_steps=0,
            pseudo_supcon_ramp_steps=0,
            pseudo_supcon_assignment_source="q_k",
            pseudo_supcon_confidence_threshold=0.8,
        )
        processor.criterion = RecordingCriterion()
        processor.global_step = 1
        processor.contrastive_schedule = processor._normalize_contrastive_schedule()

        features = torch.randn(3, 2, 4)
        cluster_pack = {"q_k": torch.tensor([[0.90, 0.10], [0.80, 0.20]])}

        with self.assertRaises(ValueError):
            processor._compute_scheduled_contrastive_loss(
                epoch=1,
                features_sup=features,
                cluster_pack=cluster_pack,
            )

    @unittest.skipUnless(HAS_GEOOPT, "geoopt is required for hyperbolic SupCon")
    def test_supcon_accepts_pseudo_positive_mask(self):
        import geoopt

        from tools.losses import SupConLoss

        torch.manual_seed(0)
        manifold = geoopt.PoincareBall(1.0)
        raw = torch.randn(4, 2, 3, requires_grad=True)
        tangent = F.normalize(raw, dim=2) * 0.2
        features = manifold.expmap0(tangent)
        mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )

        loss = SupConLoss(temperature=0.07, curvature=1.0)(features, mask=mask)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw.grad).all())

    @unittest.skipUnless(HAS_GEOOPT, "geoopt is required for hyperbolic distance floor")
    def test_pseudo_cluster_distance_floor_loss_pushes_too_close_pairs(self):
        import geoopt

        from tools.losses import pseudo_cluster_distance_floor_loss

        manifold = geoopt.PoincareBall(1.0)
        raw = torch.tensor(
            [
                [[0.01, 0.00], [0.02, 0.00]],
                [[0.03, 0.00], [0.04, 0.00]],
                [[0.40, 0.00], [0.45, 0.00]],
            ],
            requires_grad=True,
        )
        features = manifold.expmap0(raw)
        weights = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        loss, metrics = pseudo_cluster_distance_floor_loss(
            features,
            weights,
            distance_floor=0.2,
            curvature=1.0,
        )
        loss.backward()

        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(metrics["pseudo_supcon_floor_violation_fraction"], 0.0)
        self.assertTrue(torch.isfinite(raw.grad).all())

    @unittest.skipUnless(HAS_GEOOPT, "geoopt is required for hyperbolic SupCon")
    def test_supcon_rejects_wrong_mask_shape(self):
        from tools.losses import SupConLoss

        features = torch.randn(4, 2, 3)
        mask = torch.eye(3)

        with self.assertRaises(ValueError):
            SupConLoss(temperature=0.07, curvature=1.0)(features, mask=mask)


if __name__ == "__main__":
    unittest.main()
