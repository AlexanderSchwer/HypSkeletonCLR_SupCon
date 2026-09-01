import os
import tempfile
import unittest
from argparse import Namespace

from processor.work_dir import (
    build_canonical_work_dir,
    prepare_training_work_dir,
    wandb_run_identity_from_work_dir,
)


class TrainingWorkDirTest(unittest.TestCase):
    def test_auto_allocates_unique_run_directory_below_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, 'experiment')

            first = prepare_training_work_dir(base, mode='auto', run_id='20260728-143012')
            second = prepare_training_work_dir(base, mode='auto', run_id='20260728-143012')

            self.assertEqual(first, os.path.join(base, 'runs', '20260728-143012'))
            self.assertEqual(second, os.path.join(base, 'runs', '20260728-143012_001'))
            self.assertTrue(os.path.isdir(first))
            self.assertTrue(os.path.isdir(second))

    def test_auto_saved_run_config_allocates_sibling_not_nested_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, 'experiment')
            old_run = prepare_training_work_dir(base, mode='auto', run_id='old-run')

            new_run = prepare_training_work_dir(old_run, mode='auto', run_id='new-run')

            self.assertEqual(new_run, os.path.join(base, 'runs', 'new-run'))

    def test_error_mode_fails_when_work_dir_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileExistsError):
                prepare_training_work_dir(tmpdir, mode='error')

    def test_resume_mode_uses_exact_work_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = os.path.join(tmpdir, 'existing-run')

            resolved = prepare_training_work_dir(work_dir, mode='resume')

            self.assertEqual(resolved, work_dir)
            self.assertFalse(os.path.exists(work_dir))

    def test_canonical_skeletonclr_geoopt_hyperbolic_cluster_path(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr/1_xview_frame50_channel16_epoch300_cross150/c005_cosann_hyp_clust',
            config='config/SkeletonCLR/skeletonclr_xview_hyp_clust.yaml',
            model='net.skeletonclr.SkeletonCLR',
            model_args={
                'num_class': 60,
                'hidden_channels': 16,
                'curvature': 1.0,
                'cluster_enabled': True,
                'num_clusters': 5,
            },
            train_feeder_args={
                'data_path': 'xview/train_position.npy',
                'label_path': './data/ntu60/xview/train_label.pkl',
            },
            test_feeder_args={},
            base_lr=0.05,
            step=[250],
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/skeletonclr/ntu60-xview-frame50-hc16/'
                'geoopt-c1-clust5-lrb0p05-multistep'))

    def test_canonical_encodes_pseudo_supcon_schedule(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr/1_xview_frame50_channel16_epoch300_cross150/c005_cosann_hyp_pseudo_supcon',
            config='config/SkeletonCLR/skeletonclr_xview_hyp_pseudo_supcon.yaml',
            model='net.skeletonclr.SkeletonCLR',
            model_args={
                'num_class': 60,
                'hidden_channels': 16,
                'curvature': 1.0,
                'cluster_enabled': True,
                'num_clusters': 5,
            },
            train_feeder_args={'data_path': 'xview/train_position.npy'},
            test_feeder_args={},
            contrastive_schedule=[
                {'mode': 'augmentation', 'start_epoch': 1, 'end_epoch': 150},
                {'mode': 'pseudo_hard', 'start_epoch': 151, 'end_epoch': 300},
            ],
            base_lr=0.01,
            step=[],
            lr_scheduler='cosine',
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')
        name, group = wandb_run_identity_from_work_dir(
            os.path.join(resolved, 'runs', '20260825-143012_pid12345')
        )

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/skeletonclr/ntu60-xview-frame50-hc16/'
                'geoopt-c1-clust5-pseudo-lrb0p01-cosine'))
        self.assertEqual(
            name,
            'skeletonclr/ntu60-xview-frame50-hc16/'
            'geoopt-c1-clust5-pseudo-lrb0p01-cosine')
        self.assertIsNone(group)

    def test_canonical_uses_xsubject_for_xsub_dataset(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr/1_xview_frame50_channel16_epoch300_cross150',
            config='config/SkeletonCLR/skeletonclr_xsub.yaml',
            model='net.skeletonclr.SkeletonCLR',
            model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
            train_feeder_args={'data_path': 'xsub/train_position.npy'},
            test_feeder_args={},
            base_lr=0.05,
            step=[250],
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')

        self.assertIn(
            os.path.normpath('work_dir/skeletonclr/ntu60-xsubject-frame50-hc16'),
            resolved)

    def test_canonical_tags_hyptorch_implementation(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr',
            config='config/SkeletonCLR/skeletonclr_hyptorch_xview.yaml',
            model='net.skeletonclr_hyptorch.SkeletonCLRHyptorch',
            model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
            train_feeder_args={'data_path': 'xview/train_position.npy'},
            test_feeder_args={},
            base_lr=0.05,
            step=[250],
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/skeletonclr/ntu60-xview-frame50-hc16/'
                'hyptorch-c1-lrb0p05-multistep'))

    def test_canonical_uses_skeletonclr_3views_model_family(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr/1_xview_frame50_channel16_epoch300_cross150/3views_c005_cosann',
            config='config/SkeletonCLR/skeletonclr_3views_xview.yaml',
            model='net.skeletonclr_3views.SkeletonCLR_3views',
            model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
            train_feeder_args={'data_path': 'xview/train_position.npy'},
            test_feeder_args={},
            base_lr=0.01,
            step=[],
            lr_scheduler='cosine',
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_3views_Processor')

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/skeletonclr_3views/ntu60-xview-frame50-hc16/'
                'geoopt-c1-lrb0p01-cosine'))

    def test_canonical_omits_geometry_and_curvature_for_euclidean_run(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr',
            config='config/SkeletonCLR/skeletonclr_eucl_3views_xview.yaml',
            model='net.skeletonclr_3views_eucl.SkeletonCLR_3views_Eucl',
            model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
            train_feeder_args={'data_path': 'xview/train_position.npy'},
            test_feeder_args={},
            base_lr=0.05,
            step=[250],
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/skeletonclr_3views_eucl/ntu60-xview-frame50-hc16/'
                'lrb0p05-multistep'))

    def test_canonical_preserves_absolute_work_dir_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, 'work_dir')
            arg = Namespace(
                work_dir=os.path.join(root, 'skeletonclr', 'old-name'),
                config='config/SkeletonCLR/skeletonclr_xview.yaml',
                model='net.skeletonclr.SkeletonCLR',
                model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
                train_feeder_args={'data_path': 'xview/train_position.npy'},
                test_feeder_args={},
                base_lr=0.05,
                step=[250],
            )

            resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')

            self.assertEqual(
                resolved,
                os.path.join(
                    root,
                    'skeletonclr',
                    'ntu60-xview-frame50-hc16',
                    'geoopt-c1-lrb0p05-multistep'))

    def test_canonical_uses_cosine_tag_when_cosine_scheduler_is_enabled(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr',
            config='config/SkeletonCLR/skeletonclr_xview.yaml',
            model='net.skeletonclr.SkeletonCLR',
            model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
            train_feeder_args={'data_path': 'xview/train_position.npy'},
            test_feeder_args={},
            base_lr=0.05,
            step=[250],
            lr_scheduler='cosine',
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/skeletonclr/ntu60-xview-frame50-hc16/'
                'geoopt-c1-lrb0p05-cosine'))

    def test_canonical_uses_constant_lr_tag_when_scheduler_is_none(self):
        arg = Namespace(
            work_dir='work_dir/skeletonclr',
            config='config/SkeletonCLR/skeletonclr_xview.yaml',
            model='net.skeletonclr.SkeletonCLR',
            model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
            train_feeder_args={'data_path': 'xview/train_position.npy'},
            test_feeder_args={},
            base_lr=0.05,
            step=[],
            lr_scheduler='none',
        )

        resolved = build_canonical_work_dir(arg, processor_name='SkeletonCLR_Processor')

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/skeletonclr/ntu60-xview-frame50-hc16/'
                'geoopt-c1-lrb0p05-constant-lr'))

    def test_canonical_linear_eval_uses_argument_details_in_experiment_path(self):
        arg = Namespace(
            work_dir='work_dir/linear_eval/1_model300',
            config='config/linear_eval/linear_eval_skeletonclr_xview.yaml',
            model='net.skeletonclr.SkeletonCLR',
            model_args={'num_class': 60, 'hidden_channels': 16, 'curvature': 1.0},
            train_feeder_args={'data_path': 'xview/train_position.npy'},
            test_feeder_args={'data_path': 'xview/val_position.npy'},
            base_lr=8.0,
            num_epoch=100,
            step=[],
            lr_scheduler='multistep',
            lr_milestones=[80],
            lr_gamma=0.1,
        )

        resolved = build_canonical_work_dir(arg, processor_name='LE_Processor')

        self.assertEqual(
            resolved,
            os.path.normpath(
                'work_dir/linear_eval/ntu60-xview-frame50-hc16/'
                'geoopt-c1-lrb8-ep100-multistep-ms80-g0p1'))

    def test_wandb_identity_uses_experiment_path_as_name_without_group(self):
        work_dir = os.path.normpath(
            'work_dir/skeletonclr/ntu60-xview-frame50-hc16/'
            'geoopt-c1-clust5-lrb0p1-cosine/runs/20260825-143012_pid12345')

        name, group = wandb_run_identity_from_work_dir(work_dir)

        self.assertEqual(
            name,
            'skeletonclr/ntu60-xview-frame50-hc16/'
            'geoopt-c1-clust5-lrb0p1-cosine')
        self.assertIsNone(group)


if __name__ == '__main__':
    unittest.main()
