import os
import tempfile
import unittest

from processor.work_dir import prepare_training_work_dir


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


if __name__ == '__main__':
    unittest.main()
