"""CPU scheduler/preflight tests; never launch training or remote commands."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'gpu' / 'run_inspection_tasks.py'
spec = importlib.util.spec_from_file_location('inspection_pipeline_under_test', SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class SchedulerTests(unittest.TestCase):
    def test_no_receipt(self):
        self.assertIsNone(pipeline.select_ready_task(pipeline.TASKS, {}))

    def test_crop_can_run_before_fire(self):
        self.assertEqual(pipeline.select_ready_task(pipeline.TASKS, {'ip102': 20}), pipeline.TASKS[1])

    def test_earliest_receipt_wins(self):
        self.assertEqual(pipeline.select_ready_task(pipeline.TASKS, {'dfire': 30, 'ip102': 20}), pipeline.TASKS[1])
        self.assertEqual(pipeline.select_ready_task(pipeline.TASKS, {'dfire': 10, 'ip102': 20}), pipeline.TASKS[0])

    def test_tie_is_stable_and_input_not_mutated(self):
        pending = list(pipeline.TASKS)
        self.assertEqual(pipeline.select_ready_task(pending, {'dfire': 20, 'ip102': 20}), pending[0])
        self.assertEqual(pending, pipeline.TASKS)

    def test_completed_source_not_selected_again(self):
        self.assertEqual(pipeline.select_ready_task([pipeline.TASKS[0]], {'ip102': 10, 'dfire': 20}), pipeline.TASKS[0])

    def test_preflight_imports_yaml(self):
        with patch.object(pipeline.importlib, 'import_module') as module:
            pipeline.preflight_dependencies()
            module.assert_called_once_with('yaml')

    def test_preflight_failure_is_explicit(self):
        with patch.object(pipeline.importlib, 'import_module', side_effect=ImportError('missing yaml')):
            with self.assertRaisesRegex(RuntimeError, 'Dependency preflight failed.*PyYAML'):
                pipeline.preflight_dependencies()


class CompletedTaskGuardTests(unittest.TestCase):
    """A coordinator restart must not redo a task that already finished.

    Before this guard, restarting the orchestrator re-ran the finished task: the
    trainer's --resume-last re-exports the ONNX and rewrites that task's status and
    report files, which would invalidate an artifact that was already verified.
    """

    def _root(self, tmp):
        root = Path(tmp)
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        return root

    def test_absent_status_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pipeline, 'ROOT', self._root(tmp)):
                self.assertFalse(pipeline.already_complete('crop_pest'))

    def test_complete_phase_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            (root / 'logs' / 'crop_pest_pipeline_status.json').write_text(
                json.dumps({'task': 'crop_pest', 'phase': 'complete'}), encoding='utf-8')
            with patch.object(pipeline, 'ROOT', root):
                self.assertTrue(pipeline.already_complete('crop_pest'))
                self.assertFalse(pipeline.already_complete('fire_smoke'))

    def test_failed_and_unparsable_status_are_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            path = root / 'logs' / 'fire_smoke_pipeline_status.json'
            path.write_text(json.dumps({'phase': 'failed'}), encoding='utf-8')
            with patch.object(pipeline, 'ROOT', root):
                self.assertFalse(pipeline.already_complete('fire_smoke'))
            path.write_text('{not json', encoding='utf-8')
            with patch.object(pipeline, 'ROOT', root):
                self.assertFalse(pipeline.already_complete('fire_smoke'))


_AUTO_SCRIPT = SCRIPT.parent / 'auto_relaunch_when_idle.py'
_auto_spec = importlib.util.spec_from_file_location('inspection_autorelaunch_under_test', _AUTO_SCRIPT)
auto_relaunch = importlib.util.module_from_spec(_auto_spec)
_auto_spec.loader.exec_module(auto_relaunch)


class AutoRelaunchTriggerTests(unittest.TestCase):
    """The dead-end watcher must never relaunch a healthy run.

    A relaunch while the coordinator is alive would be caught by that launcher's
    pidfile guard, but the trigger itself has to be correct: replacing a coordinator
    that is mid-training is the one outcome worth proving impossible.
    """

    def test_dead_end_triggers_relaunch(self):
        self.assertEqual(auto_relaunch.decide('complete', 'failed'), 'launch')

    def test_failed_crop_pest_triggers_relaunch(self):
        self.assertEqual(auto_relaunch.decide('failed', 'waiting_for_verified_download'), 'launch')

    def test_healthy_run_never_triggers_relaunch(self):
        for crop in ('not_started', 'waiting_for_verified_download', 'prepare', 'smoke', 'train', 'complete'):
            for fire in ('not_started', 'waiting_for_verified_download', 'prepare', 'smoke', 'train', 'complete'):
                self.assertEqual(auto_relaunch.decide(crop, fire), 'wait', f'{crop}/{fire}')


if __name__ == '__main__':
    unittest.main()
