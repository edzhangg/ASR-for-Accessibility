import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_asr_baselines as baselines


class ModelSelectionTests(unittest.TestCase):
    def test_default_qwen_hub_id_is_preserved(self):
        args = SimpleNamespace(qwen_model="Qwen/Qwen3-ASR-1.7B-hf", qwen_adapter="")
        self.assertEqual("Qwen/Qwen3-ASR-1.7B-hf", baselines.model_id_for("qwen", args))

    def test_local_checkpoint_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-200"
            checkpoint.mkdir()
            args = SimpleNamespace(qwen_model=str(checkpoint), qwen_adapter="")
            self.assertEqual(str(checkpoint.resolve()), baselines.model_id_for("qwen", args))

    def test_non_qwen_model_is_unchanged(self):
        args = SimpleNamespace(qwen_model="ignored", qwen_adapter="")
        self.assertEqual(baselines.MODEL_IDS["whisper"], baselines.model_id_for("whisper", args))

    def test_adapter_is_part_of_saved_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "checkpoint-200"
            adapter.mkdir()
            args = SimpleNamespace(qwen_model="Qwen/Qwen3-ASR-1.7B", qwen_adapter=str(adapter))
            identity = baselines.model_identity_for("qwen", args)
            self.assertIn("LoRA:", identity)
            self.assertIn(str(adapter.resolve()), identity)


if __name__ == "__main__":
    unittest.main()
