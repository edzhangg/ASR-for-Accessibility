import json
import tempfile
import unittest
from pathlib import Path

import train_qwen_lora as train_lora


class TrainQwenLoraHelperTests(unittest.TestCase):
    def test_validates_official_audio_text_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            manifest.write_text(
                json.dumps({"audio": "example.wav", "text": "language English<asr_text>hello"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(1, train_lora.validate_jsonl_schema(manifest))

    def test_rejects_missing_training_text(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            manifest.write_text(json.dumps({"audio": "example.wav"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                train_lora.validate_jsonl_schema(manifest)

    def test_finds_highest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoint-2").mkdir()
            (output / "checkpoint-11").mkdir()
            (output / "notes").mkdir()
            self.assertEqual(output / "checkpoint-11", train_lora.find_latest_checkpoint(output))


if __name__ == "__main__":
    unittest.main()
