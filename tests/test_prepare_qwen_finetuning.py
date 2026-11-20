import json
import tempfile
import unittest
from pathlib import Path

import prepare_qwen_finetuning as prepare


def write_manifest(path, split, speakers, rows_per_speaker=2):
    with path.open("w", encoding="utf-8") as handle:
        for speaker in speakers:
            for index in range(rows_per_speaker):
                row = {
                    "audio_path": str(path.parent / f"{speaker}-{index}.wav"),
                    "speaker_id": speaker,
                    "text": f"transcript {speaker} {index}",
                    "category": "Novel Sentences" if index % 2 else "Digital Assistant Commands",
                    "split": split,
                }
                handle.write(json.dumps(row) + "\n")


class PrepareQwenFinetuningTests(unittest.TestCase):
    def test_writes_official_schema_and_speaker_complete_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "source_train.jsonl"
            val = root / "source_val.jsonl"
            out = root / "output"
            write_manifest(train, "train", ["train-a", "train-b"], rows_per_speaker=3)
            write_manifest(val, "val", ["val-a"], rows_per_speaker=2)

            prepare.main([
                "--train", str(train), "--val", str(val), "--out", str(out),
                "--pilot-size", "4", "--seed", "7",
            ])

            pilot = [json.loads(line) for line in (out / "train_pilot.jsonl").read_text().splitlines()]
            self.assertEqual(4, len(pilot))
            self.assertEqual({"audio", "text"}, set(pilot[0]))
            self.assertTrue(all(row["text"].startswith(prepare.QWEN_PREFIX) for row in pilot))
            report = json.loads((out / "preparation_report.json").read_text())
            self.assertEqual(2, report["pilot"]["speakers"])
            self.assertEqual(0, report["integrity"]["speaker_overlap"])

    def test_rejects_speaker_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "source_train.jsonl"
            val = root / "source_val.jsonl"
            write_manifest(train, "train", ["same-speaker"])
            write_manifest(val, "val", ["same-speaker"])
            with self.assertRaises(SystemExit):
                prepare.main([
                    "--train", str(train), "--val", str(val),
                    "--out", str(root / "output"),
                ])


if __name__ == "__main__":
    unittest.main()
