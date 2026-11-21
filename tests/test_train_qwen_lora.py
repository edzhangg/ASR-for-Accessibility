import json
import tempfile
import unittest
from pathlib import Path

import train_qwen_lora as train_lora


class TrainQwenLoraHelperTests(unittest.TestCase):
    def test_bridges_outer_embedding_method_to_thinker(self):
        expected = object()

        class Thinker:
            def get_input_embeddings(self):
                return expected

        class Outer:
            thinker = Thinker()

            def get_input_embeddings(self):
                raise NotImplementedError

        model = Outer()
        self.assertIs(expected, train_lora.bridge_input_embeddings(model))
        self.assertIs(expected, model.get_input_embeddings())

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

    def test_read_speaker_ids_requires_speaker(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.jsonl"
            good.write_text(json.dumps({"audio": "a", "text": "t", "speaker_id": "s1"}) + "\n", encoding="utf-8")
            self.assertEqual([{"speaker_id": "s1"}], train_lora.read_speaker_ids(good))
            bad = Path(directory) / "bad.jsonl"
            bad.write_text(json.dumps({"audio": "a", "text": "t"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                train_lora.read_speaker_ids(bad)

    def test_sampler_none_and_missing_weights(self):
        from types import SimpleNamespace
        self.assertIsNone(train_lora.build_train_sampler(SimpleNamespace(sampling="none")))
        with self.assertRaises(SystemExit):
            train_lora.build_train_sampler(SimpleNamespace(sampling="e3", weights_csv=None))

    def test_gradient_check_passes_when_all_groups_train(self):
        stats = [("m.layers.0.q_proj.lora_B.default.weight", True),
                 ("m.audio_tower.layers.0.q_proj.lora_B.default.weight", True)]
        summary = train_lora.check_gradient_flow(stats, "L2")
        self.assertEqual(summary["audio"]["with_gradient"], 1)
        train_lora.check_gradient_flow(stats[:1], "L1")          # L1 has no audio matrices: fine

    def test_gradient_check_fails_when_audio_adapters_get_no_gradient(self):
        # The silent failure we guard against: decoder trains, audio adapters never receive gradient.
        stats = [("m.layers.0.q_proj.lora_B.default.weight", True),
                 ("m.audio_tower.layers.0.q_proj.lora_B.default.weight", False),
                 ("m.audio_tower.layers.1.q_proj.lora_B.default.weight", False)]
        with self.assertRaisesRegex(RuntimeError, "audio: only 0/2"):
            train_lora.check_gradient_flow(stats, "L2")
        train_lora.check_gradient_flow(stats[:1], "L1")          # same model is fine as L1

    def test_gradient_check_fails_when_a_required_group_is_missing_or_dead(self):
        with self.assertRaisesRegex(RuntimeError, "decoder: no LoRA"):
            train_lora.check_gradient_flow([], "L1")
        with self.assertRaisesRegex(RuntimeError, "audio: no LoRA"):
            train_lora.check_gradient_flow([("m.layers.0.q_proj.lora_B.w", True)], "L2")
        with self.assertRaisesRegex(RuntimeError, "decoder: only 0/1"):
            train_lora.check_gradient_flow([("m.layers.0.q_proj.lora_B.w", False)], "L1")
        with self.assertRaisesRegex(RuntimeError, "exist under L1"):
            train_lora.check_gradient_flow([("m.layers.0.q.lora_B.w", True),
                                            ("m.audio_tower.layers.0.q.lora_B.w", True)], "L1")

    def test_finds_highest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoint-2").mkdir()
            (output / "checkpoint-11").mkdir()
            (output / "notes").mkdir()
            self.assertEqual(output / "checkpoint-11", train_lora.find_latest_checkpoint(output))


if __name__ == "__main__":
    unittest.main()
