"""Target-selection tests with fake module names. No torch/peft/GPU required."""

import unittest

from lora_configs import check_trainable, select_targets

NAMES = (
    [f"thinker.model.layers.{i}.self_attn.{p}" for i in range(3) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    + [f"thinker.model.layers.{i}.mlp.{p}" for i in range(3) for p in ("gate_proj", "up_proj", "down_proj")]
    + [f"thinker.audio_tower.layers.{i}.self_attn.{p}" for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "out_proj")]
    + [f"thinker.audio_tower.layers.{i}.fc1" for i in range(2)]
    + ["thinker.lm_head"]
)


class TestSelectTargets(unittest.TestCase):
    def test_l1_decoder_attention_only(self):
        dec, aud = select_targets(NAMES, "L1")
        self.assertEqual(len(dec), 12)
        self.assertEqual(aud, [])
        self.assertTrue(all("audio_tower" not in n and "mlp" not in n for n in dec))

    def test_l2_is_superset_of_l1(self):
        d1, _ = select_targets(NAMES, "L1")
        d2, a2 = select_targets(NAMES, "L2")
        self.assertEqual(d1, d2)
        self.assertEqual(len(a2), 8)
        self.assertTrue(all("fc1" not in n for n in a2))

    def test_l2_audio_layer_subset(self):
        _, aud = select_targets(NAMES, "L2", audio_layers={1})
        self.assertEqual(len(aud), 4)
        self.assertTrue(all(".layers.1." in n for n in aud))

    def test_errors(self):
        with self.assertRaises(ValueError):
            select_targets(NAMES, "L3")
        with self.assertRaises(RuntimeError):
            select_targets(["foo.bar"], "L1")
        with self.assertRaises(RuntimeError):
            select_targets([n for n in NAMES if "audio_tower" not in n], "L2")

    def test_check_trainable(self):
        check_trainable(["a.audio_tower.x.lora_A.weight", "b.lora_B.weight"], "L2")
        with self.assertRaises(RuntimeError):
            check_trainable(["b.lora_B.weight"], "L2")
        with self.assertRaises(RuntimeError):
            check_trainable(["a.audio_tower.x.lora_A.weight"], "L1")
        with self.assertRaises(RuntimeError):
            check_trainable(["b.weight"], "L1")


if __name__ == "__main__":
    unittest.main()
