"""The native MiniMax template matches the installed INT8 model."""

import json
import unittest
from pathlib import Path


class MiniMaxTemplateTests(unittest.TestCase):
    def test_native_template_has_model_specific_editor(self):
        path = (Path(__file__).resolve().parents[2] / "ai_lab" /
                "comfyui_templates" / "audio_minimax_music_3.json")
        graph = json.loads(path.read_text())
        self.assertTrue(graph["definitions"]["subgraphs"])
        self.assertIn("Text to Music (MiniMax Music 3)",
                      [item["name"] for item in graph["definitions"]["subgraphs"]])
        self.assertIn("minimax_music3_dit_int8_convrot.safetensors",
                      json.dumps(graph))
        model_nodes = [item for subgraph in graph["definitions"]["subgraphs"]
                       for item in subgraph["nodes"] if item["type"] == "UNETLoader"]
        self.assertEqual(model_nodes[0]["widgets_values"][0],
                         "minimax_music3_dit_int8_convrot.safetensors")
        self.assertEqual(graph["nodes"][1]["widgets_values"][4],
                         "minimax_music3_dit_int8_convrot.safetensors")


if __name__ == "__main__":
    unittest.main()
