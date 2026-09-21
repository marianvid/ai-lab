"""Native ComfyUI editors use only the retained local model variants."""

import json
import unittest

from ai_lab.comfyui_templates import for_model


class NativeTemplatesTests(unittest.TestCase):
    def test_every_retained_comfyui_model_has_a_native_template(self):
        variants = {
            "qwen-image-2512-nvfp4": "qwen_image_nvfp4.safetensors",
            "qwen-image-edit-2511-fp8mixed": "qwen_image_edit_2511_fp8mixed.safetensors",
            "flux2-klein-4b-bf16": "flux-2-klein-4b.safetensors",
            "minimax-h3": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
            "ltx-2.5-nvfp4": "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
            "minimax-music3": "minimax_music3_dit_int8_convrot.safetensors",
        }
        for name, checkpoint in variants.items():
            with self.subTest(name=name):
                path = for_model(name)
                self.assertTrue(path.is_file())
                graph = json.loads(path.read_text())
                self.assertTrue(graph["definitions"]["subgraphs"])
                self.assertIn(checkpoint, json.dumps(graph))

    def test_flux2_q8_uses_platform_specific_gguf_loaders(self):
        for platform, name in (("linux", "flux2-dev-Q8_0.gguf"),
                               ("darwin", "flux2-dev-q8-0/diffusion_models/flux2-dev-Q8_0.gguf")):
            with self.subTest(platform=platform):
                graph = json.loads(for_model("flux2-dev-q8-0", platform).read_text())
                loaders = [node for sub in graph["definitions"]["subgraphs"]
                           for node in sub["nodes"] if node["type"] == "UnetLoaderGGUF"]
                self.assertEqual(loaders[0]["widgets_values"], [name])

    def test_klein_runs_only_the_installed_distilled_branch(self):
        graph = json.loads(for_model("flux2-klein-4b-bf16").read_text())
        self.assertEqual(len(graph["definitions"]["subgraphs"]), 1)
        self.assertEqual(len([node for node in graph["nodes"]
                              if node["type"] == "SaveImage"]), 1)
        self.assertNotIn('"flux-2-klein-base-4b.safetensors"', json.dumps(graph))

    def test_adapted_graphs_have_connected_links_and_no_missing_optional_branches(self):
        names = [
            "qwen-image-2512-nvfp4", "qwen-image-edit-2511-fp8mixed",
            "flux2-klein-4b-bf16", "minimax-h3", "ltx-2.5-nvfp4",
            "minimax-music3", "flux2-dev-q8-0",
        ]
        for name in names:
            with self.subTest(name=name):
                graph = json.loads(for_model(name).read_text())
                for subgraph in graph["definitions"]["subgraphs"]:
                    nodes = {node["id"]: node for node in subgraph["nodes"]}
                    links = {link["id"]: link for link in subgraph["links"]}
                    for link in links.values():
                        self.assertTrue(link["origin_id"] < 0 or link["origin_id"] in nodes)
                        self.assertTrue(link["target_id"] < 0 or link["target_id"] in nodes)
                        if link["origin_id"] < 0:
                            self.assertLess(link["origin_slot"], len(subgraph["inputs"]))
                    for node in nodes.values():
                        for output in node.get("outputs", []):
                            for link_id in output.get("links") or []:
                                self.assertIn(link_id, links)
                                self.assertEqual(links[link_id]["origin_id"], node["id"])
                        for index, input_ in enumerate(node.get("inputs", [])):
                            link_id = input_.get("link")
                            if link_id is not None:
                                self.assertIn(link_id, links)
                                self.assertEqual(links[link_id]["target_id"], node["id"])
                                self.assertEqual(links[link_id]["target_slot"], index)
                    self.assertFalse(any(node["type"] == "LoraLoaderModelOnly"
                                         for node in nodes.values()))
                    for node in nodes.values():
                        if node["type"] == "PrimitiveBoolean" and "LoRA" in node.get("title", ""):
                            self.assertEqual(node.get("widgets_values", [None])[0], False)


if __name__ == "__main__":
    unittest.main()
