"""Every setting of the configuration file is on a page; the rest shown read-only, secrets hidden."""

import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from ai_lab.application.file_settings import (EDITABLE, FIELDS, READ_ONLY, READ_ONLY_KIND,
                                              SCHEMA, FileSettings, masked)
from ai_lab.application.setting_fields import BY_KEY
from ai_lab.application.settings_changes import GATEWAY_SETTINGS
from ai_lab.config.policy import GatewayPolicy, MachineLimits
from ai_lab.config.store import Config, ConfigStore
from ai_lab.gpu_health.settings import GpuHealthSettings


def leaves(cls, prefix):
    """Every setting a section's parser reads, by its dotted key."""
    keys = set()
    for item in fields(cls):
        default = getattr(cls(), item.name)
        if hasattr(default, "__dataclass_fields__"):
            keys |= leaves(type(default), f"{prefix}.{item.name}")
        else:
            keys.add(f"{prefix}.{item.name}")
    return keys


class SchemaTests(unittest.TestCase):
    def test_every_key_of_the_file_is_on_a_page(self):
        """A key added to `Config` later fails here until it is given a page."""
        self.assertEqual(set(SCHEMA), {item.name for item in fields(Config)})
        for key, (page, how) in SCHEMA.items():
            with self.subTest(key):
                self.assertTrue(page)
                self.assertIn(how, (EDITABLE, READ_ONLY_KIND, FIELDS))

    def test_every_setting_of_a_section_shown_field_by_field_has_its_field(self):
        on_gateway_page = {f"gateway.{key}" for key in GATEWAY_SETTINGS} | {"gateway.task_timeouts"}
        for cls, prefix in ((MachineLimits, "limits"), (GatewayPolicy, "gateway"),
                            (GpuHealthSettings, "gpu_health")):
            with self.subTest(prefix):
                self.assertEqual(SCHEMA[prefix][1], FIELDS)
                missing = leaves(cls, prefix) - set(BY_KEY) - on_gateway_page
                self.assertEqual(missing, set())

    def test_every_read_only_key_is_shown_and_said_so(self):
        shown = {item.key for item in READ_ONLY}
        read_only = {key for key, (_page, how) in SCHEMA.items() if how == READ_ONLY_KIND}
        # host and port are shown fixed by the setting fields; engines' binary is editable.
        self.assertEqual(read_only - shown, {"host", "port"})
        for item in READ_ONLY:
            with self.subTest(item.key):
                self.assertIn(item.page, ("settings", "library", "storage", "lanes"))
                if item.key != "schema_version":   # written by AI-Lab, not by hand
                    self.assertIn("config.json", item.help)
                self.assertTrue(item.restart)


class ViewTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "config.json"
        self.path.write_text(json.dumps({
            "engines": {"comfyui": {"binary": "/opt/py", "comfyui": "/opt/ComfyUI/main.py"}},
            "providers": {"anthropic": {"cli": "claude", "api_key": "sk-secret",
                                        "env": {"AUTH_TOKEN": "t0k3n", "HOME": "/h"}}},
            "model_notes": {"qwen": {"short": "fast"}},
        }))
        self.view = FileSettings(ConfigStore(self.path))

    def section(self, page, key):
        return next(item for item in self.view.view(page)["sections"] if item["key"] == key)

    def test_a_page_gets_only_its_own_sections(self):
        self.assertEqual({item["key"] for item in self.view.view("library")["sections"]},
                         {"images", "downloads", "model_notes"})
        self.assertEqual(self.section("library", "model_notes")["value"],
                         {"qwen": {"short": "fast"}})

    def test_a_secret_is_never_shown_only_that_it_is_set(self):
        providers = self.section("lanes", "providers")["value"]["anthropic"]
        self.assertEqual(providers["api_key"], "(set)")
        self.assertEqual(providers["env"], {"AUTH_TOKEN": "(set)", "HOME": "/h"})
        self.assertNotIn("sk-secret", json.dumps(self.view.view()))
        self.assertNotIn("t0k3n", json.dumps(self.view.view()))
        self.assertEqual(masked({"password": ""}), {"password": "(empty)"})
        self.assertEqual(masked({"token_warn_after_days": 30}), {"token_warn_after_days": 30})

    def test_every_shape_a_secret_takes_is_hidden(self):
        """The reviewer's ten shapes (FIX FIRST on 3d4b0e6), each value hidden."""
        shapes = {
            "list under a secret key": ({"tokens": ["s1"]}, {"tokens": "(set)"}),
            "object under a secret key": ({"auth": {"user": "u", "pass": "s2"}},
                                          {"auth": "(set)"}),
            "private_key": ({"private_key": "s3"}, {"private_key": "(set)"}),
            "key": ({"key": "s4"}, {"key": "(set)"}),
            "arguments": ({"args": ["--api-key", "s5", "--token=s6", "--port", "8"]},
                          {"args": ["--api-key", "(set)", "--token=(set)", "--port", "8"]}),
            "url": ({"url": "https://user:s7@host/x"}, {"url": "https://user:(set)@host/x"}),
            "environment": ({"env": {"OPENAI_KEY": "s8", "HOME": "/h"}},
                            {"env": {"OPENAI_KEY": "(set)", "HOME": "/h"}}),
            "cookie": ({"headers": {"Cookie": "s9"}}, {"headers": {"Cookie": "(set)"}}),
            "bearer": ({"headers": {"X": "Bearer s10"}}, {"headers": {"X": "Bearer (set)"}}),
            "hf token and authorization": ({"HF_TOKEN": "s11", "Authorization": "s12"},
                                           {"HF_TOKEN": "(set)", "Authorization": "(set)"}),
            "a token alone as a URL's user": ({"source": "https://s13@github.com/x.git"},
                                              {"source": "https://(set)@github.com/x.git"}),
            "a flag inside one command line": ({"cli": "tool --password s14 --x y --token=s15"},
                                               {"cli": "tool --password (set) --x y --token=(set)"}),
        }
        for name, (raw, expected) in shapes.items():
            with self.subTest(name):
                self.assertEqual(masked(raw), expected)
        everything = json.dumps(masked({name: raw for name, (raw, _) in shapes.items()}))
        for number in range(1, 16):
            self.assertNotRegex(everything, rf"\bs{number}\b")
        self.assertEqual(masked({"token_warn_after_days": 30, "keep": "x", "keys_order": "y"}),
                         {"token_warn_after_days": 30, "keep": "x", "keys_order": "y"})
        plain = {"url": "https://host/x", "cli": "tool --model m1  --port 8", "path": "/a b"}
        self.assertEqual(masked(plain), plain)

    def test_an_engine_shows_what_it_has_beside_its_program(self):
        engines = self.section("settings", "engines")["value"]
        self.assertEqual(engines, {"comfyui": {"comfyui": "/opt/ComfyUI/main.py"}})

    def test_the_schema_lists_every_key_once(self):
        rows = FileSettings.schema()
        self.assertEqual([row["key"] for row in rows], list(SCHEMA))


if __name__ == "__main__":
    unittest.main()
