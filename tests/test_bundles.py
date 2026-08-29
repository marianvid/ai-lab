"""Models assembled from named upstream files.

An image model is several files from several folders, and sometimes from
several repositories. Nothing upstream says which of them belong together, so
the grouping is written down. These tests cover what that declaration is
allowed to say, what a declaration turns into when it meets a real listing,
and what the download queue does with one — including everything it must
refuse.
"""

import io
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.catalog import Catalog
from ai_lab.config import Repository
from ai_lab.downloads.bundles import parse
from ai_lab.downloads.huggingface import (HuggingFaceClient, RemoteFile,
                                          bundle_sets)
from ai_lab.downloads.transfers import WORKING, DownloadManager, State

QWEN = "Comfy-Org/Qwen-Image_ComfyUI"
EDIT = "Comfy-Org/Qwen-Image-Edit_ComfyUI"

DECLARED = {
    "name": "qwen-image-2512-nvfp4",
    "repo": QWEN,
    "format": "comfyui",
    "task": "image-generation",
    "components": [
        {"role": "diffusion_model",
         "path": "split_files/diffusion_models/qwen_image_nvfp4.safetensors"},
        {"role": "text_encoder",
         "path": "split_files/text_encoders/qwen_2.5_vl_7b_nvfp4.safetensors"},
        {"role": "vae", "path": "split_files/vae/qwen_image_vae.safetensors"},
    ],
}


def listing(*paths):
    return [RemoteFile(path=path, size_bytes=size, sha256=digest)
            for path, size, digest in paths]


class DeclarationTests(unittest.TestCase):
    def test_a_bundle_names_its_parts_exactly(self):
        bundle = parse([DECLARED])[0]
        self.assertEqual(bundle.name, "qwen-image-2512-nvfp4")
        self.assertEqual([item.role for item in bundle.components],
                         ["diffusion_model", "text_encoder", "vae"])
        self.assertEqual(bundle.components[2].file_name, "qwen_image_vae.safetensors")

    def test_a_part_may_come_from_another_repository(self):
        """Qwen-Image-Edit is the real case: its own model, the other's encoder."""
        bundle = parse([{**DECLARED, "name": "edit", "repo": EDIT,
                         "components": [
                             {"role": "diffusion_model", "path": "a/edit.safetensors"},
                             {"role": "text_encoder", "repo": QWEN,
                              "path": "b/encoder.safetensors"}]}])[0]
        self.assertEqual(bundle.repos, (EDIT, QWEN))

    def test_a_name_that_is_a_path_is_refused(self):
        """The name becomes a folder, so it must not be able to choose one."""
        for name in ("../escape", "images/thing", ".hidden", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                parse([{**DECLARED, "name": name}])
            self.assertIn("name", str(caught.exception))

    def test_a_path_that_leaves_the_repository_is_refused(self):
        for path in ("../secrets/key", "/etc/passwd", "a/../../b", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                parse([{**DECLARED, "components": [
                    {"role": "vae", "path": path}]}])

    def test_a_part_that_is_not_a_part_of_a_model_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            parse([{**DECLARED, "components": [
                {"role": "sidecar", "path": "a/b.safetensors"}]}])
        self.assertIn("sidecar", str(caught.exception))

    def test_two_parts_cannot_share_a_file_name(self):
        """They land in one folder, so the second would overwrite the first."""
        with self.assertRaises(ValueError) as caught:
            parse([{**DECLARED, "components": [
                {"role": "vae", "path": "one/model.safetensors"},
                {"role": "text_encoder", "path": "two/model.safetensors"}]}])
        self.assertIn("model.safetensors", str(caught.exception))

    def test_two_bundles_cannot_share_a_name(self):
        with self.assertRaises(ValueError):
            parse([DECLARED, DECLARED])

    def test_a_bundle_with_no_parts_is_refused(self):
        with self.assertRaises(ValueError):
            parse([{**DECLARED, "components": []}])


class MatchingTheListingTests(unittest.TestCase):
    def test_a_bundle_becomes_one_set_spanning_two_repositories(self):
        bundles = parse([{**DECLARED, "name": "edit", "repo": EDIT,
                          "components": [
                              {"role": "diffusion_model", "path": "a/edit.safetensors"},
                              {"role": "text_encoder", "repo": QWEN,
                               "path": "b/encoder.safetensors"}]}])
        sets = bundle_sets(bundles, {
            EDIT: listing(("a/edit.safetensors", 20, "aaa")),
            QWEN: listing(("b/encoder.safetensors", 6, "bbb")),
        })
        self.assertEqual(len(sets), 1)
        self.assertTrue(sets[0].complete)
        self.assertEqual(sets[0].size_bytes, 26)
        self.assertEqual([item.repo for item in sets[0].files], [EDIT, QWEN])
        self.assertEqual([item.sha256 for item in sets[0].files], ["aaa", "bbb"])

    def test_each_part_is_fetched_from_its_own_repository(self):
        bundles = parse([{**DECLARED, "name": "edit", "repo": EDIT,
                          "components": [{"role": "vae", "repo": QWEN,
                                          "path": "b/vae.safetensors"}]}])
        remote = bundle_sets(bundles, {QWEN: listing(("b/vae.safetensors", 1, ""))})[0]
        self.assertIn(QWEN, remote.files[0].url(remote.repo))

    def test_a_part_upstream_has_renamed_makes_the_set_incomplete(self):
        """Silence here would publish a model with a hole in it."""
        remote = bundle_sets(parse([DECLARED]), {QWEN: listing(
            ("split_files/vae/qwen_image_vae.safetensors", 1, ""))})[0]
        self.assertFalse(remote.complete)
        self.assertEqual(len(remote.missing), 2)
        self.assertIn("qwen_image_nvfp4.safetensors", remote.missing[0])

    def test_the_bundle_is_listed_before_its_own_parts(self):
        client = HuggingFaceClient(opener=lambda url: [
            {"type": "file", "size": 3,
             "path": "split_files/vae/qwen_image_vae.safetensors",
             "lfs": {"oid": "vvv", "size": 3}}])
        sets = client.sets(QWEN, parse([{**DECLARED, "components": [
            {"role": "vae", "path": "split_files/vae/qwen_image_vae.safetensors"}]}]))
        self.assertEqual(sets[0].name, "qwen-image-2512-nvfp4")
        self.assertEqual(sets[0].files[0].sha256, "vvv")
        # The individual file is still there to be had on its own.
        self.assertIn("split_files/vae/qwen_image_vae", [item.name for item in sets[1:]])


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class DownloadingABundleTests(unittest.TestCase):
    """What the queue does with a set whose files come from several places."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.destination = self.root / "images" / "generation" / "qwen-image"
        self.fetched: list[str] = []

    def opener(self, payload=b"x" * 8, fails: str = ""):
        def open_range(url, resume_from):
            self.fetched.append(url)
            if fails and fails in url:
                raise ValueError("HTTP 500")
            return FakeResponse(payload[resume_from:])
        return open_range

    def remote(self, sha=("", "", ""), sizes=(8, 8, 8)):
        from ai_lab.downloads.huggingface import RemoteSet
        names = ("diffusion_models/a.safetensors", "text_encoders/b.safetensors",
                 "vae/c.safetensors")
        repos = (QWEN, QWEN, EDIT)
        return RemoteSet(
            repo=QWEN, name="qwen-image", format="comfyui",
            task="image-generation",
            files=tuple(RemoteFile(path=path, size_bytes=size, repo=repo, sha256=digest)
                        for path, size, repo, digest in zip(names, sizes, repos, sha)),
            roles={"a.safetensors": "diffusion_model"},
        )

    def wait(self, transfer, timeout=5.0):
        deadline = time.monotonic() + timeout
        while transfer.state in (State.QUEUED, State.RUNNING):
            if time.monotonic() > deadline:
                self.fail(f"transfer stuck in {transfer.state}")
            time.sleep(0.01)
        return transfer

    def working(self, transfer):
        return self.destination.parent / WORKING / transfer.id

    def test_every_part_lands_in_one_folder_as_one_model(self):
        manager = DownloadManager(opener=self.opener())
        transfer = self.wait(manager.enqueue(self.remote(), self.destination,
                                             storage_tier="benchmark"))
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual(sorted(item.name for item in self.destination.iterdir()),
                         ["a.safetensors", "b.safetensors", "c.safetensors"])
        # Flat, under their own names: the folders they sat in upstream are
        # not recreated, because ComfyUI looks a part up by file name.
        self.assertFalse((self.destination / "vae").exists())

    def test_the_catalog_sees_one_model_and_not_three(self):
        manager = DownloadManager(opener=self.opener())
        self.wait(manager.enqueue(self.remote(), self.destination))
        repository = Repository(id="images-comfyui-generation", name="Generation",
                                path=str(self.destination.parent),
                                format="comfyui", task="image-generation")
        models = Catalog().scan([repository])
        self.assertEqual([item.name for item in models], ["qwen-image"])
        self.assertEqual(len(models[0].files), 3)
        self.assertEqual(models[0].entrypoint, str(self.destination))

    def test_nothing_is_published_until_every_part_has_arrived(self):
        """Two parts of three is not a model, and must not look like one."""
        manager = DownloadManager(opener=self.opener(fails="vae"))
        transfer = self.wait(manager.enqueue(self.remote(), self.destination))
        self.assertEqual(transfer.state, State.FAILED)
        self.assertFalse(self.destination.exists(),
                         "a failed bundle must leave nothing behind under its name")
        # The bytes that did arrive are kept, out of sight, for the next try.
        self.assertTrue((self.working(transfer) / "a.safetensors").exists())

    def test_asking_again_continues_from_what_was_already_fetched(self):
        manager = DownloadManager(opener=self.opener(fails="vae"))
        first = self.wait(manager.enqueue(self.remote(), self.destination))
        self.assertEqual(first.state, State.FAILED)
        self.fetched.clear()
        second = self.wait(DownloadManager(opener=self.opener()).enqueue(
            self.remote(), self.destination))
        self.assertEqual(second.state, State.DONE)
        # Only the part that failed was asked for a second time.
        self.assertEqual(self.fetched, [
            "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/"
            "resolve/main/vae/c.safetensors"])

    def test_a_cancelled_bundle_publishes_nothing_and_keeps_its_bytes(self):
        release = []

        def open_range(url, resume_from):
            if "text_encoders" in url:
                release[0].cancel(release[1].id)
            return FakeResponse(b"x" * 8)

        manager = DownloadManager(opener=open_range)
        transfer = manager.enqueue(self.remote(), self.destination)
        release[:] = [manager, transfer]
        self.wait(transfer)
        self.assertEqual(transfer.state, State.CANCELLED)
        self.assertFalse(self.destination.exists())
        self.assertTrue((self.working(transfer) / "a.safetensors").exists())

    def test_a_file_that_does_not_match_its_checksum_is_never_published(self):
        """A truncated download can still end in a clean-looking file."""
        manager = DownloadManager(opener=self.opener())
        transfer = self.wait(manager.enqueue(
            self.remote(sha=("wrong", "", "")), self.destination))
        self.assertEqual(transfer.state, State.FAILED)
        self.assertIn("checksum", transfer.error)
        self.assertFalse(self.destination.exists())

    def test_a_file_of_the_wrong_size_is_never_published(self):
        manager = DownloadManager(opener=self.opener())
        transfer = self.wait(manager.enqueue(
            self.remote(sizes=(8, 8, 99)), self.destination))
        self.assertEqual(transfer.state, State.FAILED)
        self.assertIn("upstream says", transfer.error)
        self.assertFalse(self.destination.exists())

    def test_the_report_says_what_was_actually_checked(self):
        """Two of three files carry a published hash; the third has only a size."""
        manager = DownloadManager(opener=self.opener())
        digest = __import__("hashlib").sha256(b"x" * 8).hexdigest()
        transfer = self.wait(manager.enqueue(
            self.remote(sha=(digest, digest, "")), self.destination))
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual(transfer.json()["checked_hash"], 2)
        self.assertEqual(transfer.json()["checked_size"], 3)

    def test_an_incomplete_bundle_is_refused_before_anything_is_fetched(self):
        from ai_lab.downloads.huggingface import RemoteSet
        manager = DownloadManager(opener=self.opener())
        incomplete = RemoteSet(repo=QWEN, name="qwen-image", format="comfyui",
                               files=(), complete=False,
                               missing=(f"{QWEN}:split_files/vae/gone.safetensors",))
        with self.assertRaises(ValueError) as caught:
            manager.enqueue(incomplete, self.destination)
        self.assertIn("missing", str(caught.exception))
        self.assertEqual(self.fetched, [])

    def test_the_working_folder_is_not_a_model(self):
        """Half a bundle sitting beside the library must stay out of it."""
        manager = DownloadManager(opener=self.opener(fails="vae"))
        self.wait(manager.enqueue(self.remote(), self.destination))
        repository = Repository(id="images-comfyui-generation", name="Generation",
                                path=str(self.destination.parent),
                                format="comfyui", task="image-generation")
        self.assertEqual(Catalog().scan([repository]), [])


if __name__ == "__main__":
    unittest.main()
