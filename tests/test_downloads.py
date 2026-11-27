import io
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.downloads.huggingface import RemoteFile, RemoteSet
from ai_lab.downloads.transfers import (WORKING, DownloadManager, RangeRefusedError, State,
                                        transfer_id)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def remote_set(complete=True, sizes=(("a.gguf", 8),), repo="org/model",
               name="model"):
    return RemoteSet(
        repo=repo, name=name, format="gguf",
        files=tuple(RemoteFile(path=name, size_bytes=size) for name, size in sizes),
        complete=complete, missing=() if complete else ("a-00002-of-00002",),
    )


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.destination = Path(self._temporary.name) / "gguf"
        self.addCleanup(self._temporary.cleanup)
        self.requested: list[tuple[str, int]] = []

    def opener(self, payload=b"x" * 8):
        def open_range(url, resume_from):
            self.requested.append((url, resume_from))
            return FakeResponse(payload[resume_from:])
        return open_range

    def wait(self, manager, transfer, timeout=5.0):
        deadline = time.monotonic() + timeout
        while transfer.state in (State.QUEUED, State.RUNNING):
            if time.monotonic() > deadline:
                self.fail(f"transfer stuck in {transfer.state}")
            time.sleep(0.01)
        return transfer

    def test_a_whole_set_lands_on_disk(self):
        manager = DownloadManager(opener=self.opener())
        transfer = manager.enqueue(remote_set(sizes=(("a.gguf", 8), ("tokenizer.json", 8))),
                                   self.destination)
        self.wait(manager, transfer)
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual(sorted(item.name for item in self.destination.iterdir()),
                         ["a.gguf", "tokenizer.json"])

    def test_an_incomplete_set_is_refused_up_front(self):
        """Downloading four shards of five produces a model that cannot load."""
        manager = DownloadManager(opener=self.opener())
        with self.assertRaises(ValueError) as caught:
            manager.enqueue(remote_set(complete=False), self.destination)
        self.assertIn("missing", str(caught.exception))

    def test_nothing_is_visible_until_it_is_complete(self):
        """A partial file must never look like a model to the catalog."""
        started = []

        def open_range(url, resume_from):
            started.append(url)
            return FakeResponse(b"y" * 8)

        manager = DownloadManager(opener=open_range)
        transfer = manager.enqueue(remote_set(), self.destination)
        self.wait(manager, transfer)
        self.assertEqual([item.name for item in self.destination.iterdir()], ["a.gguf"])

    def test_an_interrupted_file_resumes_instead_of_restarting(self):
        """Bytes already fetched are kept — and they were never on show.

        The half of a file that arrived before the connection dropped sits in
        the working folder beside the repository, not in the repository, so
        the library never had a chance to see it. Asking again continues from
        where it stopped rather than fetching those bytes twice.
        """
        working = self.destination.parent / WORKING / transfer_id("org/model", "model")
        working.mkdir(parents=True)
        (working / "a.gguf.part").write_bytes(b"x" * 5)
        self.assertFalse(self.destination.exists(),
                         "a partial download must not create the model's folder")
        manager = DownloadManager(opener=self.opener())
        transfer = manager.enqueue(remote_set(), self.destination)
        self.wait(manager, transfer)
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual(self.requested[0][1], 5)          # asked to resume at byte 5
        self.assertEqual((self.destination / "a.gguf").read_bytes(), b"x" * 8)

    def test_a_file_with_a_wrong_checksum_is_removed_so_asking_again_fetches_it(self):
        # Audit m9: kept, it was skipped on every retry and failed for ever.
        import hashlib
        good = b"x" * 8
        files = (RemoteFile(path="a.gguf", size_bytes=8,
                            sha256=hashlib.sha256(good).hexdigest()),)
        remote = RemoteSet(repo="org/model", name="model", format="gguf", files=files,
                           complete=True, missing=())
        manager = DownloadManager(opener=lambda *_: FakeResponse(b"y" * 8))
        failed = self.wait(manager, manager.enqueue(remote, self.destination))
        self.assertEqual(failed.state, State.FAILED)
        self.assertIn("removed", failed.error)
        manager = DownloadManager(opener=self.opener(good))
        done = self.wait(manager, manager.enqueue(remote, self.destination))
        self.assertEqual(done.state, State.DONE)
        self.assertEqual((self.destination / "a.gguf").read_bytes(), good)

    def test_a_resume_answered_with_the_whole_file_starts_the_file_over(self):
        # Audit L27: a 200 to a Range request is the whole file, not the rest.
        working = self.destination.parent / WORKING / transfer_id("org/model", "model")
        working.mkdir(parents=True)
        (working / "a.gguf.part").write_bytes(b"x" * 5)

        class Whole(FakeResponse):
            status = 200

        manager = DownloadManager(opener=lambda *_: Whole(b"z" * 8))
        transfer = self.wait(manager, manager.enqueue(remote_set(), self.destination))
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual((self.destination / "a.gguf").read_bytes(), b"z" * 8)

    def test_a_refused_range_starts_the_file_over(self):
        # Audit L27: a part already whole was asked "bytes=8-", 416, for ever.
        working = self.destination.parent / WORKING / transfer_id("org/model", "model")
        working.mkdir(parents=True)
        (working / "a.gguf.part").write_bytes(b"x" * 8)

        def open_range(url, resume_from):
            self.requested.append((url, resume_from))
            if resume_from:
                raise RangeRefusedError("Download failed with HTTP 416")
            return FakeResponse(b"w" * 8)

        manager = DownloadManager(opener=open_range)
        transfer = self.wait(manager, manager.enqueue(remote_set(), self.destination))
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual([start for _, start in self.requested], [8, 0])
        self.assertEqual((self.destination / "a.gguf").read_bytes(), b"w" * 8)

    def test_a_model_cancelled_while_queued_and_asked_again_downloads_once(self):
        # Audit L29: its id was in the queue twice and ran twice.
        release = threading.Event()

        class Slow(FakeResponse):
            def read(self, size=-1):
                release.wait(2)
                return super().read(size)

        arrived: list = []
        manager = DownloadManager(opener=lambda *_: Slow(b"x" * 8), arrived=arrived.append)
        first = manager.enqueue(remote_set(name="first"), self.destination)
        queued = manager.enqueue(remote_set(), self.destination)
        manager.cancel(queued.id)
        again = manager.enqueue(remote_set(), self.destination)
        release.set()
        self.wait(manager, first)
        self.wait(manager, again)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and len(arrived) < 2:
            time.sleep(0.01)
        time.sleep(0.2)
        self.assertEqual(len(arrived), 2, "the model asked for again ran twice, or not at all")

    def test_two_models_never_share_a_transfer_id(self):
        # Audit L30: "org/m" + "a_b" and "org/m" + "a/b" were both org_m_a_b.
        self.assertNotEqual(transfer_id("org/m", "a_b"), transfer_id("org/m", "a/b"))
        self.assertTrue(transfer_id("org/m", "a/b").startswith("org_m_a_b-"))

    def test_an_already_present_file_is_not_fetched_again(self):
        self.destination.mkdir(parents=True)
        (self.destination / "a.gguf").write_bytes(b"x" * 8)
        manager = DownloadManager(opener=self.opener())
        transfer = manager.enqueue(remote_set(), self.destination)
        self.wait(manager, transfer)
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual(self.requested, [])

    def test_progress_reaches_a_hundred_percent(self):
        manager = DownloadManager(opener=self.opener())
        transfer = manager.enqueue(remote_set(), self.destination)
        self.wait(manager, transfer)
        self.assertEqual(transfer.percent, 100.0)
        self.assertEqual(transfer.files_done, transfer.files_total)

    def test_a_failure_is_recorded_not_raised(self):
        def broken(url, resume_from):
            raise ValueError("HTTP 404")

        manager = DownloadManager(opener=broken)
        transfer = manager.enqueue(remote_set(), self.destination)
        self.wait(manager, transfer)
        self.assertEqual(transfer.state, State.FAILED)
        self.assertIn("404", transfer.error)

    def test_the_same_model_cannot_be_queued_twice(self):
        manager = DownloadManager(opener=self.opener())
        transfer = manager.enqueue(remote_set(), self.destination)
        with self.assertRaises(ValueError):
            manager.enqueue(remote_set(), self.destination)
        self.wait(manager, transfer)          # let the worker finish before cleanup

    def test_listing_reports_every_transfer(self):
        manager = DownloadManager(opener=self.opener())
        transfer = manager.enqueue(remote_set(), self.destination)
        self.wait(manager, transfer)
        rows = manager.list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "done")

    def test_cancelling_an_unknown_transfer_raises(self):
        with self.assertRaises(KeyError):
            DownloadManager(opener=self.opener()).cancel("nope")

    def test_a_running_transfer_becomes_cancelled_immediately_and_stays_that_way(self):
        started = threading.Event()
        release = threading.Event()

        class SlowResponse(FakeResponse):
            def read(self, size=-1):
                started.set()
                release.wait(2)
                return super().read(size)

        manager = DownloadManager(opener=lambda *_: SlowResponse(b"x" * 8))
        transfer = manager.enqueue(remote_set(), self.destination)
        self.assertTrue(started.wait(2), "the download never started")

        manager.cancel(transfer.id)
        self.assertEqual(transfer.state, State.CANCELLED)
        release.set()
        deadline = time.monotonic() + 2
        while manager._worker and manager._worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(transfer.state, State.CANCELLED,
                         "the worker changed a cancellation into a failure")


if __name__ == "__main__":
    unittest.main()


class AfterItArrives(DownloadTests):
    """A model that has landed is read straight away, on this thread."""

    def test_a_finished_download_says_so(self):
        landed = []
        manager = DownloadManager(opener=self.opener(),
                                  arrived=lambda where: landed.append(where))
        transfer = manager.enqueue(remote_set(sizes=(("a.gguf", 8),)),
                                   self.destination)
        self.wait(manager, transfer)
        self.assertEqual(transfer.state, State.DONE)
        self.assertEqual(landed, [self.destination])

    def test_a_failed_download_does_not(self):
        def broken(url, resume_from):
            raise OSError("the network went away")

        landed = []
        manager = DownloadManager(opener=broken,
                                  arrived=lambda where: landed.append(where))
        transfer = manager.enqueue(remote_set(sizes=(("a.gguf", 8),)),
                                   self.destination)
        self.wait(manager, transfer)
        self.assertEqual(transfer.state, State.FAILED)
        self.assertEqual(landed, [], "read a model that never finished arriving")

    def test_a_reader_that_throws_does_not_break_the_queue(self):
        """The download worked. Whatever this was for can wait; the bytes cannot."""
        def angry(where):
            raise RuntimeError("could not read the new model")

        manager = DownloadManager(opener=self.opener(), arrived=angry)
        first = manager.enqueue(remote_set(sizes=(("a.gguf", 8),)), self.destination)
        self.wait(manager, first)
        self.assertEqual(first.state, State.DONE)

        # The worker thread is still alive and still takes work.
        second = manager.enqueue(remote_set(name="second", sizes=(("b.gguf", 8),)),
                                 self.destination)
        self.wait(manager, second)
        self.assertEqual(second.state, State.DONE)
