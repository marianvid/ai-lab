"""One request to an engine, sent and read on a thread of its own.

The gateway used to wait for an engine inside a blocking read. While blocked
it could not notice anything else: a client that had left, an engine that had
stopped working. A non-streaming request whose client gave up kept its place
on the card until the engine finished writing for nobody (measured on
qwen38-nvfp4, 22 October 2026: still running 25 s after the client left).

Here a thread does the blocking and puts what arrives in a queue — the
answer's head, then its pieces, then the end or the failure. The gateway's
own thread waits on that queue a second at a time and looks around between
waits (`passthrough._Answer`). To stop, it closes the connection: vLLM sees
its client leave and stops the request within a second, streaming or not
(measured 22 October 2026).
"""

from __future__ import annotations

import contextlib
import http.client
import queue
import socket
import threading
from urllib.parse import urlsplit

# The size of one read: whatever has arrived, up to this much.
CHUNK = 8192
# How many pieces may wait for the gateway to pass them on before the reading
# thread pauses: a slow client slows the engine's connection, as before.
QUEUED_PIECES = 256
# How often a paused reading thread checks whether it has been told to stop.
PAUSE_CHECK_S = 1.0

# What the reading thread hands over, in order.
HEAD = "head"
PIECE = "piece"
END = "end"
FAILED = "failed"


class Upstream:
    """The connection to the engine for one request, and the thread reading it.

    Connecting happens at once, on the caller's thread: an engine that is not
    listening is an error straight away. `read_limit_s` bounds any single
    read on the connection, as a last resort behind the gateway's own limits.
    """

    def __init__(self, url: str, data: bytes, content_type: str, read_limit_s: float) -> None:
        parts = urlsplit(url)
        self._path = parts.path + (f"?{parts.query}" if parts.query else "")
        self._connection = http.client.HTTPConnection(
            parts.hostname or "", parts.port, timeout=read_limit_s)
        self._connection.connect()
        self._stopped = threading.Event()
        self.events: queue.Queue = queue.Queue(maxsize=QUEUED_PIECES)
        self._thread = threading.Thread(target=self._run, args=(data, content_type),
                                        name="engine-answer", daemon=True)
        self._thread.start()

    def _run(self, data: bytes, content_type: str) -> None:
        """Send the request and pass on everything that comes back."""
        try:
            self._connection.request("POST", self._path, body=data,
                                     headers={"Content-Type": content_type})
            response = self._connection.getresponse()
            self._hand(HEAD, response)
            while piece := response.read1(CHUNK):
                self._hand(PIECE, piece)
            self._hand(END, None)
        except Exception as error:
            self._hand(FAILED, error)

    def _hand(self, kind: str, value) -> None:
        """Queue one event; give up quietly once told to stop."""
        while not self._stopped.is_set():
            try:
                self.events.put((kind, value), timeout=PAUSE_CHECK_S)
                return
            except queue.Full:
                continue

    def ended(self, timeout: float) -> bool:
        """Whether the reading thread has finished, waiting up to `timeout` seconds."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def stop(self) -> None:
        """Close the connection. The engine sees its client leave and stops the request."""
        self._stopped.set()
        sock = self._connection.sock
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        self._connection.close()
