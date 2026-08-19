"""Getting models from Hugging Face.

Two separate jobs, two files:

    huggingface.py   talking to the remote API and reading its listings
    transfers.py     the queue, and moving the bytes

The unit of work is a *model*, never a file. Picking a model resolves to the
whole set — every shard plus the tokenizer and config beside it — and the set
is queued together. Four shards out of five produce a model that fails to load
with an unhelpful message, so partial selection is not offered.
"""

from .huggingface import HuggingFaceClient, RemoteFile, RemoteSet
from .transfers import DownloadManager, Transfer

__all__ = ["HuggingFaceClient", "RemoteFile", "RemoteSet",
           "DownloadManager", "Transfer"]
