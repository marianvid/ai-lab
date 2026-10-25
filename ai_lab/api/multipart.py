"""Kept so older imports still work: the multipart reader now lives one level
down, in `ai_lab/multipart.py`, where the image job queue may use it too.

The image calls (`media/images.py`) sit above the gateway, while `api/` sits
beside it. Borrowing from `api/` broke the rule that imports only point
downward. The reader itself holds no web logic, so it moved to the shared
level and this file just passes it through.
"""

from __future__ import annotations

from ..multipart import MultipartBody

__all__ = ["MultipartBody"]
