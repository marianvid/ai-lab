"""Versioned on-disk configuration upgrades.

Migrations operate on plain JSON values before domain objects are created. They
leave unknown fields intact so a save never erases settings from a newer peer.
"""

from __future__ import annotations

SCHEMA_VERSION = 1


def migrate(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("AI-Lab configuration must be a JSON object")
    version = raw.get("schema_version", 0)
    if type(version) is not int or version < 0:
        raise ValueError("schema_version must be a non-negative integer")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Configuration schema {version} is newer than this AI-Lab "
            f"release supports ({SCHEMA_VERSION}); upgrade the application")
    document = dict(raw)
    while version < SCHEMA_VERSION:
        if version == 0:
            # Version zero is the original unversioned layout. Its fields
            # already match version one; the explicit marker enables future
            # migrations without guessing which layout was written.
            document["schema_version"] = 1
        version += 1
    return document
