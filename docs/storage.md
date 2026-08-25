# Storage

Storage is where disk space outside the model library is made visible and can
be reclaimed. It has two subjects:

- downloaded package and generated-kernel caches, plus explicitly named
  leftovers from interrupted work;
- inactive engine environments and compiled builds kept as rollback versions.

Models do not appear here. Their files, deletion rules and confirmations stay
in Library, so there is one place responsible for model storage.

## An explicit allow-list

The page cannot send an arbitrary path to the manager. Every removable cache
or leftover is declared under `storage.reclaimable` in configuration with an
id, a fixed path, a kind and a plain-language consequence. A delete request
sends only that id. Broad paths are refused when configuration is loaded.
The configuration is a list of places AI-Lab knows how to interpret, not a
list of rows to draw: missing leftovers and empty caches do not appear.

Clearing a cache may make the next engine start or update slower because files
must be downloaded or generated again. It never removes model weights. A
leftover is a specifically identified incomplete directory and is deleted as
such; AI-Lab does not scan the disk and guess what is unwanted.

## Rollback versions

Engines install or compile a new version beside the one in use. Storage shows
the inactive environments and builds because they are simultaneously a
recovery option and recoverable disk space. The active version cannot be
deleted. Nothing removes an inactive version automatically: only the person
who has decided the new version has proved itself can know that the rollback
copy is no longer needed.

---

[← all documents](../README.md)  ·  [Library](library.md)  ·  [Updating an engine](engines.md)
