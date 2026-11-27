"""A YuE2 song's files, and moving them between AI-Labs.

A service (ARCHITECTURE.md): the bundle a song travels as (`bundles.py`: its
manifest, the tar, checking a received one file by file, the rename into
place), a pull from another AI-Lab through this machine's scheduler door and a
put-back of a kept bundle (`transfers.py`), this machine's list of songs
(`listing.py`), and what YuE2 Studio decides about a song: busy or not, and how
it is removed (`studio.py`). The routes (`api/routes/music_transfers.py`) only read the
request and hand it here. Every refusal names its own HTTP status
(`errors.py`).
"""
