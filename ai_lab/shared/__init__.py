"""Shared files that anything in the manager may import (ARCHITECTURE.md, "Three kinds of file").

They hold data shapes, rules or arithmetic, and decide nothing about what to
do. `unexpected.py`: the failures each kind of reading expects, and the one
line said when another one happens. `public_services.py`: the public addresses
(PyPI, Hugging Face, GitHub), read from `public_services.json`. `song_names.py`:
the names a YuE2 song and a window call go by (the YuE2 adapter keeps an equal
copy in its own process).
"""
