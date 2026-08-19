"""Tests, laid out to mirror the package.

    tests/support/   fixtures and stand-ins shared by everything
    tests/hosts/     one file per host implementation
    tests/engines/   one file per engine
    tests/api/       routing and the event stream
    tests/test_*.py  one file per service

Run them with:

    python3 -m unittest discover -t . -s tests

The `-t .` matters: it makes `tests` a package, so the shared fixtures can be
imported instead of duplicated.

Two rules keep the suite fast and honest. Anything below the service layer is
passed in rather than imported, so a test supplies a fake host and a fake
engine and never needs a GPU. Filesystem tests build a directory of empty
files, because a model is defined by its names and sizes — real weights are
never required.
"""
