"""AI-Lab: a web manager for local inference engines.

The package is organised in one direction only:

    web -> api -> services -> engines -> hosts

`services` are catalog, runtime, downloads and settings. Nothing imports
upward, and no service imports another service. Shared data structures live in
`types.py` so services never import each other just to borrow a type.

See ARCHITECTURE.md for what each module is responsible for.
"""

__version__ = "0.2.0"
