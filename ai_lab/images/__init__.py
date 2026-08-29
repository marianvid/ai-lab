"""Isolated adapters for image-shaped work: OCR today, generation and editing later.

Kept separate from `ai_lab/audio` because the two share nothing but the
pattern — a small stand-alone HTTP process, started by an engine's
`LaunchPlan`, that this manager never imports directly. See
`ai_lab/audio/server.py` for why the isolation exists.
"""
