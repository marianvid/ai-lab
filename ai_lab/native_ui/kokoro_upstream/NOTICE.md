The interface in `app.py` is adapted from the official hexgrad/Kokoro demo.

Upstream: https://github.com/hexgrad/kokoro/tree/dfb907a02bba8152ca444717ca5d78747ccb4bec/demo

Copyright 2025 Kokoro contributors. Licensed under Apache-2.0. AI-Lab's
adaptation removes Hugging Face Spaces/ZeroGPU integration, uses the model
already resident in the AI-Lab process, and resolves voice packs from the
configured local checkpoint directory. The Generate, Stream, pronunciation,
US/UK voice, speed, token and sample-text features remain available.
