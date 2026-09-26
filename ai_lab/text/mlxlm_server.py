"""Start Apple's mlx-lm text server, with two changes AI-Lab needs.

This file runs inside the mlx-lm environment, not inside the manager. It
imports only mlx-lm, and it passes every command-line option straight to
mlx-lm's own server, so what the server accepts is decided by the installed
mlx-lm version.

The two changes, and why:

1. **`/health` says "not yet" until the weights are in memory.** mlx-lm opens
   its HTTP port at once and reads the model afterwards, in the background
   thread that later produces the answers. Left alone, its `/health` page
   answers "ok" while nothing is loaded, and the manager would believe a load
   had finished before it had. Here `/health` answers 503 ("service
   unavailable") until that thread has finished reading the model.

   The reading itself is left where mlx-lm puts it, in that background
   thread. Reading the model in the main thread first was tried: MLX ties
   some of a model's working memory to the thread that created it, and
   Gemma 4 then failed on its first request with "There is no Stream(gpu, 1)
   in current thread", while the health page kept answering.

2. **Any model name in a request means the one model this process holds.**
   mlx-lm treats the `model` field of a request as something to load — a
   folder or a Hugging Face name. The manager's gateway fills that field with
   the model's own short name, which is neither, so mlx-lm would go looking
   for it on the internet. One process serves one model here, so the name is
   ignored and the model given with `--model` answers every request.
"""

from __future__ import annotations

import sys
import threading


def main() -> None:
    from mlx_lm import server

    loaded = threading.Event()
    provider_class = server.ModelProvider
    handler_class = server.APIHandler

    load = provider_class.load
    load_default = provider_class.load_default
    health = handler_class.handle_health_check

    def only_this_model(self, model_path, adapter_path=None, draft_model_path=None):
        return load(self, "default_model", adapter_path, draft_model_path)

    def load_default_then_say_so(self):
        load_default(self)
        if not loaded.is_set():
            print("AI-Lab: weights loaded", flush=True)
            loaded.set()

    def health_once_loaded(self):
        if loaded.is_set():
            return health(self)
        self._set_completion_headers(503)
        self.end_headers()
        self.wfile.write(b'{"status": "loading"}')
        self.wfile.flush()

    provider_class.load = only_this_model
    provider_class.load_default = load_default_then_say_so
    handler_class.handle_health_check = health_once_loaded
    sys.argv[0] = "mlx_lm.server"
    server.main()


if __name__ == "__main__":
    main()
