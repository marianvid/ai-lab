Higgs Audio v3 playground files are from sgl-project/sglang-omni,
commit `89e60d0bf216bacd0e072d79f969550469614662`, Apache-2.0.

Upstream: https://github.com/sgl-project/sglang-omni/tree/89e60d0bf216bacd0e072d79f969550469614662/playground/higgs

`playground/http_utils.py` is the upstream shared favicon helper. The
playground source and assets are vendored unchanged; AI-Lab supplies the
existing SGLang-Omni worker URL at launch, rather than starting a second
model server.

On a Mac, engine `higgs_local` has no SGLang-Omni worker, and its runtime has
neither FastAPI nor SGLang-Omni. There `ai_lab/speech/higgs_playground.py`
serves the same unchanged `frontend/` files and answers the page's requests
from the model already loaded in the speech host; `app.py` and
`http_utils.py` are not used on that path.
