"""Higgs TTS 3 in-process, through its transformers port.

Where SGLang-Omni cannot run — Apple silicon has no backend for it — the same
weights load through the transformers port published beside them, which
carries its own architecture code. Generation is the model's own
`generate_speech`; nothing of a third-party interface is involved.

The checkpoint directory holds the weights, the port's code and, under
`codec/`, the audio codec the port would otherwise fetch from the Hub at
first use. AI-Lab runs offline, so the codec is pointed at locally.
"""
from __future__ import annotations

import io
from pathlib import Path
from threading import Lock

from .contract import apply_seed, validate_payload, wav_result

SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 27  # measured on the M3 Max: 155 frames made 5.9 s


def frame_limit(text: str) -> int:
    """Longest plausible answer for this text, in audio frames.

    Higgs occasionally fails to stop and speaks on to its token limit —
    eighty seconds for one sentence was measured. A limit derived from the
    text keeps a runaway to a few seconds instead of a minute: about 0.13 s
    per character plus three seconds of slack, never above the model's own
    ceiling.
    """
    seconds = 0.13 * max(len(text), 1) + 3.0
    return max(64, min(2048, int(seconds * FRAMES_PER_SECOND)))


class HiggsLocalBackend:
    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        codec = model_path / "codec"
        if not (model_path / "config.json").is_file() or not codec.is_dir():
            raise ValueError("Higgs checkpoint or its codec/ directory is absent")
        self.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, dtype=torch.bfloat16)
        model.config.audio_tokenizer_id = str(codec)
        self.model = model.to(self.device).eval()
        self.model.get_audio_codec()  # load the codec now, not on first request
        self.model_name = model_path.name
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        request = validate_payload(body, seed=True, reference=True)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        if request["instruction"]:
            raise ValueError("Higgs accepts style controls inline in the text")
        if request["speaker"]:
            raise ValueError("Higgs has no named voices; clone one with reference audio")
        import torch

        clone = {}
        if request["reference_audio"]:
            import soundfile as sf

            samples, rate = sf.read(io.BytesIO(request["reference_audio"]),
                                    dtype="float32", always_2d=True)
            clone = {"reference_audio": torch.from_numpy(samples).mean(dim=1),
                     "reference_sample_rate": rate,
                     "reference_text": request["reference_text"] or None}
        with self.lock, torch.no_grad():
            apply_seed(request["seed"])
            audio = self.model.generate_speech(
                request["text"], self.tokenizer,
                max_new_tokens=frame_limit(request["text"]),
                temperature=1.0, top_p=0.95, top_k=50, **clone)
        samples = audio.detach().cpu().float().numpy()
        return wav_result(self.model_name, "higgs", samples, SAMPLE_RATE)
