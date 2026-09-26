"""Higgs TTS 3 in-process, through its transformers port, with batching.

Where SGLang-Omni cannot run — Apple silicon has no backend for it — the same
weights load through the transformers port published beside them, which
carries its own architecture code. The port generates one line per call;
here lines that arrive together are generated as one group instead (see
`batching.py`): every step advances all of them in one pass, and a finished
line leaves the group and the cache. Each line samples with its own random
generator, so its seed is its own whoever else is in the group.

The checkpoint directory holds the weights, the port's code and, under
`codec/`, the audio codec the port would otherwise fetch from the Hub at
first use. AI-Lab runs offline, so the codec is pointed at locally.
"""
from __future__ import annotations

import hashlib
import io
import secrets
import sys
from collections import OrderedDict
from pathlib import Path

from .batching import BatchQueue
from .contract import validate_payload, wav_result

SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 27  # measured on the M3 Max: 155 frames made 5.9 s
TEMPERATURE, TOP_P, TOP_K = 1.0, 0.95, 50
MAX_FRAMES = 2048  # the model's own ceiling on audio frames for one answer


def sampling_settings(temperature: float | None = None,
                      top_p: float | None = None, top_k: int | None = None,
                      max_new_tokens: int | None = None) -> dict:
    """Check the "how adventurous" knobs one request may set, with defaults.

    Only the browser playground sends these; the public speech API always
    uses the defaults. In plain words:
    - `temperature` flattens (above 1) or sharpens (below 1) the odds of each
      next sound, so higher sounds livelier and less predictable;
    - `top_k` keeps only that many most likely next sounds;
    - `top_p` keeps the smallest set of likely sounds whose odds add up to it;
    - `max_new_tokens` caps the length, in audio frames (27 per second).
    A value outside its range is refused rather than quietly clamped.
    """
    result = {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
              "max_frames": MAX_FRAMES}
    if temperature is not None:
        if not 0.05 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be from 0.05 to 2")
        result["temperature"] = float(temperature)
    if top_p is not None:
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be above 0 and at most 1")
        result["top_p"] = float(top_p)
    if top_k is not None:
        if type(top_k) is not int or not 1 <= top_k <= 1000:
            raise ValueError("top_k must be an integer from 1 to 1000")
        result["top_k"] = top_k
    if max_new_tokens is not None:
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        result["max_frames"] = min(max_new_tokens, MAX_FRAMES)
    return result


def frame_limit(text: str) -> int:
    """Longest plausible answer for this text, in audio frames.

    Higgs occasionally fails to stop and speaks on to its token limit —
    eighty seconds for one sentence was measured. A limit derived from the
    text keeps a runaway to a few seconds instead of a minute: about 0.13 s
    per character plus three seconds of slack, never above the model's own
    ceiling.
    """
    seconds = 0.13 * max(len(text), 1) + 3.0
    return max(64, min(MAX_FRAMES, int(seconds * FRAMES_PER_SECOND)))


class HiggsLocalBackend:
    def __init__(self, model_path: Path, max_batch: int = 8,
                 window_ms: int = 100) -> None:
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
        # The port's helpers — delay pattern, sampler states, special ids —
        # live in its own module, loaded as remote code.
        self.port = sys.modules[type(self.model).__module__]
        self.model_name = model_path.name
        self._references: OrderedDict[str, tuple] = OrderedDict()
        self.queue = BatchQueue(self._run, max_batch, window_ms / 1000)

    # -- the contract ---------------------------------------------------------

    def generate(self, body: dict, sampling: dict | None = None) -> dict:
        """One speech request. `sampling` comes from `sampling_settings`;
        only the browser playground passes it, the API uses the defaults."""
        request = validate_payload(body, seed=True, reference=True)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        if request["instruction"]:
            raise ValueError("Higgs accepts style controls inline in the text")
        if request["speaker"]:
            raise ValueError("Higgs has no named voices; clone one with reference audio")
        seed = request["seed"] if request["seed"] is not None else secrets.randbelow(2**31)
        samples = self.queue.submit({
            "text": request["text"], "seed": seed,
            "reference": request["reference_audio"],
            "reference_text": request["reference_text"],
            "sampling": sampling or sampling_settings()})
        result = wav_result(self.model_name, "higgs", samples, SAMPLE_RATE)
        result["seed"] = seed
        return result

    # -- one group, on the batch worker thread --------------------------------

    def _reference(self, audio: bytes | None, text: str):
        """Encoded reference clip, remembered: a voice is reused line after line."""
        if audio is None:
            return None, None
        import soundfile as sf
        import torch

        key = hashlib.sha256(audio).hexdigest()
        if key not in self._references:
            samples, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
            codes = self.model._encode_reference(torch.from_numpy(samples).mean(dim=1), rate)
            self._references[key] = self.port.apply_delay_pattern(codes.cpu())
            while len(self._references) > 32:
                self._references.popitem(last=False)
        self._references.move_to_end(key)
        return self._references[key], (text or None)

    def _run(self, requests: list[dict]) -> list:
        import torch

        with torch.no_grad():
            prepared = []
            for request in requests:
                try:
                    delayed, ref_text = self._reference(request["reference"],
                                                        request["reference_text"])
                    ids = self.model._build_prompt_ids(
                        self.tokenizer, request["text"],
                        num_ref_tokens=0 if delayed is None else delayed.shape[0],
                        reference_text=ref_text)
                    prepared.append(self.model._prefill_embeds(ids, delayed)[0])
                except Exception as error:  # a bad reference fails its own line only
                    prepared.append(error)
            good = [i for i, p in enumerate(prepared) if not isinstance(p, Exception)]
            waves = self._generate([prepared[i] for i in good],
                                   [requests[i] for i in good]) if good else []
            results = list(prepared)
            for i, wave in zip(good, waves):
                results[i] = wave
            return results

    def _generate(self, embeds: list, requests: list[dict]) -> list:
        import torch

        m, port, device = self.model, self.port, self.device
        N = m.num_codebooks
        B = len(embeds)
        generators = []
        for request in requests:
            generator = torch.Generator()
            generator.manual_seed(request["seed"])
            generators.append(generator)
        limits = [min(frame_limit(r["text"]), r["sampling"]["max_frames"])
                  for r in requests]
        # Each line keeps its own sampling knobs, one value per row.
        knobs = {name: [r["sampling"][name] for r in requests]
                 for name in ("temperature", "top_p", "top_k")}

        # Left padding: every row ends in the same column, so the next
        # position of every row is the same cache slot.
        S = max(e.shape[0] for e in embeds)
        x = torch.zeros(B, S, embeds[0].shape[1], dtype=embeds[0].dtype, device=device)
        mask = torch.zeros(B, S, dtype=torch.long, device=device)
        for b, e in enumerate(embeds):
            x[b, S - e.shape[0]:] = e
            mask[b, S - e.shape[0]:] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        out = m.model(inputs_embeds=x, attention_mask=mask, position_ids=positions,
                      use_cache=True)
        past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        last = positions[:, -1]
        states = [port._SamplerState(num_codebooks=N) for _ in range(B)]
        rows: list[list] = [[] for _ in range(B)]
        active, length = list(range(B)), S

        while active:
            probs = self._probabilities(
                m.audio_head(hidden).to(torch.float32),
                *(torch.tensor([knobs[name][b] for b in active], device=device)
                  for name in ("temperature", "top_p", "top_k"))).cpu()
            codes, keep = [], []
            for j, b in enumerate(active):
                drawn = torch.multinomial(probs[j], 1, generator=generators[b]).squeeze(-1)
                code = self._advance(drawn.to(torch.long), states[b], N)
                if states[b].generation_done or len(rows[b]) >= limits[b]:
                    continue
                rows[b].append(code.clone())
                codes.append(code)
                keep.append(j)
            if not keep:
                break
            if len(keep) < len(active):  # finished lines leave the group and the cache
                index = torch.tensor(keep, device=device)
                past.batch_select_indices(index)
                mask, last = mask[index], last[index]
                active = [active[j] for j in keep]
            step = m.audio_embedding(torch.stack(codes).to(device)).unsqueeze(1).to(x.dtype)
            mask = torch.cat([mask, torch.ones(len(active), 1, dtype=mask.dtype,
                                               device=device)], dim=1)
            last = last + 1
            out = m.model(inputs_embeds=step, attention_mask=mask,
                          position_ids=last[:, None], past_key_values=past, use_cache=True,
                          cache_position=torch.tensor([length], device=device))
            past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
            length += 1

        waves = []
        for b in range(B):
            if len(rows[b]) < N:
                waves.append(torch.zeros(0).numpy())
                continue
            codes_TN = port.reverse_delay_pattern(torch.stack(rows[b]).cpu())
            waves.append(m._decode_codes(codes_TN).float().cpu().numpy())
        return waves

    @staticmethod
    def _probabilities(logits, temperature, top_p, top_k):
        """Temperature, top-k and top-p for the whole group at once, on the GPU.

        `logits` holds one row per line in the group (then one per codebook,
        then one score per possible sound). The three knobs hold one value
        per line, so lines in the same group may sample differently.
        """
        import torch

        rows = logits.shape[0]
        shape = (rows,) + (1,) * (logits.dim() - 1)
        logits = logits / temperature.to(logits.dtype).view(shape)
        k = top_k.clamp(max=logits.shape[-1]).to(torch.long)
        best = logits.topk(int(k.max()), dim=-1).values
        index = (k - 1).view(shape).expand(*logits.shape[:-1], 1)
        kth = best.gather(-1, index)
        logits = torch.where(logits < kth, float("-inf"), logits)
        ordered, order = torch.sort(logits, descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p.to(logits.dtype).view(shape)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scattered = torch.zeros_like(remove)
        scattered.scatter_(-1, order, remove)
        return torch.where(scattered, float("-inf"), logits).softmax(-1)

    def _advance(self, codes, state, N: int):
        """The port's delay/end-of-audio state machine, on codes drawn here."""
        port = self.port
        if state.delay_count < N:
            if state.delay_count + 1 < N:
                codes[state.delay_count + 1:] = port.BOC_ID
            state.delay_count += 1
        elif state.eoc_countdown is not None:
            state.eoc_countdown -= 1
            if state.eoc_countdown <= 0:
                state.generation_done = True
        elif int(codes[0]) == port.EOC_ID:
            if N <= 2:
                state.generation_done = True
            else:
                state.eoc_countdown = N - 2
        return codes
