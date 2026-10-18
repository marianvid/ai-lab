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
# The ranges a request may set the sampling knobs in.
LOWEST_TEMPERATURE, HIGHEST_TEMPERATURE = 0.05, 2.0
HIGHEST_TOP_K = 1000
KNOBS = ("temperature", "top_p", "top_k")
# How many encoded reference voices are remembered.
MAX_REFERENCES = 32
# The fewest frames an answer may be cut to, whatever its text.
MIN_FRAMES = 64
# The end-of-audio tail: after the end code, every codebook but the first two
# still has frames to give before the line is complete.
EOC_TAIL = 2


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
    return {"temperature": TEMPERATURE if temperature is None else _temperature(temperature),
            "top_p": TOP_P if top_p is None else _top_p(top_p),
            "top_k": TOP_K if top_k is None else _top_k(top_k),
            "max_frames": MAX_FRAMES if max_new_tokens is None else _frames(max_new_tokens)}


def _temperature(value) -> float:
    if not LOWEST_TEMPERATURE <= float(value) <= HIGHEST_TEMPERATURE:
        raise ValueError("temperature must be from 0.05 to 2")
    return float(value)


def _top_p(value) -> float:
    if not 0.0 < float(value) <= 1.0:
        raise ValueError("top_p must be above 0 and at most 1")
    return float(value)


def _top_k(value) -> int:
    if type(value) is not int or not 1 <= value <= HIGHEST_TOP_K:
        raise ValueError("top_k must be an integer from 1 to 1000")
    return value


def _frames(value) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    return min(value, MAX_FRAMES)


def frame_limit(text: str) -> int:
    """Longest plausible answer for this text, in audio frames.

    Higgs occasionally fails to stop and speaks on to its token limit —
    eighty seconds for one sentence was measured. A limit derived from the
    text keeps a runaway to a few seconds instead of a minute: about 0.13 s
    per character plus three seconds of slack, never above the model's own
    ceiling.
    """
    seconds = 0.13 * max(len(text), 1) + 3.0
    return max(MIN_FRAMES, min(MAX_FRAMES, int(seconds * FRAMES_PER_SECOND)))


class HiggsLocalBackend:
    """Higgs TTS 3 in this process, lines arriving together generated as one group."""

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
            while len(self._references) > MAX_REFERENCES:
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
            for i, wave in zip(good, waves, strict=True):
                results[i] = wave
            return results

    def _generate(self, embeds: list, requests: list[dict]) -> list:
        """Every line of the group, frame by frame, then decoded to samples."""
        group = _Group(self, embeds, requests)
        while group.active and group.step():
            pass
        return [self._decode(row) for row in group.rows]

    def _decode(self, row: list):
        """One line's frames as audio samples; empty when it never got going."""
        import torch

        if len(row) < self.model.num_codebooks:
            return torch.zeros(0).numpy()
        codes = self.port.reverse_delay_pattern(torch.stack(row).cpu())
        return self.model._decode_codes(codes).float().cpu().numpy()

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

    def _advance(self, codes, state, codebooks: int):
        """The port's delay/end-of-audio state machine, on codes drawn here."""
        port = self.port
        if state.delay_count < codebooks:
            if state.delay_count + 1 < codebooks:
                codes[state.delay_count + 1:] = port.BOC_ID
            state.delay_count += 1
        elif state.eoc_countdown is not None:
            state.eoc_countdown -= 1
            if state.eoc_countdown <= 0:
                state.generation_done = True
        elif int(codes[0]) == port.EOC_ID:
            if codebooks <= EOC_TAIL:
                state.generation_done = True
            else:
                state.eoc_countdown = codebooks - EOC_TAIL
        return codes


class _Group:
    """One group of lines generated together: its cache, its rows and sampler states.

    Left padding: every row ends in the same column, so the next position of
    every row is the same cache slot. Each line keeps its own sampling knobs
    and its own seeded random generator. A finished line leaves the group and
    the cache.
    """

    def __init__(self, backend: HiggsLocalBackend, embeds: list, requests: list[dict]) -> None:
        import torch

        self.torch, self.backend = torch, backend
        self.model, self.device = backend.model, backend.device
        self.codebooks = self.model.num_codebooks
        self.generators = [torch.Generator() for _ in requests]
        for generator, request in zip(self.generators, requests, strict=True):
            generator.manual_seed(request["seed"])
        self.limits = [min(frame_limit(r["text"]), r["sampling"]["max_frames"])
                       for r in requests]
        self.knobs = {name: [r["sampling"][name] for r in requests] for name in KNOBS}
        self.states = [backend.port._SamplerState(num_codebooks=self.codebooks)
                       for _ in requests]
        self.rows: list[list] = [[] for _ in requests]
        self.active = list(range(len(requests)))
        self._prefill(embeds)

    def _prefill(self, embeds: list) -> None:
        """Every prompt through the model at once, left-padded to one width."""
        torch, device = self.torch, self.device
        width = max(e.shape[0] for e in embeds)
        self.dtype = embeds[0].dtype
        x = torch.zeros(len(embeds), width, embeds[0].shape[1], dtype=self.dtype, device=device)
        mask = torch.zeros(len(embeds), width, dtype=torch.long, device=device)
        for row, embed in enumerate(embeds):
            x[row, width - embed.shape[0]:] = embed
            mask[row, width - embed.shape[0]:] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model.model(inputs_embeds=x, attention_mask=mask, position_ids=positions,
                               use_cache=True)
        self.past, self.hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        self.mask, self.last, self.length = mask, positions[:, -1], width

    def step(self) -> bool:
        """One frame for every line still going; False when none goes on."""
        codes, keep = self._draw()
        if not keep:
            return False
        if len(keep) < len(self.active):
            self._drop_finished(keep)
        self._feed(codes)
        return True

    def _draw(self) -> tuple[list, list[int]]:
        """Sample each active line's next codes; which lines (by position) go on."""
        torch = self.torch
        probs = HiggsLocalBackend._probabilities(
            self.model.audio_head(self.hidden).to(torch.float32),
            *(torch.tensor([self.knobs[name][b] for b in self.active], device=self.device)
              for name in KNOBS)).cpu()
        codes, keep = [], []
        for j, b in enumerate(self.active):
            drawn = torch.multinomial(probs[j], 1, generator=self.generators[b]).squeeze(-1)
            code = self.backend._advance(drawn.to(torch.long), self.states[b], self.codebooks)
            if self.states[b].generation_done or len(self.rows[b]) >= self.limits[b]:
                continue
            self.rows[b].append(code.clone())
            codes.append(code)
            keep.append(j)
        return codes, keep

    def _drop_finished(self, keep: list[int]) -> None:
        """Finished lines leave the group and the cache."""
        index = self.torch.tensor(keep, device=self.device)
        self.past.batch_select_indices(index)
        self.mask, self.last = self.mask[index], self.last[index]
        self.active = [self.active[j] for j in keep]

    def _feed(self, codes: list) -> None:
        """The frames just drawn, through the model, one cache slot further."""
        torch, device = self.torch, self.device
        step = self.model.audio_embedding(torch.stack(codes).to(device)).unsqueeze(1).to(self.dtype)
        self.mask = torch.cat([self.mask, torch.ones(len(self.active), 1, dtype=self.mask.dtype,
                                                     device=device)], dim=1)
        self.last = self.last + 1
        out = self.model.model(inputs_embeds=step, attention_mask=self.mask,
                               position_ids=self.last[:, None], past_key_values=self.past,
                               use_cache=True,
                               cache_position=torch.tensor([self.length], device=device))
        self.past, self.hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        self.length += 1
