"""Official Kokoro demo adapted to the model already loaded by AI-Lab.

The UI structure and features follow hexgrad/Kokoro's upstream demo. The
Hugging Face Spaces hardware selector is intentionally absent: AI-Lab owns
the device and the resident KModel, so selecting another device would load a
second checkpoint.
"""
from __future__ import annotations

import random
from pathlib import Path

import gradio as gr


CHOICES = {
    "🇺🇸 🚺 Heart ❤️": "af_heart", "🇺🇸 🚺 Bella 🔥": "af_bella",
    "🇺🇸 🚺 Nicole 🎧": "af_nicole", "🇺🇸 🚺 Aoede": "af_aoede",
    "🇺🇸 🚺 Kore": "af_kore", "🇺🇸 🚺 Sarah": "af_sarah",
    "🇺🇸 🚺 Nova": "af_nova", "🇺🇸 🚺 Sky": "af_sky",
    "🇺🇸 🚺 Alloy": "af_alloy", "🇺🇸 🚺 Jessica": "af_jessica",
    "🇺🇸 🚺 River": "af_river", "🇺🇸 🚹 Michael": "am_michael",
    "🇺🇸 🚹 Fenrir": "am_fenrir", "🇺🇸 🚹 Puck": "am_puck",
    "🇺🇸 🚹 Echo": "am_echo", "🇺🇸 🚹 Eric": "am_eric",
    "🇺🇸 🚹 Liam": "am_liam", "🇺🇸 🚹 Onyx": "am_onyx",
    "🇺🇸 🚹 Santa": "am_santa", "🇺🇸 🚹 Adam": "am_adam",
    "🇬🇧 🚺 Emma": "bf_emma", "🇬🇧 🚺 Isabella": "bf_isabella",
    "🇬🇧 🚺 Alice": "bf_alice", "🇬🇧 🚺 Lily": "bf_lily",
    "🇬🇧 🚹 George": "bm_george", "🇬🇧 🚹 Fable": "bm_fable",
    "🇬🇧 🚹 Lewis": "bm_lewis", "🇬🇧 🚹 Daniel": "bm_daniel",
}

QUOTES = [
    "A brand for a company is like a reputation for a person. You earn reputation by trying to do hard things well.",
    "The only way to do great work is to love what you do.",
    "The future depends on what you do today.",
]
GATSBY = "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since."
FRANKENSTEIN = "You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings."

TOKEN_NOTE = """
💡 Customize pronunciation with Markdown link syntax and /slashes/ like
`[Kokoro](/kˈOkəɹO/)`.

💬 Adjust intonation with punctuation `;:,.!?—…“”` or stress `ˈ` and `ˌ`.
Lower stress with `[one level](-1)`; raise it with `[or](+2)`.
"""


def create_demo(backend, model_path: Path):
    """Build upstream's full editor around one resident Kokoro KModel."""
    from kokoro import KPipeline

    voices_dir = Path(model_path) / "voices"
    choices = {label: voice for label, voice in CHOICES.items()
               if (voices_dir / f"{voice}.pt").is_file()}
    pipelines = {
        code: KPipeline(lang_code=code, repo_id=backend.pipeline.repo_id,
                        model=backend.model, device=backend.device)
        for code in "ab"
    }
    pipelines["a"].g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
    pipelines["b"].g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"

    def voice_file(voice: str) -> str:
        path = voices_dir / f"{voice}.pt"
        if not path.is_file():
            raise gr.Error(f"Voice pack is not installed: {voice}")
        return str(path)

    def generate_first(text, voice="af_heart", speed=1):
        with backend.lock, backend.torch.inference_mode():
            for item in pipelines[voice[0]](text, voice_file(voice), speed):
                return (24000, item.audio.detach().float().cpu().numpy()), item.phonemes
        return None, ""

    def tokenize_first(text, voice="af_heart"):
        with backend.lock, backend.torch.inference_mode():
            for item in pipelines[voice[0]](text, voice_file(voice), model=False):
                return item.phonemes
        return ""

    def generate_all(text, voice="af_heart", speed=1):
        with backend.lock, backend.torch.inference_mode():
            for item in pipelines[voice[0]](text, voice_file(voice), speed):
                yield 24000, item.audio.detach().float().cpu().numpy()

    with gr.Blocks() as generate_tab:
        out_audio = gr.Audio(label="Output Audio", interactive=False,
                             streaming=False, autoplay=True)
        generate_btn = gr.Button("Generate", variant="primary")
        with gr.Accordion("Output Tokens", open=True):
            out_ps = gr.Textbox(interactive=False, show_label=False,
                                info="Phonemes used to generate the audio; context is up to 510 tokens.")
            tokenize_btn = gr.Button("Tokenize", variant="secondary")
            gr.Markdown(TOKEN_NOTE)

    with gr.Blocks() as stream_tab:
        out_stream = gr.Audio(label="Output Audio Stream", interactive=False,
                              streaming=True, autoplay=True)
        with gr.Row():
            stream_btn = gr.Button("Stream", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")

    with gr.Blocks() as app:
        gr.Markdown("# Kokoro")
        gr.Markdown("Official Kokoro synthesis controls, served by the model already loaded in AI-Lab.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input Text",
                                  info="Arbitrarily long text is split into synthesis chunks.")
                voice = gr.Dropdown(list(choices.items()),
                                    value=backend.default_voice,
                                    label="Voice",
                                    info="Native American and British English voices")
                speed = gr.Slider(minimum=0.5, maximum=2, value=1,
                                  step=0.1, label="Speed")
                random_btn = gr.Button("🎲 Random Quote 💬", variant="secondary")
                with gr.Row():
                    gatsby_btn = gr.Button("🥂 Gatsby 📕", variant="secondary")
                    frankenstein_btn = gr.Button("💀 Frankenstein 📗", variant="secondary")
            with gr.Column():
                gr.TabbedInterface([generate_tab, stream_tab], ["Generate", "Stream"])
        random_btn.click(fn=lambda: random.choice(QUOTES), outputs=[text])
        gatsby_btn.click(fn=lambda: GATSBY, outputs=[text])
        frankenstein_btn.click(fn=lambda: FRANKENSTEIN, outputs=[text])
        generate_btn.click(fn=generate_first, inputs=[text, voice, speed],
                           outputs=[out_audio, out_ps])
        tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
        event = stream_btn.click(fn=generate_all, inputs=[text, voice, speed],
                                 outputs=[out_stream])
        stop_btn.click(fn=None, cancels=event)
    return app
