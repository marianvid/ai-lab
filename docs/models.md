# Models — what is configured, and what is running

One row per configured model. This is where models are started, stopped and
set up. What is happening *right now* across all of them is on the
[Gateway page](gateway.md).

One line per configured model: the name you gave it, the model it runs, and —
at the right, against the buttons — what the model can do, the weight format
and the engine that will serve it. How long the last load took sits on the row
too, because a load runs from four seconds to a minute and by the time you look
back a message elsewhere would be gone. Port, context, temperature, state and
the breakdown of that load by phase live in the tooltip, because they are
wanted occasionally and were costing three lines of screen every time.

When the model has a note written for it (see
[Curation](library.md#curation)), its short line sits under the name.

![The model list](screenshots/models.png)

**The task says what kind of request an entry accepts.** It is in the row's
tooltip, and beside every model in the list you pick from. The tasks are text
generation, transcription, alignment, voice activity detection, diarization,
speech synthesis (text to speech), music generation, OCR (reading the text out
of an image), image generation, image editing and video generation. The two
small pictures on text models say whether they can call tools or read pictures.
Those capabilities are not configured anywhere —
both are read from the model's own files, once, and remembered. A directory of
weights must carry both a vision section *and* a token to put a picture in
before it claims pictures, because a text model's config can name a vision
tower it never uses. For GGUF it is the chat template inside the weights file,
plus the `mmproj-` file beside it that llama.cpp is handed to see with — so the
same model downloaded without that file honestly shows no picture icon.

The weights decide what a model *can* do; a setting can only take something
away. vLLM's "Text only" loads a model that can see without the part that sees,
so the picture icon goes when it is set — and the wrench stays, because that
setting has nothing to do with the chat template.

**Load** and **Unload** act directly, not through the queue, so they ask first
when a model is mid-answer and offer to stop it anyway. A wedged model has to be
stoppable — but by decision rather than by accident. Load never unloads
anything else: it is a manual act, for looking at one model, and if there is no
room it says what is in the way. On llama.cpp it loads anyway and lets the
engine leave the layers that will not fit in system memory.

**The weight format decides the engine.** GGUF goes to llama.cpp, the NVIDIA
formats to vLLM, and on a Mac the `mlx` format — a folder written by Apple's
mlx-lm converter — goes to MLX LM. An MLX folder is one model, named after the
folder, exactly like any other directory of safetensors; its picture and tool
icons come from its `config.json` and chat template in the same way. MLX LM
itself reads text only, so its rows never show the picture icon, even when
the model's files say it can see. The engine says so itself (it "withholds"
pictures), the same way vLLM's "Text only" setting does.

**Settings** on a row shows what the model will be started with — context,
cache precision, how many requests at once, precision for NeMo, and per engine
the rest. Settings that do not apply to the selected task are not shown. Changing
them means restarting the model, which is why the button says **Apply & reload**.
**Save** writes them down without touching the card.

**Log** shows what the engine is printing about itself. It works only while
the model is running: a stopped model has a log on Linux and none on macOS, and
why a model would not start is already in the message a failed load shows.

**Remove** deletes the entry, not the model. The downloaded files stay in
[Library](library.md). It is switched off while the model is loaded, and it
asks first.

## Adding a model

**+ Add model** opens a small form. It lists only the models on disk that an
installed engine on this machine can run.

- **Model** — which weights. Each is shown with its task, format and size.
- **Name** — lower-case letters, digits and hyphens. A request sends this as
  `"model"`, so it cannot be changed later. Renaming means removing the entry
  and adding it again.
- **Port** — the first free one is filled in. Change it only if a client
  expects another.

Below those come the engine's own settings for that model's task.

## Using a model directly

Many engines come with their own web page: llama.cpp has a chat page, ComfyUI
has its editor, and the speech and music engines have upstream editors. When
such a model is loaded and ready, its row carries a button that opens that
page in a new tab — **Chat**, **Speak**, **Music**, **Create**, **Edit** or
**Video**. While the model is stopped or still loading, the button is there
but greyed out. Which engine opens what is in
[Using a model directly](direct-use.md).

---

[← all documents](../README.md)  ·  [Library](library.md)  ·  [Using a model directly](direct-use.md)  ·  [Audio](audio.md)
