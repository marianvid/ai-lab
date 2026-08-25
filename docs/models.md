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

![The model list](screenshots/models.png)

**The task label says what kind of request an entry accepts**: text generation,
transcription, alignment, voice activity detection or diarization. The two
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

**Settings** on a row shows what the model will be started with — context,
cache precision, how many requests at once, precision for NeMo, and per engine
the rest. Settings that do not apply to the selected task are not shown. Changing
them means restarting the model, which is why the button says **Apply & reload**.
**Save** writes them down without touching the card.

---

[← all documents](../README.md)  ·  [Library](library.md)  ·  [Audio](audio.md)
