# Music generation

AI-Lab currently has one music runtime: ACE-Step 1.5 XL Turbo. The selected
checkpoint must have an `acestep` engine entry and a complete shared checkpoint
tree. The public repository supplies the adapter; machine paths and model
selection belong to private `opts`.

From **Models**, choose the configured music entry and select **Music**. Enter a
style prompt, optional lyrics, duration from 5 to 180 seconds, and an optional
seed. AI-Lab loads the model when needed, keeps its gateway lease for the whole
generation, then shows a player and WAV download.

Agents use the same gateway:

```http
POST /v1/audio/music/generations
Content-Type: application/json

{"model":"music-ace-xl","prompt":"Ambient piano, no vocals",
 "duration":15,"instrumental":true}
```

The response contains `data[0].b64_wav` and `mime_type: audio/wav`, plus the
actual seed. It is synchronous: the connection remains open until generation
ends. The generated file is returned in the response and removed from the
runtime's temporary output directory; save the result if it is needed later.
No historical music jobs or cancellation API exists yet.

The engine accepts only an explicitly configured checkpoint name. The manager
cannot load an arbitrary safetensors directory as ACE-Step. This prevents a
Library label from presenting an unusable Generate button.
