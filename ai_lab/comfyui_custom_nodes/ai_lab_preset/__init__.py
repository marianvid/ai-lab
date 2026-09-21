"""Load the AI-Lab instance's configured graph into native ComfyUI."""

import json
import os
from pathlib import Path

from aiohttp import web
from server import PromptServer


WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}


@PromptServer.instance.routes.get("/ai-lab/preset")
async def preset(_request):
    path = (os.environ.get("AI_LAB_COMFYUI_UI_PRESET")
            or os.environ.get("AI_LAB_COMFYUI_PRESET"))
    if not path:
        return web.json_response({"error": "No preset configured"}, status=404)
    graph = json.loads(Path(path).read_text())
    if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
        return web.json_response(graph)
    # Media adapters store the prompt graph alongside metadata.
    if isinstance(graph, dict) and isinstance(graph.get("prompt"), dict):
        graph = graph["prompt"]
    if not isinstance(graph, dict) or not graph:
        return web.json_response({"error": "Invalid preset"}, status=500)
    defaults = {
        "__AI_LAB_PROMPT__": "",
        "__AI_LAB_CAPTION__": "",
        "__AI_LAB_LYRICS__": "[Instrumental]",
        "__AI_LAB_INPUT__": "",
        "__AI_LAB_SEED__": 0,
        "__AI_LAB_DURATION__": 30,
        "__AI_LAB_OUTPUT__": "ai_lab_manual",
    }

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return defaults.get(value, value) if isinstance(value, str) else value

    return web.json_response(replace(graph))
