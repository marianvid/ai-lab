"""AI-Lab's own refusals, each with a machine-readable code.

The server turns each into its HTTP status and an answer that carries the code
twice: as `error.code` for OpenAI-shaped clients, and as `ai_lab.reason` for
everyone (the approved list, A7).
"""

from __future__ import annotations


class RefusalError(Exception):
    """A refusal with a code and the extra facts a client may act on."""

    code = "refused"
    kind = "ai_lab"

    def __init__(self, message: str, **facts) -> None:
        super().__init__(message)
        self.detail = {"ai_lab": {"reason": self.code, **facts}}


class TokenMissingError(RefusalError):
    """The call carried no token, and this AI-Lab requires one."""

    code = "token_missing"


class TokenInvalidError(RefusalError):
    """The call carried a token that belongs to no registered client."""

    code = "token_invalid"


class AiLabOffError(RefusalError):
    """AI-Lab was switched Off: it serves nothing until it is switched On."""

    code = "ai_lab_off"


class ModelReplacedError(RefusalError):
    """The call's model was taken off the card to make room for another call."""

    code = "model_replaced"
