"""Gateway refusals translated into HTTP responses by the API layer."""

class NotConfigured(KeyError):
    """No entry serves that model name.

    A KeyError so the web layer answers 404 without being told, since that is
    already the rule for "no such thing".
    """

    def __str__(self) -> str:
        # KeyError renders its argument with repr(), which wraps the whole
        # sentence in quotes and escapes what is inside it. This message lists
        # the names that would have worked, and it is read by a person
        # debugging an agent, so it should arrive as a sentence.
        return self.args[0] if self.args else ""


class CouldNotLoad(RuntimeError):
    """The entry exists but the card could not be made ready for it."""


class ShapeNotServed(ValueError):
    """The entry exists, but its engine does not answer that kind of request.

    A request can arrive in more than one shape, and not every engine speaks
    every shape. Refused here, with the entries that would have worked, rather
    than forwarded to an engine that would answer 404 about a path the client
    never chose.
    """


class CardBusy(RuntimeError):
    """The card is serving a request, and the action asked for would cut it off.

    Raised at the request of the interface, not by the gateway's own work: the
    buttons on the page reach the engines directly, and this is how they find
    out that somebody is mid-answer. It carries `detail` so the page can offer
    to go ahead anyway rather than only printing a sentence.
    """

    def __init__(self, message: str, holder: dict) -> None:
        super().__init__(message)
        self.detail = {"busy": holder}
