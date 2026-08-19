# Interface tests

The browser interface has no dependencies and no build step. jsdom is here for
the tests alone and is never deployed — the container does not even have node.

```bash
npm install      # once
npm test
```

They also run as the first step of `scripts/deploy.sh`, before anything is
uploaded, and are skipped with a notice if `node_modules` is absent.

## What these are for

Every test names something a person sees, what they would expect from it, and
checks that is what happens. They exist because a run of interface faults
reached the user unnoticed: the word `null` printed on the page, a search field
wiped mid-word by the automatic refresh, a bar reading zero next to the word
"ready", a button that appeared to do nothing because its effect landed below
the fold.

They will not catch everything. A button labelled "Open" when it should say
"Show models", a checkbox whose caption rewrote itself, a dropdown asking a
question whose answer was already known — no test would have objected to any of
those. Those are design faults, and the remedy is to walk through each screen
and ask what the reader sees and expects. The tests cover the other half: that
what was intended is what actually appears.

## jsdom rather than a stand-in

A hand-written DOM would have been written with the same assumptions that
produced the bugs. jsdom reproduces the behaviour that actually bit —
`replaceChildren` really does turn a null into the text "null" there.

Where jsdom falls short it is patched in `support/dom.js`, and each patch says
what it is standing in for. `<dialog>` is the current one: jsdom has no modal
support, so `showModal` and `close` are supplied, with `close` queueing its
event the way the standard requires. That detail is not pedantry — dispatching
it inline hid a real fault, where the confirmation resolved to "no" before the
button that meant "yes" had finished.
