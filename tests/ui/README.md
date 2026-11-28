# Interface tests

The browser interface has no dependencies and no build step. jsdom is here for
the tests alone and is never deployed — the container does not even have node.

```bash
npm install      # once
npm test
```

They also run as the first step of `scripts/deploy.sh`, before anything is
uploaded, and are skipped with a notice if `node_modules` is absent.

## How much of the interface they reach

```bash
npm run coverage
```

`tests/ui/coverage.mjs`: the tests run with V8's raw coverage, and every copy
of a module counts as one. The tests import a page as `views/x.js?<random>`,
for a fresh copy of its state per test, and Node's own report
(`--experimental-test-coverage`) lists every copy as a file of its own. That is
why it read 79.5 % at the audit, with `gpu_health.js` at 31 % although every
test there draws the whole page. Counted per file, lines that hold code:

| Measured | All of `ai_lab/web/js` | Least reached |
|---|---|---|
| 2026-11-28, before round 6's UI tests | 90.5 % | `settings_engines.js` 66.0 %, `app.js` 70.9 %, `runtime_progress.js` 71.4 %, `library_installed.js` 75.0 % |
| 2026-11-28, after them | **92.4 %** | `app.js` 70.9 %, `runtime_progress.js` 71.4 %, `api.js` 78.3 %; `settings_engines.js` 95.1 %, `library_installed.js` 87.1 % |

`node tests/ui/coverage.mjs views/<file>.js` also prints that file's
unreached lines. In `runtime_progress.js` they are the bars drawn from load
events, which no test sends yet; `app.js` and `api.js` were not gone through
in round 6. No floor is set: the figure is written here so a change can be
compared with it.

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

## Every request must be answered

`support/dom.js` answers the page's requests from what a test gives it. A path
the test does not answer fails that test, named in the message (the reviewer's
read of the tests, item 14): before, it got 200 and `[]`, so a view calling a
misspelt or removed route passed. Five reads every page makes in passing
(`BACKGROUND`: the read-only sections, the setting fields, builds, installs,
the login state) still get `[]` when unanswered; a test lists any other path
it does not care about in `quiet`. `test_dom_support.js` holds the rule.
