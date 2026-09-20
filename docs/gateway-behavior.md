# Gateway behavior

One address for an agent workflow that uses several models.

An agent asks the researcher model to read, the developer model to write, the
reviewer model to check. Each is a separate entry here on its own port, and only
one of them can be on the card. An agent that names a model which is not running
gets a refused connection, and the workflow stops there.

This removes that. A request names a model; if it is already loaded the request
goes through, and if it is not, room is made and that model is loaded first.
The agent waits longer for that one request and sees nothing else.

## As many models as fit. Many requests to each.

How many is not a decision here, it is the machine: whatever the memory budget
allows, which is what is free less what is held back for the machine itself.
Measured on the container, a 3.5 GB model and a 17 GB one sit together and both
answer without reloading; a second vLLM model at its default share of the card
does not, and never will.

Requests to a model that is loaded run together — up to the number the engine was started to serve. vLLM interleaves
them in the same pass, which is where its throughput comes from: measured on
this machine at up to seventeen times as concurrency rises, against about 1.4
for llama.cpp. Making them take turns threw that away.

The number is the engine's own, per entry: slots for llama.cpp, sequences for
vLLM. Ask the engine, do not guess from a setting name.

Requests wanting a different model wait, and the queueing rules are in
`scheduler.py`. In one line: **the queue is served in order, and requests next
to each other wanting the same model go in together.**

Nothing younger is served first, ever — not even when it wants the model that
is already loaded and would cost nothing. It arrived after the request that is
waiting, and a workflow can be held up by exactly that one.

The guard that makes it work: the moment anybody waits, the door closes for the
model on the card. Without it a busy model never goes idle, the switch never
happens, and the other request waits for ever.

Which leads to the one thing to design workflows around: **a request must not
wait, inside itself, on another request to this same gateway.** Fill every
place with things that cannot finish and nothing finishes.

A model plus the settings it was started with is one "shape". Two requests for
the same model wanting different context sizes are not requests for the same
thing: one of them needs a reload.

## The buttons on the page are the other way in

Requests here are safe from each other: they queue. The Load and Unload buttons
are not part of that. They reach the engines directly and know nothing about
who is mid-answer, so pressing Unload during a long answer would kill it in the
middle of a sentence.

`guard` is what the routes behind those buttons call first. It refuses while
anything is running or waiting, and says what it found, so the page can offer
to go ahead anyway. Going ahead means a clean slate: everything in flight dies,
everyone waiting is turned away, the card is empty. Half-forced is worse than
either.

## Emptying the card properly

Switching does not unload the outgoing model and start the next one straight
away. The driver returns VRAM a moment after a process exits, and starting a
model on top of memory that has not come back yet fails in a way that reads like
the new model being too large.

So a switch unloads everything running, waits for the card to actually go quiet,
and only then loads. If it does not go quiet, the switch fails and says so,
rather than loading into a mess.

## How long to wait for an engine

Two limits, and they are limits of safety rather than of patience: in normal
work nothing comes near them.

**To the first byte** covers connecting and reading the prompt — and, for a
request that did not ask for streaming, the whole answer, because such an
engine sends nothing until it has finished. Measured here: 8,400 tokens of
prompt read in 0.78 s on the card. On a model split between card and system
memory, or on Apple silicon, it is far slower — that is what this number is
sized for.

**Between bytes** catches an engine that starts answering and stops. At the
slowest generation measured on either machine, 17 tokens a second, the gap
between them is 59 milliseconds. Anything of the order of seconds means
something is wrong, not slow.

One number cannot do both jobs: their sane values are four orders of magnitude
apart. The single hour-long timeout this replaced was absurd for one and
useless for the other.

There is no HTTP in the gateway module. It decides which entry serves a name and makes
sure it is the one on the card; forwarding the request is the web layer's job.
