# Gateway behavior

One address for an agent workflow that uses several models.

An agent asks the researcher model to read, the developer model to write, the
reviewer model to check. Each is a separate entry here on its own port, and they
rarely all fit on the card at once. An agent that names a model which is not
running gets a refused connection, and the workflow stops there.

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
to go ahead anyway. Going ahead means a clean slate: the engine is stopped
under whoever was using it, everyone waiting is turned away, and the gateway
forgets what it believed was loaded and reads the machine again before the
next request. Half-forced is worse than either.

Any Load or Unload from the page, forced or not, also tells the gateway that
the card has changed, so it does not admit a request onto a model that is no
longer there.

## Emptying the card properly

Switching does not unload the outgoing model and start the next one straight
away. The driver returns VRAM (the card's own memory) a moment after a process
exits, and starting a model on top of memory that has not come back yet fails
in a way that reads like the new model being too large.

So a switch unloads only the models chosen to make room (the rules are in
[the Gateway](gateway.md#how-loading-and-unloading-is-decided)), waits up to
60 seconds for those processes to exit, and only then loads. If nothing else
is meant to stay loaded, it also waits for the card to read almost empty
(512 MB or less — a driver keeps a little for itself). With other models still
loaded their memory legitimately stays, so instead the card is read again and
the load is refused if the room is still not there. Either way the switch fails
and says so, rather than loading into a mess. On Apple silicon there is no
separate card memory, so there is nothing to wait for.

## How long to wait for an engine

Two limits, and they are limits of safety rather than of patience: in normal
work nothing comes near them.

**To the first byte** covers connecting and reading the prompt — and, for a
request that did not ask for streaming, the whole answer, because such an
engine sends nothing until it has finished. It is counted from when the
request is sent to the first byte of the *answer*. Two things arrive earlier
and do not count: the reply's headers, which a streaming engine sends at
once, and "still here" lines (event-stream comments, a line starting with
`:`), which llama.cpp and mlx-lm send while they read a long prompt. Those
lines are passed on to the client unchanged. Counting either as the first
byte started the idle limit too early: a 34,000-token prompt on the Mac, a
40-second read, came back empty after 30 s. Measured here: 8,400 tokens of
prompt read in 0.78 s on the card. On a model split between card and system
memory, or on Apple silicon, it is far slower — that is what this number is
sized for.

Each piece of a streamed answer is passed on the moment it arrives, not
collected into larger blocks first, so the first word reaches the client when
the engine sends it.

**Between bytes** catches an engine that starts answering and stops. At the
slowest generation measured on either machine, 17 tokens a second, the gap
between them is 59 milliseconds. Anything of the order of seconds means
something is wrong, not slow.

One number cannot do both jobs: their sane values are four orders of magnitude
apart. The single hour-long timeout this replaced was absurd for one and
useless for the other.

Where they are set:

| setting | on the Gateway page | default |
|---|---|---|
| `first_byte_s` | Max TTFT (time to first token) | 120 s |
| `between_bytes_s` | Max idle time | 30 s |
| `max_waiting` | Max queue size | 150 requests |

The page accepts 1 to 3,600 seconds and saves at once. The same names can go in
the `gateway` section of the configuration file, where the waits may be up to a
day.

A slow kind of request should not share a text model's limits, so the
configuration can also override the two waits per kind of request, under
`task_timeouts`, keyed by the task name — `ocr`, `speech-synthesis`,
`music-generation`, `video-generation`, `transcription` and so on:

```json
"gateway": {
  "task_timeouts": { "music-generation": { "first_byte_s": 1800 } }
}
```

A task not named there uses the two defaults. Background media jobs use their
task's first-byte limit too — see [Media jobs](media-jobs.md).

There is no HTTP in the gateway module. It decides which entry serves a name and makes
sure it is the one on the card; forwarding the request is the web layer's job.
