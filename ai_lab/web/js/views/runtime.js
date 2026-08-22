// The model list: what is configured, what is loaded, and managing both.
//
// One row is one model. There is no "swap" — changing which model a row runs
// is an edit followed by a reload, which is the same act as starting it for
// the first time.

import { api } from '../api.js';
import { confirmDestructive, showNotice } from '../confirm.js';
import { onProgress } from '../events.js';
import { settingsForm } from '../form.js';
import { bytes, element, seconds } from '../format.js';
import { setStatus } from '../status.js';
import { onPanelChange, toggleLogs, watching } from '../logpane.js';

const progress = new Map();     // instance id -> latest event
const open = new Set();         // rows with their settings expanded
let adding = false;
let subscribed = false;
let redraw = () => {};

// -- the progress bar -------------------------------------------------------

// The bar is the progress of the operation, 0 to 100%, worked out by the
// server. Deliberately not memory occupancy: a 4 GB model on a 32 GB card
// would sit at 13% while completely loaded.
function paint(bar, event) {
  bar.className = 'bar thin' + (event.phase === 'ready' ? ' done'
    : event.phase === 'failed' ? ' failed' : '');
  bar.firstChild.style.width = `${Math.round((event.progress || 0) * 100)}%`;
  bar.title = `${Math.round((event.progress || 0) * 100)}% · ${event.phase}`
    + ` · ${seconds(event.elapsed_ms)}`
    + ` · this model ${Math.round(event.memory_used_mb)} MB`;
}

function clearBar(bar) {
  bar.className = 'bar thin';
  bar.firstChild.style.width = '0%';
  bar.title = 'nothing loaded';
}

// On a fresh page there are no events yet, but the model may well be loaded
// already. Drawing an empty bar beside the word "ready" puts two contradictory
// things on screen and makes the reader trust neither.
function paintFromState(bar, instance) {
  if (instance.ready) {
    bar.className = 'bar thin done';
    bar.firstChild.style.width = '100%';
    bar.title = 'loaded';
  } else if (instance.running) {
    bar.className = 'bar thin';
    bar.firstChild.style.width = '50%';
    bar.title = 'starting';
  } else {
    clearBar(bar);
  }
}

function subscribe() {
  if (subscribed) return;
  subscribed = true;
  // The bar and nothing else. These events arrive for every load, including
  // ones an agent caused, and announcing them at the foot of the page meant
  // being told about work you did not start — in a message naming the entry
  // and the file on disk, two names from two places, one of which is on
  // another page. What you pressed yourself is reported by the thing you
  // pressed.
  onProgress((event) => {
    progress.set(event.instance_id, event);
    const bar = document.querySelector(`[data-bar="${CSS.escape(event.instance_id)}"]`);
    if (bar) paint(bar, event);
  });
}

// -- actions ----------------------------------------------------------------

// A failure interrupts; a success does not.
//
// A load takes between four seconds and a minute, so you look away. The status
// line is right for "finished in 12.7 s" and wrong for "it would not start":
// the page is long, that line sits below it with no border and no background,
// and the next action wipes it. The explanation is also the useful part — an
// engine that refuses a context says which one would have fitted — and it is
// worth taking the page away for.
function failed(label, detail) {
  setStatus(detail, 'error');
  return showNotice({ title: `${label} failed`, body: detail });
}

function report(label, result) {
  const operation = result.operation || result;
  if (operation.ok === false) return failed(label, operation.error);
  if (operation.total_ms !== undefined) {
    setStatus(`${label} finished in ${seconds(operation.total_ms)}`, 'ok');
  } else setStatus(`${label} done`, 'ok');
  return undefined;
}

// An agent may be streaming an answer off the card right now. The server
// refuses to stop a model in that state, and says which one is working. Rather
// than printing that refusal and leaving the reader stuck, ask: cutting the
// answer short is sometimes exactly what is wanted — a model wedged in a bad
// state has to be stoppable — but it should be a decision, not an accident.
async function askThenForce(label, work, busy) {
  const who = busy.instance_id || 'A model';
  const forced = await confirmDestructive({
    title: `${who} is busy`,
    body: busy.answering
      ? `${who} is answering a request right now. Going ahead cuts that answer `
        + 'off in the middle, and whatever asked for it sees the connection drop.'
      : `${who} is still loading, for a request that is already waiting. `
        + 'Going ahead abandons that load.',
    confirmLabel: 'Stop it anyway',
  });
  if (!forced) {
    setStatus(`Left ${who} alone.`, 'ok');
    return;
  }
  setStatus(`${label}, forced…`);
  await report(label, await work(true));
}

// `work` takes one argument: whether to go ahead despite a busy card.
async function run(label, work) {
  setStatus(`${label}…`);
  try {
    await report(label, await work(false));
  } catch (error) {
    if (error.busy) {
      try {
        await askThenForce(label, work, error.busy);
      } catch (again) {
        await failed(label, again.message);
      }
    } else await failed(label, error.message);
  }
  redraw();
}

async function removeInstance(instance) {
  const confirmed = await confirmDestructive({
    title: `Remove ${instance.name}?`,
    body: 'This removes only the configured entry from Models. '
          + 'The downloaded model files remain in Library.',
    confirmLabel: 'Remove',
  });
  if (!confirmed) return;
  setStatus(`Removing ${instance.name}…`);
  try {
    await api.deleteInstance(instance.id);
    open.delete(instance.id);
    setStatus(`Removed ${instance.name}. Its files are still in Library.`, 'ok');
  } catch (error) {
    setStatus(error.message, 'error');
  }
  await redraw();
}

// -- one row ----------------------------------------------------------------

// The one thing on the line besides the name and the engine: which model this
// entry runs. Everything else about it is in the tooltip.
function modelLabel(instance, models) {
  const model = models.find((item) => item.id === instance.model_id);
  return model ? model.name : instance.model_id;
}

// Everything that used to be printed under the name. It is still worth having
// — it is just not worth three lines of screen for every entry, every time.
// Hovering asks for it; the line no longer insists on it.
function details(instance, models, latest) {
  const model = models.find((item) => item.id === instance.model_id);
  const settings = instance.params || {};
  const lines = [
    model ? `${model.name} · ${model.format} · ${bytes(model.size_bytes)}`
          : instance.model_id,
    [`port ${instance.port}`,
     settings.context_size ? `ctx ${settings.context_size}` : null,
     settings.parallel ? `${settings.parallel} slot${settings.parallel === 1 ? '' : 's'}` : null,
     settings.temperature !== undefined ? `temp ${settings.temperature}` : null,
    ].filter(Boolean).join(' · '),
    instance.ready ? 'ready' : instance.running ? 'starting' : 'stopped',
    // A request can ask for the model started differently from how the entry
    // is configured — a bigger context, usually — and the settings are not
    // saved. Without this the row shows a number the running model is not
    // using, and nothing says so.
    runningDifferently(instance),
  ];
  if (latest && instance.running && latest.memory_used_mb) {
    lines.push(`this model is holding ${Math.round(latest.memory_used_mb)} MB`);
  }
  const operation = instance.last_operation;
  if (operation && operation.steps.length) {
    lines.push(operation.steps
      .map((step) => `${step.phase} ${seconds(step.elapsed_ms)}`)
      .concat(`total ${seconds(operation.total_ms)}`)
      .join(' · '));
  }
  return lines.filter(Boolean).join('\n');
}

// What it is actually running with, when that is not what it says above.
function runningDifferently(instance) {
  const active = instance.active_params || {};
  const keys = Object.keys(active);
  if (!keys.length) return null;
  return 'started by a request with '
    + keys.map((key) => `${key} ${active[key]}`).join(', ')
    + ' — not saved, so it returns to the settings above when reloaded';
}

function formatOf(instance, models) {
  const model = models.find((item) => item.id === instance.model_id);
  return model ? model.format : '';
}

function card(instance, models, engines) {
  const engine = engines.find((item) => item.id === instance.engine);
  const specs = engine ? engine.params : [];
  const expanded = open.has(instance.id);
  const form = expanded ? settingsForm(specs, instance.params || {}) : null;

  // The label is the only part of an entry a person writes themselves, so it is
  // editable like any other setting rather than fixed at creation. Leaving it
  // empty is allowed and means "no label": the row then reads as the engine and
  // the model, which is all some entries ever needed.
  const nameField = expanded
    ? element('input', {
        value: instance.name || '', size: 24, placeholder: 'No label',
        title: 'What this entry is called. Empty is fine.',
      })
    : null;
  const edits = () => ({
    name: nameField.value.trim(),
    model_id: chooser.value,
    params: form.read(),
  });

  const latest = progress.get(instance.id);
  if (!instance.running) progress.delete(instance.id);
  const bar = element('div', { class: 'bar thin', 'data-bar': instance.id },
                      element('span', {}));
  if (instance.running && latest) paint(bar, latest); else paintFromState(bar, instance);

  const chooser = element('select', { 'data-model': instance.id },
    models
      .filter((item) => !engine || engine.formats.includes(item.format))
      .map((item) => element('option', {
        value: item.id, text: `${item.name} · ${bytes(item.size_bytes)}`,
        ...(item.id === instance.model_id ? { selected: 'selected' } : {}),
      })));

  // Always visible so the row does not change shape, but only usable when it
  // would do something: applying settings means restarting the model.
  const apply = element('button', {
    class: 'action', text: 'Apply & reload',
    title: instance.running
      ? (expanded ? 'Save these settings and restart the model with them'
                  : 'Open Settings to change something first')
      : 'Nothing is running — use Load',
    ...(instance.running && expanded ? {} : { disabled: 'disabled' }),
    onclick: () => run(`Reloading ${instance.name || instance.id}`, (force) =>
      api.apply(instance.id, edits(), force)),
  });

  const primary = instance.running
    ? element('button', {
        class: 'action', text: 'Unload',
        onclick: () => run(`Unloading ${instance.name || instance.id}`,
                           (force) => api.unload(instance.id, force)),
      })
    : element('button', {
        class: 'action', text: 'Load',
        onclick: () => run(`Loading ${instance.name || instance.id}`, async (force) => {
          if (expanded) await api.update(instance.id, edits());
          return api.load(instance.id, force);
        }),
      });

  const remove = element('button', {
    class: 'action danger', text: 'Remove',
    title: instance.running
      ? 'Unload this entry before removing it'
      : 'Remove this entry; downloaded files stay in Library',
    ...(instance.running ? { disabled: 'disabled' } : {}),
    onclick: () => removeInstance(instance),
  });

  // Only while it is running. A stopped model has a journal on Linux and no
  // log at all on macOS, so offering it would work on one machine and not the
  // other — and why a model *would not start* is answered by the sentence a
  // failed load already carries.
  const logsButton = element('button', {
    class: 'action', text: watching() === instance.id ? 'Hide log' : 'Log',
    title: instance.running
      ? 'What this engine is printing about itself'
      : 'Only while the model is running',
    ...(instance.running ? {} : { disabled: 'disabled' }),
    onclick: () => toggleLogs(instance),
  });

  const settingsButton = element('button', {
    class: 'action', text: expanded ? 'Hide settings' : 'Settings',
    onclick: () => {
      if (expanded) open.delete(instance.id); else open.add(instance.id);
      redraw();
    },
  });

  // One line per entry: what it is called, which engine runs it, which model.
  // Which engine matters enough to be on the line rather than in the tooltip —
  // llama.cpp and vLLM behave differently enough that reading a row without
  // knowing which one it is tells you very little.
  //
  // Everything else — port, context, slots, temperature, size, state, and how
  // long the last load took — moved into the tooltip. It was three lines of
  // screen per entry for numbers that are only wanted occasionally.
  //
  // Removing the configured entry lives here; Library deletes the weight
  // files. Two actions, two pages, so it is always clear which kind of data
  // is about to disappear.
  //
  // The Remove button is not optional. Deleting a model in Library is refused
  // while an entry points at it — otherwise the entry could never start
  // again — and that refusal sends the reader here. Take this away and the
  // pair becomes a dead end: an entry that cannot be removed and a model
  // that cannot be deleted.
  const head = element('div', {
    class: 'row instance',
    title: details(instance, models, latest),
  }, [
    // Left: what you named it, and what it runs. This is what you read down the
    // page to find a row.
    element('div', { class: 'inline ident' }, [
      instance.name ? element('strong', { text: instance.name }) : null,
      element('span', { class: 'muted model', text: modelLabel(instance, models) }),
    ].filter(Boolean)),
    // Right: what will actually run it, then the things you press. Format and
    // engine are a pair — nvfp4 on vLLM, gguf on llama.cpp — so they stay
    // together, at the end, out of the middle of the name.
    element('div', { class: 'inline' }, [
      chatLink(instance),
      formatOf(instance, models)
        ? element('span', { class: 'pill format', text: formatOf(instance, models) })
        : null,
      element('span', { class: 'pill engine',
                        text: engine ? engine.name : instance.engine }),
      settingsButton, logsButton, primary, apply, remove,
    ].filter(Boolean)),
  ]);

  return element('div', { class: 'card instance' }, [
    head,
    expanded
      ? element('div', { class: 'settings-open' }, [
          element('label', { class: 'field' }, [
            element('span', { text: 'Label' }), nameField,
          ]),
          element('label', { class: 'field' }, [
            element('span', { text: 'Model' }), chooser,
          ]),
          form.node,
        ])
      : null,
    bar,
  ].filter(Boolean));
}

// llama.cpp serves a chat page of its own, on the model's own port.
//
// Shown only when there is something on the other end, and first in the row's
// right-hand group so that it can be. That group sits against the right edge
// and is only as wide as its contents, so removing something from it moves
// whatever is to its *left* and nothing to its right. First means nothing
// moves at all — which is why the format pill, the other thing here that comes
// and goes, is also at that end.
//
// Greying it out instead was worse. vLLM serves no page and never will, so a
// disabled button sat on eleven rows out of eleven promising something six of
// them could never do.
//
// Shaped like the pills it sits beside rather than like the buttons after
// them, because it belongs with the pills: they describe the model, and this
// one says the model is answering and here is the door. It borrows their rules
// outright — same size, same round ends — and differs only in colour: the
// green a finished load ends on, which is the state this link exists in.
//
// No arrow on it. The convention for one is "opens elsewhere", which is true,
// but the two pills beside it are grey and do nothing, and green among them
// already reads as the one you can press.
//
// The address is built from the page you are looking at, not from the server's
// idea of itself: the manager may be reached by name, by address, or through a
// tunnel, and the engine sits on the same host under a different port.
function chatLink(instance) {
  if (!instance.ready || !instance.web_ui) return null;
  return element('a', {
    class: 'pill chat-link', target: '_blank', rel: 'noopener',
    href: `${window.location.protocol}//${window.location.hostname}:${instance.port}/`,
    title: 'Open the chat page this engine serves, on its own port',
    text: 'Chat',
  });
}

// -- adding -----------------------------------------------------------------

function addCard(form) {
  const usable = form.models.filter((model) =>
    form.engines.some((engine) => engine.available && engine.formats.includes(model.format)));

  if (!usable.length) {
    return element('div', { class: 'card' }, [
      element('h3', { text: 'Add a model' }),
      element('p', { class: 'muted',
                     text: 'No usable models on disk. Download one first.' }),
      element('button', { class: 'action', text: 'Cancel',
                          onclick: () => { adding = false; redraw(); } }),
    ]);
  }

  const chooser = element('select', {},
    usable.map((model) => element('option', {
      value: model.id,
      text: `${model.name} · ${model.format} · ${bytes(model.size_bytes)}`,
    })));
  const name = element('input', { placeholder: 'Name, e.g. Coding', size: 28 });
  const port = element('input', { type: 'number', value: String(form.port), size: 8 });

  const engineFor = (modelId) => {
    const model = usable.find((item) => item.id === modelId);
    return form.engines.find((engine) =>
      engine.available && engine.formats.includes(model.format));
  };

  let settings = settingsForm(engineFor(chooser.value).params, {});
  const holder = element('div', {}, settings.node);
  chooser.addEventListener('change', () => {
    settings = settingsForm(engineFor(chooser.value).params, {});
    holder.replaceChildren(settings.node);
  });

  return element('div', { class: 'card' }, [
    element('h3', { text: 'Add a model' }),
    element('label', { class: 'field' }, [element('span', { text: 'Model' }), chooser]),
    element('label', { class: 'field' }, [element('span', { text: 'Name' }), name]),
    element('label', { class: 'field' }, [
      element('span', {}, [
        element('span', { text: 'Port' }),
        element('small', { class: 'muted', text: 'first free one, change if a client expects another' }),
      ]),
      port,
    ]),
    holder,
    element('div', { class: 'row' }, [
      element('span', {}),
      element('div', {}, [
        element('button', { class: 'action', text: 'Cancel',
                            onclick: () => { adding = false; redraw(); } }),
        element('button', {
          class: 'action', text: 'Add',
          onclick: async () => {
            const model = usable.find((item) => item.id === chooser.value);
            const label = name.value.trim() || model.name;
            await run(`Adding ${label}`, () => api.createInstance({
              id: label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''),
              name: label,
              engine: engineFor(chooser.value).id,
              model_id: chooser.value,
              port: parseInt(port.value, 10),
              params: settings.read(),
            }));
            adding = false;
            redraw();
          },
        }),
      ]),
    ]),
  ]);
}

// -- the view ---------------------------------------------------------------

export async function render(container) {
  subscribe();
  redraw = () => render(container);
  // The panel is pinned to the window, not drawn in the list, so the button
  // label is the only thing here that has to follow it.
  onPanelChange(() => redraw());

  const [instances, models, settings] = await Promise.all([
    api.instances(), api.models(), api.settings(),
  ]);
  const engines = settings.engines;

  const header = element('div', { class: 'row' }, [
    adding ? null : element('button', {
      class: 'action', text: '+ Add model',
      onclick: () => { adding = true; redraw(); },
    }),
  ]);

  let addSection = null;
  if (adding) addSection = addCard(await api.newInstanceForm());

  container.replaceChildren(...[
    header,
    addSection,
    ...(instances.length
      ? instances.map((instance) => card(instance, models, engines))
      : [element('p', { class: 'muted', text: 'Nothing configured yet. Add a model.' })]),
  ].filter(Boolean));
}
