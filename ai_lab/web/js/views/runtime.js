// The model list: what is configured, what is loaded, and managing both.
//
// One row is one model. There is no "swap" — changing which model a row runs
// is an edit followed by a reload, which is the same act as starting it for
// the first time.

import { api } from '../api.js';
import { confirmDestructive, showNotice } from '../confirm.js';
import { whileWorking } from '../working.js';
import { element } from '../format.js';
import { onPanelChange } from '../logpane.js';
import { createProgress } from './runtime_progress.js';
import { addCard } from './runtime_add.js';
import { createCard } from './runtime_card.js';

const open = new Set();         // rows with their settings expanded
let adding = false;
let redraw = () => {};
const { progress, paint, paintFromState, subscribe } = createProgress();
const card = createCard({ progress, open, paint, paintFromState, run, removeInstance,
  redraw: () => redraw() });

// -- actions ----------------------------------------------------------------

// A failure interrupts; a success says nothing.
//
// A success has already been shown: the row redraws, the bar fills, the state
// changes, and how long the load took appears on the row itself. A sentence
// repeating that is noise. A failure is the opposite — the explanation is the
// useful part, an engine that refuses a context says which one would have
// fitted, and it is worth taking the page away for.
function failed(label, detail) {
  return showNotice({ title: `${label} failed`, body: detail });
}

function report(label, result) {
  const operation = result.operation || result;
  if (operation.ok === false) return failed(label, operation.error);
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
  if (!forced) return;
  await report(label, await work(true));
}

// `work` takes one argument: whether to go ahead despite a busy card.
async function run(label, work) {
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
    title: `Remove ${instance.id}?`,
    body: 'This removes only the configured entry from Models. '
          + 'The downloaded model files remain in Library.',
    confirmLabel: 'Remove',
  });
  if (!confirmed) return;
  try {
    await api.deleteInstance(instance.id);
    open.delete(instance.id);
  } catch (error) {
    await failed(`Removing ${instance.id}`, error.message);
  }
  await redraw();
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
  if (adding) addSection = addCard(await api.newInstanceForm(), {
    run, close: () => { adding = false; redraw(); },
  });

  container.replaceChildren(...[
    header,
    addSection,
    ...(instances.length
      ? instances.map((instance) => card(instance, models, engines))
      : [element('p', { class: 'muted', text: 'Nothing configured yet. Add a model.' })]),
  ].filter(Boolean));
}
