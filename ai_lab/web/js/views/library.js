// The library: every model on disk, and how to get more.
//
// Downloading and deleting live on the same page as the list because they are
// the same subject seen from three angles — what you have, what you want, what
// you no longer want. Splitting them meant knowing which page a model was on
// before you could do anything with it.

import { api } from '../api.js';
import { confirmDestructive, showNotice } from '../confirm.js';
import { whileWorking } from '../working.js';
import { capabilities } from '../icons.js';
import { bytes, element } from '../format.js';

let results = [];        // Hugging Face search results
let sets = [];           // downloadable models in the opened repository
let openRepo = null;
let lastQuery = '';
let currentTransfers = [];
let redraw = () => {};

// Built once and reused on every render. The page refreshes itself every few
// seconds, and rebuilding these would wipe whatever is being typed and steal
// the focus mid-word.
let searchRow = null;
let searchInput = null;
let searchGo = null;

// -- what is on disk --------------------------------------------------------

async function remove(model) {
  const confirmed = await confirmDestructive({
    title: `Delete ${model.name}?`,
    body: `${model.file_count} file${model.file_count === 1 ? '' : 's'}, `
          + `${bytes(model.size_bytes)}, will be removed from disk. `
          + 'This cannot be undone.',
  });
  if (!confirmed) return;
  try {
    await api.deleteModel(model.id);
  } catch (error) {
    await showNotice({ title: `Could not delete ${model.name}`,
                       body: error.message });
  }
  redraw();
}

function modelRow(model) {
  const state = model.complete
    ? element('span', { class: 'pill on', text: 'complete' })
    : element('span', {
        class: 'pill', style: 'color:var(--warn);border-color:var(--warn)',
        text: `missing ${model.missing.length}`, title: model.missing.join(', '),
      });
  return element('tr', {}, [
    // On disk there is no entry and so nothing to switch anything off: these
    // are what the weights themselves can do.
    element('td', {}, element('span', { class: 'inline ident' }, [
      element('strong', { text: model.name }),
      ...capabilities(model.capabilities),
    ])),
    element('td', { class: 'muted', text: model.format }),
    element('td', { class: 'number', text: bytes(model.size_bytes) }),
    element('td', { class: 'number muted', text: String(model.file_count) }),
    element('td', {}, state),
    element('td', { class: 'number' },
      element('button', { class: 'action danger', text: 'Delete',
                          onclick: (event) => whileWorking(
                            event.target, 'Deleting…', () => remove(model)) })),
  ]);
}

function repositorySection(repository, models) {
  const heading = `${repository.name} · ${repository.format} · `
                  + `${bytes(repository.free_bytes)} free`;
  return element('section', {}, [
    element('h3', { text: heading }),
    models.length
      ? element('table', {}, [
          element('thead', {}, element('tr', {}, [
            element('th', { text: 'Model' }),
            element('th', { text: 'Format' }),
            element('th', { class: 'number', text: 'Size' }),
            element('th', { class: 'number', text: 'Files' }),
            element('th', { text: '' }),
            element('th', { text: '' }),
          ])),
          element('tbody', {}, models.map(modelRow)),
        ])
      : element('p', { class: 'muted', text: 'Empty.' }),
  ]);
}

// -- getting more -----------------------------------------------------------

// What the last search came back with, said where the results are drawn.
//
// "Nothing found" and "nothing you can run" are different answers and a list of
// length zero cannot tell them apart, so the server says how many it filtered
// out and this repeats it. Kept beside the empty space it explains rather than
// at the foot of the page, which is a long way from where you are looking.
let outcome = '';

async function search(query) {
  outcome = 'Searching…';
  lastQuery = query;
  try {
    const answer = await api.search(query);
    results = answer.results;
    sets = [];
    openRepo = null;
    outcome = results.length
      ? ''
      : answer.hidden
        ? `${answer.hidden} found, none in a format this machine can run`
        : `Nothing found for “${query}”`;
  } catch (error) {
    outcome = '';
    await showNotice({ title: 'Search failed', body: error.message });
  }
}

async function openRepository(repo) {
  outcome = `Reading ${repo}…`;
  openRepo = repo;
  sets = [];
  try {
    sets = await api.remoteSets(repo);
    outcome = sets.length ? '' : `Nothing in ${repo} that this machine can run`;
  } catch (error) {
    openRepo = null;
    outcome = '';
    await showNotice({ title: `Could not read ${repo}`, body: error.message });
  }
}

// The search box, kept between redraws.
//
// Rebuilt each time it would lose what was half-typed, and the page redraws
// whenever anything changes — a download ticking along was enough to wipe a
// word mid-sentence.
function searchBox() {
  if (searchRow) {
    // The row itself survives, but whether there is anything to clear does
    // not, so that part is rebuilt.
    searchRow.replaceChildren(searchInput, searchGo, ...clearButton());
    return searchRow;
  }
  searchInput = element('input', {
    class: 'grow', placeholder: 'Search Hugging Face, e.g. qwen3 gguf',
  });
  const go = async () => {
    if (!searchInput.value.trim()) return;
    await search(searchInput.value.trim());
    redraw();
  };
  searchInput.addEventListener('keydown',
                               (event) => { if (event.key === 'Enter') go(); });
  searchGo = element('button', { class: 'action', text: 'Search', onclick: go });
  searchRow = element('div', { class: 'row' },
                      [searchInput, searchGo, ...clearButton()]);
  return searchRow;
}


// A way out of a search. Without it the results sat there for the rest of the
// session, pushing the models actually on disk — which is what this page is
// for — off the bottom of the screen.
//
// Only when there is something to clear: a button that does nothing is a
// button somebody presses once and stops trusting.
function clearButton() {
  if (!results.length && !openRepo && !outcome) return [];
  return [element('button', {
    class: 'action', text: 'Clear',
    title: 'Put the search away and show what is on disk',
    onclick: () => {
      results = [];
      sets = [];
      openRepo = null;
      outcome = '';
      lastQuery = '';
      searchInput.value = '';
      redraw();
    },
  })];
}

async function startDownload(set, button) {
  await whileWorking(button, 'Preparing…', async () => {
    try {
      // No destination: the server puts it where that format lives.
      const transfer = await api.download(set.repo, set.name);
      currentTransfers = [
        ...currentTransfers.filter((item) => item.id !== transfer.id), transfer,
      ];
      // Nothing is said on success: a progress bar for this download appears
      // on the very next redraw, which says it better than a sentence.
    } catch (error) {
      await showNotice({ title: `Could not download ${set.name}`,
                         body: error.message });
    }
  });
  await redraw();
}

async function cancelDownload(transfer) {
  try {
    await api.cancelDownload(transfer.id);
  } catch (error) {
    await showNotice({ title: `Could not cancel ${transfer.name}`,
                       body: error.message });
  }
  await redraw();
}

function matchingTransfer(set) {
  return currentTransfers.find((item) =>
    item.repo === set.repo && item.name === set.name);
}

function transferProgress(transfer) {
  if (!transfer) return null;
  const active = ['queued', 'running'].includes(transfer.state);
  let label = transfer.state;
  if (transfer.state === 'running') {
    const file = Math.min(transfer.files_done + 1, transfer.files_total);
    label = `${transfer.percent.toFixed(1)}% · ${bytes(transfer.received_bytes)} of `
      + `${bytes(transfer.total_bytes)} · file ${file} of ${transfer.files_total}`;
  }
  return element('div', { class: 'variant-progress' }, [
    element('div', { class: 'row progress-label' }, [
      element('span', {
        class: transfer.state === 'failed' ? 'error' : 'muted',
        text: label + (transfer.error ? ` · ${transfer.error}` : ''),
      }),
      active ? element('button', {
        class: 'action quiet', text: 'Cancel',
        onclick: () => cancelDownload(transfer),
      }) : null,
    ]),
    element('div', {
      class: 'bar' + (transfer.state === 'done' ? ' done' : '')
             + (transfer.state === 'failed' ? ' failed' : ''),
    }, element('span', { style: `width:${transfer.percent}%` })),
  ]);
}

// One row per variant inside a repository. A repository usually holds the same
// model at a dozen quantisations, and the format is the thing that decides
// whether this machine can run it, so it is shown on every row.
function variantRow(set) {
  const transfer = matchingTransfer(set);
  const alreadyFinished = transfer?.state === 'done';
  return element('div', { class: 'variant' }, [
    element('div', { class: 'variant-main' }, [
      element('div', { class: 'variant-name' }, [
        element('strong', { text: set.name }),
        element('span', {
          class: 'muted',
          text: ` · ${bytes(set.size_bytes)} · ${set.files.length} file`
                + `${set.files.length === 1 ? '' : 's'}`,
        }),
      ]),
      element('div', { class: 'inline variant-actions' }, [
        element('span', { class: 'pill', text: set.format }),
        !set.complete
          ? element('span', { class: 'warn',
                              text: `incomplete upstream (${set.missing.length})` })
          : alreadyFinished
            ? element('span', { class: 'pill on', text: 'Downloaded' })
            : ['queued', 'running'].includes(transfer?.state)
              ? element('span', { class: 'pill on', text: transfer.state })
              : element('button', {
                class: 'action',
                text: transfer?.state === 'failed' ? 'Retry'
                  : transfer?.state === 'cancelled' ? 'Resume' : 'Download',
                onclick: (event) => startDownload(set, event.currentTarget),
              }),
      ]),
    ]),
    transferProgress(transfer),
  ]);
}

// The variants appear directly beneath the repository they came from. They used
// to be rendered below the whole result list, which meant clicking a button
// changed something far off the bottom of the screen and looked like nothing
// had happened at all.
function repositoryRow(item) {
  const isOpen = openRepo === item.repo;
  const rows = [
    element('div', { class: 'row' }, [
      element('div', {}, [
        element('strong', { text: item.repo }),
        element('span', { class: 'muted',
                          text: ` · ${item.downloads.toLocaleString()} downloads` }),
      ]),
      element('button', {
        class: 'action',
        text: isOpen ? 'Hide' : 'Show models',
        onclick: async () => {
          if (isOpen) { openRepo = null; sets = []; redraw(); return; }
          await openRepository(item.repo);
          redraw();
        },
      }),
    ]),
  ];

  if (isOpen) {
    rows.push(element('div', { class: 'variants' },
      sets.length
        ? sets.map(variantRow)
        : [element('p', { class: 'muted',
                          text: 'Nothing here this machine can run. '
                                + 'Untick the filter to see every format.' })]));
  }
  return element('div', {}, rows);
}

function transferList(transfers) {
  if (!transfers.length) return null;
  return element('div', {}, [
    element('h4', { text: 'Transfers' }),
    ...transfers.map((transfer) => element('div', {}, [
      element('div', { class: 'row' }, [
        element('div', {}, [
          element('strong', { text: transfer.name }),
          element('span', {
            class: transfer.state === 'failed' ? 'error' : 'muted',
            text: ` · ${transfer.state} · ${bytes(transfer.received_bytes)}`
                  + ` of ${bytes(transfer.total_bytes)}`
                  + (transfer.error ? ` · ${transfer.error}` : ''),
          }),
        ]),
        ['running', 'queued'].includes(transfer.state)
          ? element('button', {
              class: 'action', text: 'Cancel',
              onclick: () => cancelDownload(transfer),
            })
          : null,
      ]),
      element('div', { class: 'bar' + (transfer.state === 'done' ? ' done' : '') },
              element('span', { style: `width:${transfer.percent}%` })),
    ])),
  ]);
}

// -- the view ---------------------------------------------------------------

export async function render(container) {
  redraw = () => render(container);
  const [models, settings, transfers] = await Promise.all([
    api.models(), api.settings(), api.transfers(),
  ]);
  currentTransfers = transfers;

  const byRepository = new Map(settings.repositories.map((item) => [item.id, []]));
  models.forEach((model) => {
    const key = model.id.split('/')[0];
    if (!byRepository.has(key)) byRepository.set(key, []);
    byRepository.get(key).push(model);
  });

  const sections = [...byRepository.entries()]
    .filter(([, entries]) => entries.length > 0)
    .map(([id, entries]) => {
      const repository = settings.repositories.find((item) => item.id === id);
      return repository ? repositorySection(repository, entries) : null;
    })
    .filter(Boolean);

  // Getting models comes first, then what you already have.
  //
  // replaceChildren turns a null into the text "null", so the list is filtered
  // rather than handed straight over — two empty slots showed up on the page
  // as "nullnull".
  const children = [
    searchBox(),
    // Right under the box that was typed into, above where the results would
    // be. Empty whenever there are results to look at instead.
    outcome ? element('p', { class: 'muted outcome', text: outcome }) : null,
    ...results.map(repositoryRow),
    transferList(transfers.filter((transfer) =>
      ['queued', 'running', 'failed'].includes(transfer.state)
      && transfer.repo !== openRepo)),
    element('hr', {}),
    ...sections,
  ].filter(Boolean);
  container.replaceChildren(...children);
}
