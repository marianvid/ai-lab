// Remote model discovery, downloads, and transfers.
import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { whileWorking } from '../working.js';
import { bytes, element } from '../format.js';

export function createDiscovery() {
  let results = [];
  let sets = [];
  let openRepo = null;
  let lastQuery = '';
  let currentTransfers = [];
  let downloadRoots = [];
  let redraw = () => {};
  let searchRow = null;
  let searchInput = null;
  let searchGo = null;

  function setDiscoveryContext(settings, transfers, callback) {
    redraw = callback;
    downloadRoots = settings.model_roots || [{
      id: 'core', name: 'Core', path: settings.models_root || '',
      enabled: true, writable: true, exists: true,
      free_bytes: Math.max(0, ...(settings.repositories || []).map(
        (repository) => repository.free_bytes || 0)),
    }];
    currentTransfers = transfers;
  }

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
    const storageTier = await chooseDownloadDestination(set);
    if (!storageTier) return;
    await whileWorking(button, 'Preparing…', async () => {
      try {
        const transfer = await api.download(set.repo, set.name, storageTier);
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

  function chooseDownloadDestination(set) {
    return new Promise((resolve) => {
      const roots = downloadRoots.map((root) => {
        const unavailable = root.enabled === false || root.exists === false
          || root.writable === false;
        const tooSmall = Number.isFinite(root.free_bytes)
          && root.free_bytes < set.size_bytes;
        return { ...root, unavailable, tooSmall, usable: !unavailable && !tooSmall };
      });
      const preferred = roots.find((root) => root.id === 'benchmark' && root.usable)
        || roots.find((root) => root.id === 'core' && root.usable)
        || roots.find((root) => root.usable);
      let answer = null;
      let selected = preferred?.id || '';

      const options = roots.map((root) => destination(root, set, () => {
        selected = root.id;
      }, root.id === selected));
      const cancel = element('button', {
        class: 'action', autofocus: 'autofocus', text: 'Cancel',
        onclick: () => dialog.close(),
      });
      const proceed = element('button', {
        class: 'action primary', text: 'Start download',
        ...(!preferred ? { disabled: 'disabled' } : {}),
        onclick: () => { answer = selected; dialog.close(); },
      });
      const dialog = element('dialog', { class: 'confirm download-destination' }, [
        element('h3', { text: 'Download model' }),
        element('div', { class: 'download-model' }, [
          element('strong', { text: set.repo }),
          element('div', { text: set.name }),
          element('span', { class: 'muted',
                            text: `${bytes(set.size_bytes)} · ${set.files.length} file`
                              + `${set.files.length === 1 ? '' : 's'}` }),
        ]),
        element('h4', { text: 'Where should it be stored?' }),
        element('div', { class: 'destination-list' }, options.length ? options : [
          element('p', { class: 'error', text: 'No model storage is configured.' }),
        ]),
        element('div', { class: 'row buttons' }, [cancel, proceed]),
      ]);
      dialog.addEventListener('close', () => {
        dialog.remove();
        resolve(answer);
      });
      document.body.append(dialog);
      dialog.showModal();
      cancel.focus();
    });
  }

  function destination(root, set, select, checked) {
    const production = root.id === 'core';
    let reason = production
      ? 'For approved models used by AI-Lab.'
      : 'Recommended for models that have not been evaluated.';
    if (root.enabled === false) reason = 'This storage is disabled.';
    else if (root.exists === false) reason = 'The storage path is not mounted.';
    else if (root.writable === false) reason = 'The storage path is read-only.';
    else if (root.tooSmall) reason = `Needs ${bytes(set.size_bytes)}, but only ${bytes(root.free_bytes)} is free.`;
    return element('label', {
      class: `destination${root.usable ? '' : ' unavailable'}`,
    }, [
      element('input', {
        type: 'radio', name: 'download-destination', value: root.id,
        ...(checked ? { checked: 'checked' } : {}),
        ...(!root.usable ? { disabled: 'disabled' } : {}),
        onchange: select,
      }),
      element('span', { class: 'destination-copy' }, [
        element('span', { class: 'destination-title',
                          text: production ? 'Production' : 'Temporary / benchmark' }),
        element('span', { class: 'path muted',
                          text: `${root.path || 'No path'} · ${bytes(root.free_bytes || 0)} available` }),
        element('span', { class: root.usable ? 'muted' : 'error', text: reason }),
      ]),
    ]);
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
                  text: transfer?.state === 'failed' ? 'Retry…'
                    : transfer?.state === 'cancelled' ? 'Resume…' : 'Download…',
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


  function searchResults() { return { outcome, results, openRepo }; }

  return { setDiscoveryContext, searchResults, searchBox, repositoryRow, transferList };
}
