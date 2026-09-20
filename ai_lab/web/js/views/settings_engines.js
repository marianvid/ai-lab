// Engine status, version reviews, managed components and build logs.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { whileWorking } from '../working.js';
import { reviewUpdate } from './whatchanges.js';
import { element } from '../format.js';
import { section } from './settings_section.js';

export function createEngineSection(logs, byEngine) {
  // Whether this engine can run, and nothing else.
  //
  // It used to turn orange and announce a new version, which put the news in
  // competition with the state: an engine working perfectly well looked like
  // something was wrong with it, and both engines could not be read the same
  // way. What there is to update to lives on the line below, beside the button
  // that reads about it.
  // Said only when there is something to say.
  //
  // A green "available" on an engine that is plainly working told nobody
  // anything: the version beside its name already says it runs, and a badge that
  // is always there is a badge nobody reads. What is left are the two states
  // worth interrupting for — it cannot run here, or it is being rebuilt now.
  function engineState(engine) {
    const source = engine.source;
    if (!engine.available) {
      return element('span', { class: 'pill off', text: engine.reason || 'not installed' });
    }
    if (source && source.state === 'running') {
      return element('span', { class: 'pill', text: 'building…' });
    }
    return null;
  }

  // What this engine could be updated to, and whether that is anything.
  //
  // A checkout is asked with git and a package engine is asked of its index,
  // both on a timer, so this page already knows when it is opened. Empty when
  // upstream could not be reached — an engine nobody could ask about must not
  // look like one with nothing waiting.
  function updateWaiting(engine) {
    const source = engine.source;
    if (source && source.update_available) return source.latest || '';
    const installed = byEngine.get(engine.id);
    return installed && installed.update_available ? installed.latest || '' : '';
  }


  // Which version is running, whichever way this engine arrives: a build number
  // read from the checkout, or the version of the environment its packages are
  // installed in. vLLM has no checkout and so had no version beside its name at
  // all, which made it look like the one engine nobody could tell anything
  // about.
  function installedVersion(engine) {
    const source = engine.source;
    if (source && source.installed) return source.installed;
    const installed = byEngine.get(engine.id);
    return (installed && installed.installed) || '';
  }


  // What pressing the real Update button does, which depends on how this engine
  // arrives. Null when neither applies, so the review shows what would change
  // without offering a button it cannot honour.
  function take(engine, buildable, installable, refresh) {
    if (installable) {
      return async () => {
        try {
          await api.installEngine(engine.id, '');
          logs.set(engine.id, []);
          refresh();
        } catch (error) {
          await showNotice({ title: `Could not install ${engine.name}`,
                             body: error.message });
        }
      };
    }
    if (buildable) {
      return async () => {
        try {
          await api.updateBuild(engine.id);
          logs.set(engine.id, []);
          refresh();
        } catch (error) {
          await showNotice({ title: `Could not build ${engine.name}`,
                             body: error.message });
        }
      };
    }
    return null;
  }

  // The update line: what is waiting, and the way to read about it.
  //
  // **No button here updates anything.** Every one opens what the update would
  // bring — what changes, what upstream wrote, which packages would be replaced
  // — and the real Update sits at the foot of that. An update taken without
  // reading it is a hope, and this machine runs one card.
  //
  // "Check for updates" used to sit here and is gone. Upstream is asked on a
  // timer anyway, so the button did what was already being done, and its only
  // real effect was to make the page look like it needed pressing.
  // What is waiting, and the way to read about it. Nothing at all when there is
  // nothing waiting: an engine at the newest version has no update to review, so
  // a button offering to review one would open a page saying "no changes".
  //
  // **This never updates anything.** It opens what the update would bring — what
  // changes, what upstream wrote, which packages would be replaced — and the real
  // Update is at the foot of that. The ellipsis is the promise.
  function updateControls(engine, source, refresh, waiting) {
    // Two ways an engine gets a new version, and an engine has exactly one of
    // them: a checkout is rebuilt in place, packages are installed into a new
    // folder beside the old one.
    const buildable = Boolean(source && source.exists);
    const installable = byEngine.has(engine.id);

    return [
      element('span', { class: 'pill',
                        style: 'color:var(--warn);border-color:var(--warn)',
                        text: `${waiting} available` }),
      element('button', {
        class: 'action', text: 'Update…',
        title: `Read what ${waiting} brings before taking it`,
        onclick: (event) => whileWorking(event.target, 'Reading…', async () => {
          await reviewUpdate(engine, take(engine, buildable, installable, refresh));
          // Reading the review can take several seconds. Update checks may have
          // completed while the dialog was open, so redraw from one fresh pair
          // of Settings responses instead of leaving whichever status happened
          // to be on the page before it opened.
          await refresh();
        }),
      }),
    ];
  }


  // An empty right side made an engine look as though update support was
  // missing. Say which of the three honest states it is in: something is
  // waiting, upstream was checked and there is nothing, or upstream could not
  // yet be read. Engines whose installation is not managed say that explicitly.
  function updateState(engine, source, refresh) {
    const installed = byEngine.get(engine.id);
    if (!engine.available && installed && !installed.installed) {
      return [element('button', {
        class: 'action', text: 'Install…',
        onclick: (event) => whileWorking(event.target, 'Reading…', async () => {
          await reviewUpdate(engine, take(engine, false, true, refresh));
          await refresh();
        }),
      })];
    }
    if (!engine.available) return [];
    const waiting = updateWaiting(engine);
    if (waiting) return updateControls(engine, source, refresh, waiting);
    const managed = Boolean((source && source.exists) || installed);
    if (!managed) {
      return [element('span', { class: 'muted', text: 'Updates not managed' })];
    }
    const known = Boolean((source && source.latest) || (installed && installed.latest));
    return [element('span', { class: 'muted',
                              text: known ? 'No update available'
                                          : 'Update status unavailable' })];
  }



  function engineEntry(engine, refresh) {
    const source = engine.source;
    const stored = logs.get(engine.id) || (source && source.log) || [];
    const version = installedVersion(engine);

    // One line: the name and what is running, then — pushed to the right — what
    // is waiting and the way to read about it. There is no second line, because
    // an engine at its newest version has nothing to say on one.
    //
    // The weight formats used to hang off the name — "llama.cpp · b10448 · gguf"
    // — which said something a reader of this page cannot act on, and which the
    // model list says better by only offering an entry the formats its engine
    // can read.
    const rows = [
      element('div', { class: 'row engine' }, [
        element('div', { class: 'grow' }, [
          element('strong', { text: engine.name }),
          version ? element('span', { class: 'muted', text: ` · ${version}` }) : null,
        ].filter(Boolean)),
        engineState(engine),
        ...updateState(engine, source, refresh),
      ].filter(Boolean)),
    ];

    if (source && source.note) rows.push(element('div', { class: 'warn', text: source.note }));
    if (source && source.error) rows.push(element('div', { class: 'error', text: source.error }));
    const building = Boolean(source && source.state === 'running');
    if (stored.length || building) rows.push(buildLog(engine, stored, building));
    const managed = byEngine.get(engine.id);
    if (managed && managed.components && managed.components.length) {
      rows.push(componentList(engine, managed.components, refresh));
    }

    return element('div', { class: 'engine-entry' }, rows);
  }

  function componentList(engine, components, refresh) {
    return element('details', { class: 'fold components' }, [
      element('summary', { text: `${components.length} managed custom node${components.length === 1 ? '' : 's'}` }),
      ...components.map((item) => element('div', { class: 'row tight' }, [
        element('span', { class: 'grow', text: item.name }),
        element('span', { class: 'muted', text: item.installed || 'unknown' }),
        item.dirty ? element('span', { class: 'pill warn', text: 'local changes' }) : null,
        item.update_available ? element('button', {
          class: 'action', text: `Update to ${item.latest}`,
          ...(item.dirty ? { disabled: 'disabled' } : {}),
          onclick: (event) => whileWorking(event.target, 'Building…', async () => {
            try {
              await api.updateInstallComponent(engine.id, item.name);
              logs.set(engine.id, []);
            } catch (error) {
              await showNotice({ title: `Could not update ${item.name}`, body: error.message });
            }
            refresh();
          }),
        }) : null,
      ].filter(Boolean))),
    ]);
  }

  // What a build said, while it says it and afterwards.
  //
  // A compile takes ten minutes and nothing on screen is worse than a wall of
  // output, so while it runs the pane is open and follows itself. When it is
  // over the same output is worth keeping — that is where "Finished at v0.2.0"
  // is, and where a failure explains itself — but a page that opens into eight
  // hundred lines of cmake every time is a page with a wall in it.
  //
  // So afterwards it folds. Available, closed, and it says which build it was
  // from rather than making that another thing to remember.
  function buildLog(engine, lines, building) {
    const pane = element('pre', { class: 'log', 'data-log': engine.id,
                                  text: lines.join('\n') });
    if (building) return pane;
    const last = lines.filter(Boolean).slice(-1)[0] || '';
    return element('details', { class: 'fold build-log' }, [
      element('summary', { text: last.startsWith('Finished')
        ? `Last update log · ${last.replace(/^Finished at /, 'finished at ')}`
        : 'Last update log' }),
      pane,
    ]);
  }

  function engines(list, refresh) {
    // All engines are one compact subject. A separator retains the scan line
    // that separate cards used to provide without spending their repeated
    // borders, margins and padding on every row.
    const found = section('Engines', list.map((engine) => engineEntry(engine, refresh)));
    found.classList.add('engines');
    return found;
  }

    return engines;
}
