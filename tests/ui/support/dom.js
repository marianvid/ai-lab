// A real DOM for the tests.
//
// jsdom rather than a hand-written stand-in, deliberately. The faults that
// reached the user came from how a browser actually behaves — replaceChildren
// turning a null into the text "null", a rebuilt input losing what was being
// typed — and a stand-in written by the same hand that wrote the bugs would
// encode the same assumptions and miss them again.

import { afterEach } from 'node:test';
import { JSDOM } from 'jsdom';

const PAGE = `<!doctype html><html><body>
  <header><div id="host-summary"></div></header>
  <nav id="tabs"></nav>
  <main id="view"></main>
  <footer id="status"></footer>
</body></html>`;

// Every page made since the last test ended, checked after each test: a view
// that asked for a path no test answered (misspelt, removed, or new and
// forgotten) fails that test instead of quietly getting 200 and [] (the
// reviewer's item 14). All of them, not only the last: a test that builds two
// pages must not lose what the first one asked (the reviewer, on 98a3818).
let pages = [];
afterEach(() => {
  const made = pages;
  pages = [];
  const asked = made.flatMap((page) => page.unmocked);
  // Closed, so no view's timer keeps the run alive or reaches the next test.
  for (const page of made) page.window.close();
  if (asked.length) {
    throw new Error(`the view asked for paths this test does not answer: ${[...new Set(asked)].join(', ')}`);
  }
});

// The reads every page makes in passing, answered with their real empty shapes
// when a test does not answer them itself: an array where a view reads
// `.fields` or `.token_holds` would send it down a path no user sees.
const BACKGROUND = {
  'GET /api/file-settings': { sections: [] },
  'GET /api/setting-fields': { groups: {}, fields: [] },
  'GET /api/builds': [],
  'GET /api/installs': [],
  'GET /api/session': { login_required: false, logged_in: true, token_holds: true },
};

// Every request the interface makes, and what to answer with. A view under
// test never reaches the network. `quiet` lists further paths a test does not
// care about, answered with []; any other unanswered path fails the test.
export function installDom(responses = {}, { quiet = [] } = {}) {
  const dom = new JSDOM(PAGE, { url: 'http://localhost:8090/' });
  const { window } = dom;

  global.window = window;
  global.document = window.document;
  global.CSS = window.CSS;
  global.HTMLElement = window.HTMLElement;
  global.confirm = () => true;

  const calls = [];
  const unmocked = [];
  pages.push({ unmocked, window });
  global.fetch = async (path, options = {}) => {
    calls.push({ path, method: options.method || 'GET', body: options.body });
    const route = path.split('?')[0];
    const key = `${options.method || 'GET'} ${route}`;
    const answered = key in responses || route in responses;
    const background = key in BACKGROUND;
    if (!answered && !background && !quiet.includes(route) && !quiet.includes(key)) {
      unmocked.push(key);
    }
    const payload = key in responses ? responses[key]
                  : route in responses ? responses[route]
                  : background ? structuredClone(BACKGROUND[key])
                  : [];
    const value = typeof payload === 'function' ? payload(path, options) : payload;
    return {
      ok: !(value && value.__status >= 400),
      // The status is read by the client, so a fake that omits it cannot
      // exercise the paths that branch on it.
      status: (value && value.__status) || 200,
      json: async () => value,
    };
  };

  // jsdom does not implement the modal methods of <dialog>. This is a gap in
  // the tool, not in the page: the behaviour under test is which button holds
  // the focus and whether closing counts as a refusal, and both survive.
  const dialogs = window.HTMLDialogElement.prototype;
  if (!dialogs.showModal) {
    dialogs.showModal = function showModal() {
      this.open = true;
      this.setAttribute('open', '');
    };
    dialogs.close = function close() {
      this.open = false;
      this.removeAttribute('open');
      // The standard queues this event rather than firing it inline, and the
      // difference matters: dispatching it synchronously would settle a
      // promise that the click handler is about to settle itself.
      setTimeout(() => this.dispatchEvent(new window.Event('close')), 0);
    };
  }

  // Views subscribe to the event stream; nothing has to arrive for a test.
  global.EventSource = class {
    constructor() { this.onmessage = null; }
    close() {}
  };

  return { dom, window, document: window.document, calls, unmocked,
           view: window.document.getElementById('view') };
}

export function text(node) {
  return node.textContent;
}

export function buttons(node, label) {
  return [...node.querySelectorAll('button')].filter(
    (item) => item.textContent.trim() === label);
}

export function button(node, label) {
  const found = buttons(node, label);
  if (!found.length) {
    throw new Error(`no button labelled "${label}"; found: `
      + [...node.querySelectorAll('button')].map((b) => `"${b.textContent.trim()}"`).join(', '));
  }
  return found[0];
}

// Let queued promises settle, since every click handler is asynchronous.
export async function settle(times = 6) {
  for (let index = 0; index < times; index += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}
