// A real DOM for the tests.
//
// jsdom rather than a hand-written stand-in, deliberately. The faults that
// reached the user came from how a browser actually behaves — replaceChildren
// turning a null into the text "null", a rebuilt input losing what was being
// typed — and a stand-in written by the same hand that wrote the bugs would
// encode the same assumptions and miss them again.

import { JSDOM } from 'jsdom';

const PAGE = `<!doctype html><html><body>
  <header><div id="host-summary"></div></header>
  <nav id="tabs"></nav>
  <main id="view"></main>
  <footer id="status"></footer>
</body></html>`;

// Every request the interface makes, and what to answer with. A view under
// test never reaches the network.
export function installDom(responses = {}) {
  const dom = new JSDOM(PAGE, { url: 'http://localhost:8090/' });
  const { window } = dom;

  global.window = window;
  global.document = window.document;
  global.CSS = window.CSS;
  global.HTMLElement = window.HTMLElement;
  global.confirm = () => true;

  const calls = [];
  global.fetch = async (path, options = {}) => {
    calls.push({ path, method: options.method || 'GET', body: options.body });
    const key = `${options.method || 'GET'} ${path.split('?')[0]}`;
    const payload = key in responses ? responses[key]
                  : path.split('?')[0] in responses ? responses[path.split('?')[0]]
                  : [];
    const value = typeof payload === 'function' ? payload(path, options) : payload;
    return {
      ok: !(value && value.__status >= 400),
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

  return { dom, window, document: window.document, calls, view: window.document.getElementById('view') };
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
