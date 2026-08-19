// The event stream.
//
// Three kinds of event share one connection, each carrying its own `kind`:
// progress through a model load, lines of output from a build, and a notice
// that something changed and is worth fetching again.
//
// The last one is why the pages do not poll. Asking every few seconds redrew
// them whether or not anything had happened, which threw away whatever was
// half-typed or selected at the time — and the quieter the machine, the more
// pointless the redraw.

const listeners = { runtime: new Set(), log: new Set(), change: new Set() };

function subscribe(kind, callback) {
  listeners[kind].add(callback);
  return () => listeners[kind].delete(callback);
}

export const onProgress = (callback) => subscribe('runtime', callback);
export const onLog = (callback) => subscribe('log', callback);
export const onChange = (callback) => subscribe('change', callback);

export function startEventStream() {
  const source = new EventSource('/api/events');
  source.onmessage = (message) => {
    let event;
    try { event = JSON.parse(message.data); } catch { return; }
    (listeners[event.kind] || []).forEach((callback) => callback(event));
  };
  return source;
}
