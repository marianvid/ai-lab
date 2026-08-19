// The line at the bottom of the page.
//
// Its own module rather than part of the entry point, because every view needs
// it and the entry point runs code as soon as it is imported — drawing tabs,
// opening the event stream. A view that imported it from there dragged all of
// that along, which made the views impossible to test and the imports circular.

let node = null;

export function attachStatus(element) {
  node = element;
}

export function setStatus(message, kind = 'muted') {
  if (!node) return;
  node.className = kind;
  node.textContent = message || '';
}

// Exposed so a test can assert on what the user was told.
export function currentStatus() {
  return node ? { text: node.textContent, kind: node.className } : null;
}
