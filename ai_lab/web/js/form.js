// Building a settings form from what an engine declares about itself.
//
// The server sends each setting's type, range, default and help text, so this
// file never names a parameter. Adding a setting to an engine makes it appear
// here with no change to the interface — which is the whole reason the engine
// describes itself rather than the form hard-coding a list.

import { element } from './format.js';

const GROUPS = [
  {
    key: 'memory',
    title: 'Memory and capacity',
    note: 'Reserved on the accelerator when the model starts. Changing these '
          + 'means reloading the model.',
  },
  {
    key: 'generation',
    title: 'Generation defaults',
    note: 'Starting values for requests. Any client can override them per call, '
          + 'so these save you sending the same values every time.',
  },
];

function input(spec, value) {
  const current = value !== undefined ? value : spec.default;

  if (spec.kind === 'bool') {
    return element('input', {
      type: 'checkbox', 'data-key': spec.key,
      ...(current ? { checked: 'checked' } : {}),
    });
  }
  if (spec.kind === 'choice') {
    return element('select', { 'data-key': spec.key },
      spec.choices.map((choice) => element('option', {
        value: choice, text: choice,
        ...(String(current) === choice ? { selected: 'selected' } : {}),
      })));
  }
  return element('input', {
    type: 'number', 'data-key': spec.key, value: String(current), size: 10,
    step: spec.kind === 'float' ? '0.01' : '1',
    ...(spec.minimum !== null ? { min: String(spec.minimum) } : {}),
    ...(spec.maximum !== null ? { max: String(spec.maximum) } : {}),
  });
}

// The explanation lives in the tooltip, not on the page.
//
// It used to be printed under every label as well. Some of these run to five
// or six lines — they say what a setting costs, which is the useful part — and
// a form of a dozen settings became a wall of prose with the fields lost in
// it. You read one of them when you are deciding about that one setting, and
// never again.
//
// A label that has something to say is marked, so the tooltip is discoverable
// rather than a thing you find by accident.
function field(spec, value) {
  return element('label', {
    class: spec.help ? 'field explained' : 'field',
    title: spec.help || '',
  }, [
    element('span', {}, element('span', { text: spec.label })),
    input(spec, value),
  ]);
}

function group(definition, specs, values) {
  const mine = specs.filter((spec) => (spec.group || 'memory') === definition.key);
  if (!mine.length) return null;
  return element('section', { class: 'group' }, [
    element('h4', { text: definition.title }),
    element('p', { class: 'muted', text: definition.note }),
    element('div', { class: 'fields' }, mine.map((spec) => field(spec, values[spec.key]))),
  ]);
}

// Builds the whole form and hands back a node plus a way to read it.
export function settingsForm(specs, values = {}) {
  const container = element('div', { class: 'settings' },
    GROUPS.map((definition) => group(definition, specs, values)).filter(Boolean));

  // The stored values are kept underneath and overlaid with what the fields
  // say, so a setting the engine knows about but the form did not draw keeps
  // its value instead of quietly reverting to a default.
  const read = () => {
    const collected = { ...values };
    container.querySelectorAll('[data-key]').forEach((node) => {
      const spec = specs.find((item) => item.key === node.dataset.key);
      if (!spec) return;
      collected[spec.key] = spec.kind === 'bool'
        ? node.checked
        : spec.kind === 'choice'
          ? node.value
          : spec.kind === 'float' ? parseFloat(node.value) : parseInt(node.value, 10);
    });
    return collected;
  };

  return { node: container, read };
}
