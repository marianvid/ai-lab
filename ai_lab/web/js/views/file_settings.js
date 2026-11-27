// The settings kept only in config.json, shown read-only on the page they
// belong to.
//
// The owner (2026-11-27): "Hai sa fie read only acolo unde nu avem
// certitudine. macar sa stiu despre ele." Each section says what it is, what
// it is for, where it is changed (the file and the key) and whether a
// restart is needed; its value is shown as saved, a secret only as "(set)".
// The sections and their explanations come from the server
// (`application/file_settings.py`).

import { api } from '../api.js';
import { element } from '../format.js';
import { section } from './settings_section.js';

// One line per value, its place written as a path: `profiles.flux.model: …`.
function lines(value, path = '') {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const entries = Object.entries(value);
    if (!entries.length) return [[path || '(none)', path ? '{}' : '']];
    return entries.flatMap(([key, item]) => lines(item, path ? `${path}.${key}` : key));
  }
  if (Array.isArray(value) && value.some((item) => item && typeof item === 'object')) {
    if (!value.length) return [[path, '[]']];
    return value.flatMap((item, index) => lines(item, `${path}[${index}]`));
  }
  return [[path, Array.isArray(value) ? value.join(', ') : String(value)]];
}

function shown(item) {
  const rows = lines(item.value).map(([path, text]) => element('div', { class: 'row tight' }, [
    element('span', { class: 'muted', text: path }),
    element('span', { text }),
  ]));
  return section(`${item.title} (read-only)`, [
    element('p', { class: 'muted', text: `${item.help} ${item.restart}` }),
    ...(rows.length ? rows : [element('p', { class: 'muted', text: 'Nothing set.' })]),
  ]);
}

// The read-only sections of `page`, drawn; an empty list when there are none.
// A failed read says so rather than leaving the sections out without a word.
export async function fileSettingsSections(page) {
  let view;
  try {
    view = await api.fileSettings(page);
  } catch (error) {
    return [element('p', { class: 'error',
      text: `The settings kept only in config.json could not be read: ${error.message}` })];
  }
  return (view.sections || []).map(shown);
}
