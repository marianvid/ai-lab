// Settings kept in the configuration file, one section per group.
//
// The owner's rule (2026-11-27): no setting lives only in a file. Each field
// says, when pointed at, what it means, its unit and default, and what too
// small or too large does; one that takes effect only at the next start says
// so beside it. The fields, their explanations and their checks come from the
// server (`application/setting_fields.py`), so the page cannot drift from what
// AI-Lab actually reads.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { element } from '../format.js';
import { section } from './settings_section.js';

// A value as the page writes it: a list as "48, 63", on/off as words.
function shown(field, value) {
  if (field.kind === 'list') return (value || []).join(', ');
  if (field.kind === 'boolean') return value ? 'on' : 'off';
  return String(value);
}

// What a field says when pointed at: its meaning, then its default.
function explanation(field) {
  const unit = field.unit ? ` ${field.unit}` : '';
  return `${field.help}\n\nDefault: ${shown(field, field.default) || '(empty)'}${unit}.`
    + (field.restart_note ? `\n${field.restart_note}` : '');
}

function input(field) {
  if (field.kind === 'boolean') {
    return element('input', { type: 'checkbox', ...(field.value ? { checked: 'checked' } : {}) });
  }
  const text = field.kind === 'text' || field.kind === 'list';
  return element('input', {
    type: text ? 'text' : 'number', value: shown(field, field.value),
    ...(text ? { size: 28 } : { size: 8, step: field.kind === 'integer' ? '1' : 'any' }),
    ...(field.low !== null && !text ? { min: String(field.low) } : {}),
    ...(field.high !== null && !text ? { max: String(field.high) } : {}),
  });
}

function label(field) {
  const name = field.unit ? `${field.label} (${field.unit})` : field.label;
  return element('span', {}, [
    element('span', { text: name }),
    field.restart_note
      ? element('span', { class: 'muted', text: ' · after a restart' }) : null,
  ].filter(Boolean));
}

// One group's fields and a Save that wakes only when something differs.
export function settingFieldsSection(view, group, refresh) {
  const fields = (view?.fields || []).filter((field) => field.group === group);
  if (!fields.length) return null;
  const inputs = new Map();
  const rows = fields.map((field) => {
    const box = input(field);
    inputs.set(field.key, { box, field });
    return element('label', { class: 'field explained', title: explanation(field) },
                   [label(field), box]);
  });
  const valueOf = ({ box, field }) => {
    if (field.kind === 'boolean') return box.checked;
    if (field.kind === 'text') return box.value;
    // A list is sent as numbers; anything that is not one is sent as typed,
    // so the server refuses it by name rather than the page dropping it.
    if (field.kind === 'list') {
      return box.value.split(',').map((part) => part.trim()).filter(Boolean)
        .map((part) => (/^\d+$/.test(part) ? Number(part) : part));
    }
    return Number(box.value);
  };
  const changes = () => {
    const now = {};
    inputs.forEach((entry, key) => {
      const value = valueOf(entry);
      if (shown(entry.field, value) !== shown(entry.field, entry.field.value)) now[key] = value;
    });
    return now;
  };
  const save = element('button', {
    class: 'action', text: 'Save', disabled: 'disabled',
    onclick: async () => {
      save.disabled = true;
      try {
        await api.updateSettingFields(changes());
      } catch (error) {
        // What was typed stays, so it can be corrected rather than typed again.
        await showNotice({ title: `Could not save ${view.groups[group].toLowerCase()}`,
                           body: error.message });
        save.disabled = !Object.keys(changes()).length;
        return;
      }
      refresh();
    },
  });
  inputs.forEach(({ box }) => {
    const wake = () => { save.disabled = !Object.keys(changes()).length; };
    box.addEventListener('input', wake);
    box.addEventListener('change', wake);
  });
  return section(view.groups[group], [
    ...rows, element('div', { class: 'row' }, [element('span', {}), save]),
  ]);
}
