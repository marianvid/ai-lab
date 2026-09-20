import { element } from '../format.js';

// A heading above its panel, the same shape for every section.
export function section(title, children) {
  return element('div', { class: 'section' }, [
    element('h3', { text: title }),
    element('div', { class: 'card' }, children),
  ]);
}
