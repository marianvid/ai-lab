// A searchable model selector shared by adding and editing an entry.

import { element } from './format.js';

export function modelPicker(models, selected = null, selectAttributes = {}) {
  const search = element('input', {
    type: 'search', placeholder: 'Search models',
    'aria-label': 'Search models', autocomplete: 'off',
  });
  const select = element('select', selectAttributes);
  const empty = element('small', { class: 'muted', text: 'No matching models' });
  empty.hidden = true;

  function show(query) {
    const words = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const matching = models.filter((model) => {
      const haystack = `${model.label} ${model.id}`.toLocaleLowerCase();
      return words.every((word) => haystack.includes(word));
    });
    const previous = select.value || selected;
    select.replaceChildren(...matching.map((model) => element('option', {
      value: model.id, text: model.label,
    })));
    if (matching.some((model) => model.id === previous)) select.value = previous;
    select.disabled = matching.length === 0;
    empty.hidden = matching.length !== 0;
    if (matching.length && select.value !== previous) {
      select.dispatchEvent(new select.ownerDocument.defaultView.Event('change'));
    }
  }

  search.addEventListener('input', () => show(search.value));
  show('');
  return { node: element('div', { class: 'model-picker' }, [search, select, empty]),
           search, select };
}
