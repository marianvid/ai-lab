// The new model form and its engine specific settings.
import { api } from '../api.js';
import { settingsForm } from '../form.js';
import { bytes, element } from '../format.js';
import { modelPicker } from '../model-picker.js';
import { taskOf, engineCanUse } from './runtime_model.js';

export function addCard(form, { run, close }) {
  const usable = form.models.filter((model) =>
    form.engines.some((engine) => engine.available
      && engineCanUse(engine, model)));

  if (!usable.length) {
    return element('div', { class: 'card' }, [
      element('h3', { text: 'Add a model' }),
      element('p', { class: 'muted',
                     text: 'No usable models on disk. Download one first.' }),
      element('button', { class: 'action', text: 'Cancel',
                          onclick: close }),
    ]);
  }

  const picker = modelPicker(usable.map((model) => ({
    id: model.id,
    label: `${model.name} · ${taskOf(model)} · ${model.format} · ${bytes(model.size_bytes)}`,
  })));
  const chooser = picker.select;
  const name = element('input', {
    placeholder: 'e.g. gemma-31b-nvfp4', size: 28,
    pattern: '[a-z0-9][a-z0-9-]*',
  });
  const port = element('input', { type: 'number', value: String(form.port), size: 8 });

  const engineFor = (modelId) => {
    const model = usable.find((item) => item.id === modelId);
    return form.engines.find((engine) =>
      engine.available && engineCanUse(engine, model));
  };

  const specsFor = (modelId) => {
    const model = usable.find((item) => item.id === modelId);
    const engine = engineFor(modelId);
    return engine.task_params?.[taskOf(model)] || engine.params;
  };

  let settings = settingsForm(specsFor(chooser.value), {});
  const holder = element('div', {}, settings.node);
  chooser.addEventListener('change', () => {
    settings = settingsForm(specsFor(chooser.value), {});
    holder.replaceChildren(settings.node);
  });

  return element('div', { class: 'card' }, [
    element('h3', { text: 'Add a model' }),
    element('label', { class: 'field' }, [element('span', { text: 'Model' }), picker.node]),
    // The rules where the name is typed, not in a message after it is refused.
    // This is the only name the entry has: a request carries it and a person
    // reads it, so it cannot hold spaces and it cannot be changed later.
    element('label', { class: 'field' }, [
      element('span', {}, [
        element('span', { text: 'Name' }),
        element('small', { class: 'muted',
                           text: 'lower-case letters, digits and hyphens · '
                                 + 'sent as "model" in a request · cannot be '
                                 + 'changed later' }),
      ]),
      name,
    ]),
    element('label', { class: 'field' }, [
      element('span', {}, [
        element('span', { text: 'Port' }),
        element('small', { class: 'muted', text: 'first free one, change if a client expects another' }),
      ]),
      port,
    ]),
    holder,
    element('div', { class: 'row' }, [
      element('span', {}),
      element('div', {}, [
        element('button', { class: 'action', text: 'Cancel',
                            onclick: close }),
        element('button', {
          class: 'action', text: 'Add',
          onclick: async () => {
            if (chooser.disabled) return;
            const chosen = name.value.trim();
            await run(`Adding ${chosen || 'a model'}`, () => api.createInstance({
              id: chosen,
              engine: engineFor(chooser.value).id,
              model_id: chooser.value,
              port: parseInt(port.value, 10),
              params: settings.read(),
            }));
            close();
          },
        }),
      ]),
    ]),
  ]);
}
