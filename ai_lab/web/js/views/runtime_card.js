// One configured model row, its settings and direct-use action.
import { api } from '../api.js';
import { capabilities, suppressed } from '../icons.js';
import { whileWorking } from '../working.js';
import { settingsForm } from '../form.js';
import { bytes, element, seconds } from '../format.js';
import { toggleLogs, watching } from '../logpane.js';
import { modelPicker } from '../model-picker.js';
import { taskOf, engineCanUse } from './runtime_model.js';

export function createCard({ progress, open, paint, paintFromState, run, removeInstance, redraw }) {
  // -- one row ----------------------------------------------------------------

  // Everything that used to be printed under the name. It is still worth having
  // — it is just not worth three lines of screen for every entry, every time.
  // Hovering asks for it; the line no longer insists on it.
  function details(instance, models, latest) {
    const model = models.find((item) => item.id === instance.model_id);
    const settings = instance.params || {};
    const lines = [
      model ? `${model.name} · ${taskOf(model)} · ${model.format} · ${bytes(model.size_bytes)}`
            : instance.model_id,
      [`port ${instance.port}`,
       settings.context_size ? `ctx ${settings.context_size}` : null,
       settings.parallel ? `${settings.parallel} slot${settings.parallel === 1 ? '' : 's'}` : null,
       // MLX LM has no slots: it produces several answers in the same step.
       settings.decode_concurrency ? `${settings.decode_concurrency} at once` : null,
       settings.temperature !== undefined ? `temp ${settings.temperature}` : null,
      ].filter(Boolean).join(' · '),
      instance.ready ? 'ready' : instance.running ? 'starting' : 'stopped',
      // A request can ask for the model started differently from how the entry
      // is configured — a bigger context, usually — and the settings are not
      // saved. Without this the row shows a number the running model is not
      // using, and nothing says so.
      runningDifferently(instance),
    ];
    if (latest && instance.running && latest.memory_used_mb) {
      lines.push(`this model is holding ${Math.round(latest.memory_used_mb)} MB`);
    }
    const operation = instance.last_operation;
    if (operation && operation.steps.length) {
      lines.push(operation.steps
        .map((step) => `${step.phase} ${seconds(step.elapsed_ms)}`)
        .concat(`total ${seconds(operation.total_ms)}`)
        .join(' · '));
    }
    return lines.filter(Boolean).join('\n');
  }

  // What it is actually running with, when that is not what it says above.
  function runningDifferently(instance) {
    const active = instance.active_params || {};
    const keys = Object.keys(active);
    if (!keys.length) return null;
    return 'started by a request with '
      + keys.map((key) => `${key} ${active[key]}`).join(', ')
      + ' — not saved, so it returns to the settings above when reloaded';
  }

  // Which model this entry runs. Smaller and quieter than the name, because it
  // is what the entry points at rather than what the entry is called.
  function modelName(instance, models) {
    const model = models.find((item) => item.id === instance.model_id);
    return model ? model.name : instance.model_id;
  }

  function modelSummary(instance, models) {
    const model = models.find((item) => item.id === instance.model_id);
    if (!model?.summary) return null;
    return element('span', {
      class: 'model-summary', text: model.summary,
      title: model.description || model.summary,
    });
  }

  // How long the last load took, on the row rather than in a line at the foot of
  // the page. A load runs from four seconds to a minute, so you look away and
  // come back — and a footer message is gone by then, wiped by whatever was
  // pressed next. Here it stays with the entry it belongs to. The breakdown by
  // phase is in the tooltip; this is the one number worth seeing without asking.
  function lastLoad(instance) {
    const operation = instance.last_operation;
    if (!operation || operation.total_ms === undefined || operation.ok === false) {
      return null;
    }
    return element('span', { class: 'muted took',
                             text: seconds(operation.total_ms),
                             title: 'How long the last load or unload took' });
  }

  // What this row will actually be able to do: what the weights can do, less
  // whatever its own settings switch off.
  function canDo(instance, models) {
    const model = models.find((item) => item.id === instance.model_id);
    if (!model) return [];
    return capabilities(model.capabilities,
                        suppressed(instance.params, instance.withheld));
  }

  function formatOf(instance, models) {
    const model = models.find((item) => item.id === instance.model_id);
    return model ? model.format : '';
  }

  function card(instance, models, engines) {
    const engine = engines.find((item) => item.id === instance.engine);
    const specs = engine
      ? (engine.task_params?.[instance.task || 'text-generation'] || engine.params)
      : [];
    const expanded = open.has(instance.id);
    const form = expanded ? settingsForm(specs, instance.params || {}) : null;

    // The name is not editable. It is what requests carry, so changing it would
    // break whatever is already sending it — and renaming is the same act as
    // deleting the entry and adding one with the new name, which the page
    // already offers.
    const edits = () => ({
      model_id: chooser.value,
      params: form.read(),
    });

    const latest = progress.get(instance.id);
    if (!instance.running) progress.delete(instance.id);
    const bar = element('div', { class: 'bar thin', 'data-bar': instance.id },
                        element('span', {}));
    if (instance.running && latest) paint(bar, latest); else paintFromState(bar, instance);

    const picker = modelPicker(models
      .filter((item) => !engine || engineCanUse(engine, item))
      .map((item) => ({ id: item.id,
        label: `${item.name} · ${bytes(item.size_bytes)}` })),
      instance.model_id, { 'data-model': instance.id });
    const chooser = picker.select;

    // Settings are configuration, and configuration can be written down whether
    // or not anything is running. Without this the only ways to save a setting
    // were to start the model or to restart it — so unticking a box on a stopped
    // entry did nothing at all, silently, and the entry kept the old value.
    const save = element('button', {
      class: 'action', text: 'Save',
      title: expanded ? 'Write these settings down without touching the card'
                      : 'Open Settings to change something first',
      disabled: 'disabled',
      onclick: () => whileWorking(save, 'Saving…', () =>
        run(`Saving ${instance.id}`, () => api.update(instance.id, edits()))),
    });
    if (form) form.watch(() => { save.disabled = !form.changed(); });

    // Always visible so the row does not change shape, but only usable when it
    // would do something: applying settings means restarting the model.
    const apply = element('button', {
      class: 'action', text: 'Apply & reload',
      title: instance.running
        ? (expanded ? 'Save these settings and restart the model with them'
                    : 'Open Settings to change something first')
        : 'Nothing is running — use Load',
      ...(instance.running && expanded ? {} : { disabled: 'disabled' }),
      onclick: () => whileWorking(apply, 'Reloading…', () =>
        run(`Reloading ${instance.id}`, (force) =>
          api.apply(instance.id, edits(), force))),
    });

    const primary = instance.running
      ? element('button', {
          class: 'action', text: 'Unload',
          onclick: () => whileWorking(primary, 'Unloading…', () =>
            run(`Unloading ${instance.id}`,
                (force) => api.unload(instance.id, force))),
        })
      : element('button', {
          class: 'action', text: 'Load',
          onclick: () => whileWorking(primary, 'Loading…', () =>
            run(`Loading ${instance.id}`, async (force) => {
              if (expanded) await api.update(instance.id, edits());
              return api.load(instance.id, force);
            })),
        });

    const remove = element('button', {
      class: 'action danger', text: 'Remove',
      title: instance.running
        ? 'Unload this entry before removing it'
        : 'Remove this entry; downloaded files stay in Library',
      ...(instance.running ? { disabled: 'disabled' } : {}),
      onclick: () => whileWorking(remove, 'Removing…',
                                  () => removeInstance(instance)),
    });

    // Only while it is running. A stopped model has a journal on Linux and no
    // log at all on macOS, so offering it would work on one machine and not the
    // other — and why a model *would not start* is answered by the sentence a
    // failed load already carries.
    const logsButton = element('button', {
      class: 'action', text: watching() === instance.id ? 'Hide log' : 'Log',
      title: instance.running
        ? 'What this engine is printing about itself'
        : 'Only while the model is running',
      ...(instance.running ? {} : { disabled: 'disabled' }),
      onclick: () => toggleLogs(instance),
    });

    const settingsButton = element('button', {
      class: 'action', text: expanded ? 'Hide settings' : 'Settings',
      onclick: () => {
        if (expanded) open.delete(instance.id); else open.add(instance.id);
        redraw();
      },
    });

    // One line per entry: what it is called, which engine runs it, which model.
    // Which engine matters enough to be on the line rather than in the tooltip —
    // llama.cpp and vLLM behave differently enough that reading a row without
    // knowing which one it is tells you very little.
    //
    // Everything else — port, context, slots, temperature, size, state, and how
    // long the last load took — moved into the tooltip. It was three lines of
    // screen per entry for numbers that are only wanted occasionally.
    //
    // Removing the configured entry lives here; Library deletes the weight
    // files. Two actions, two pages, so it is always clear which kind of data
    // is about to disappear.
    //
    // The Remove button is not optional. Deleting a model in Library is refused
    // while an entry points at it — otherwise the entry could never start
    // again — and that refusal sends the reader here. Take this away and the
    // pair becomes a dead end: an entry that cannot be removed and a model
    // that cannot be deleted.
    const head = element('div', {
      class: 'row instance',
      title: details(instance, models, latest),
    }, [
      // Left: what you named it, and what it runs. This is what you read down the
      // page to find a row.
      // The name, and the model behind it. One name, which a request carries and
      // a person reads — there used to be a label as well, and it became a
      // second way of saying the same thing that had to be kept in step with the
      // first.
      element('div', { class: 'ident' }, [
        element('div', { class: 'inline model-heading' }, [
          element('strong', { text: instance.id,
                              title: 'The name to send as "model" in a request' }),
          element('span', { class: 'muted model', text: modelName(instance, models) }),
          lastLoad(instance),
        ].filter(Boolean)),
        modelSummary(instance, models),
      ].filter(Boolean)),
      // Right: what will actually run it, then the things you press. Format and
      // engine are a pair — nvfp4 on vLLM, gguf on llama.cpp — so they stay
      // together, at the end, out of the middle of the name.
      element('div', { class: 'inline' }, [
        useLink(instance),
        ...canDo(instance, models),
        formatOf(instance, models)
          ? element('span', { class: 'pill format', text: formatOf(instance, models) })
          : null,
        element('span', { class: 'pill engine',
                          text: engine ? engine.name : instance.engine }),
        settingsButton, logsButton, primary, save, apply, remove,
      ].filter(Boolean)),
    ]);

    return element('div', { class: 'card instance' }, [
      head,
      expanded
        ? element('div', { class: 'settings-open' }, [
            element('label', { class: 'field' }, [
              element('span', { text: 'Model' }), picker.node,
            ]),
            form.node,
          ])
        : null,
      bar,
    ].filter(Boolean));
  }

  // Direct use is the engine's own interface, never a reduced AI-Lab form.
  function useLink(instance) {
    const task = instance.task || 'text-generation';
    const comfy = ['comfyui', 'comfy_music', 'comfy_video'].includes(instance.engine);
    const ace = instance.engine === 'acestep' && task === 'music-generation';
    const qwenTts = instance.engine === 'qwentts' && task === 'speech-synthesis';
    // Both Higgs engines open the same upstream playground: the SGLang worker
    // on Linux, and the in-process transformers port (higgs_local) on a Mac.
    const higgs = ['higgs', 'higgs_local'].includes(instance.engine)
      && task === 'speech-synthesis';
    const voxcpm = instance.engine === 'voxcpm' && task === 'speech-synthesis';
    const kokoro = instance.engine === 'kokoro' && task === 'speech-synthesis';
    const khala = instance.engine === 'khala' && task === 'music-generation';
    const yue2 = instance.engine === 'yue2' && task === 'music-generation';
    const heartmula = instance.engine === 'heartmula' && task === 'music-generation';
    const llama = instance.engine === 'llamacpp' && task === 'text-generation';
    if (!comfy && !ace && !qwenTts && !higgs && !voxcpm && !kokoro && !khala && !yue2 && !heartmula && !llama) return null;
    const label = comfy ? ({ 'image-generation': 'Create', 'image-edit': 'Edit',
      'music-generation': 'Music', 'video-generation': 'Video' }[task] || 'ComfyUI')
      : ace || khala || yue2 || heartmula ? 'Music' : qwenTts || higgs || voxcpm || kokoro ? 'Speak' : 'Chat';
    if (!instance.ready) return element('button', {
      class: 'pill chat-link', type: 'button', disabled: 'disabled',
      title: instance.running ? 'Wait for the model to finish loading'
                              : 'Load the model to use it directly',
      text: label,
    });
    return element('a', {
      class: 'pill chat-link', target: '_blank', rel: 'noopener',
      href: `${window.location.protocol}//${window.location.hostname}:`
        + `${instance.port + (comfy || ace || qwenTts || higgs || voxcpm || kokoro || khala || yue2 || heartmula ? 10000 : 0)}/`
        + (comfy ? '?ai_lab_preset=1' : ''),
      title: comfy ? 'Open this model in ComfyUI with its configured workflow'
        : ace ? 'Open the official ACE-Step editor using this loaded model'
        : qwenTts ? 'Open the official Qwen3-TTS editor using this loaded model'
        : higgs ? 'Open the official Higgs Audio playground using this loaded model'
        : voxcpm ? 'Open the official VoxCPM editor using this loaded model'
        : kokoro ? 'Open the official Kokoro editor using this loaded model'
        : khala ? 'Open Khala Studio using this loaded model'
        : yue2 ? 'Open the full YuE2 Studio using this loaded model'
        : heartmula ? 'Open the full HeartMuse Studio using this loaded model'
        : 'Open the chat page served by llama.cpp',
      text: label,
    });
  }


  return card;
}
