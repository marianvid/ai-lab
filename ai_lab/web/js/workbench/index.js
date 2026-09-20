import { api } from '../api.js';
import { element } from '../format.js';
import { onChange, startEventStream } from '../events.js';
import { render as renderImages } from '../views/images.js';
import { renderChat } from './chat.js';
import { renderMusic } from './music.js';
import { renderSpeech } from './speech.js';
import { renderFileTask } from './file-task.js';
import { TASK_ACTIONS } from './tasks.js';

const model = new URLSearchParams(window.location.search).get('model');
const target = document.getElementById('view');
let imageEventsStarted = false;

function watchImageJobs() {
  if (imageEventsStarted) return;
  imageEventsStarted = true;
  let pending = null;
  onChange((event) => {
    if (event.topic !== 'image-jobs' || pending) return;
    pending = setTimeout(async function refresh() {
      if (target.contains(document.activeElement)
          && document.activeElement?.closest('form')) {
        pending = setTimeout(refresh, 1000);
        return;
      }
      pending = null;
      try { await renderImages(target, model); }
      catch (error) { target.replaceChildren(element('p', {
        class: 'error', text: error.message })); }
    }, 150);
  });
  startEventStream();
}

export async function renderWorkbench() {
  if (!model) throw new Error('Choose a model from the Models page');
  const [instances, models] = await Promise.all([api.instances(), api.models()]);
  const instance = instances.find((item) => item.id === model);
  if (!instance) throw new Error(`Unknown configured model: ${model}`);
  const stored = models.find((item) => item.id === instance.model_id);
  const task = instance.task || stored?.task || 'text-generation';
  document.title = `${model} · AI-Lab`;
  document.getElementById('model-name').textContent = model;
  const surface = TASK_ACTIONS[task]?.surface;
  if (surface === 'chat') renderChat(target, model);
  else if (surface === 'music') renderMusic(target, model);
  else if (surface === 'speech') renderSpeech(target, model, instance.speech_form);
  else if (surface === 'images') {
    await renderImages(target, model);
    watchImageJobs();
  } else if (surface === 'file') {
    renderFileTask(target, model, task);
  } else throw new Error(`Direct use of ${task} is not configured yet`);
}

renderWorkbench().catch((error) => {
  target.replaceChildren(element('p', { class: 'error', text: error.message }));
});
