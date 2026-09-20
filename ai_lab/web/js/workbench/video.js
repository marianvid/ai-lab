import { api } from '../api.js';
import { element } from '../format.js';
import { readBase64 } from './file-encoding.js';
import { runMediaJob, showMediaHistory } from './media-jobs.js';

export function renderVideo(target, model, options = {}) {
  const prompt = element('textarea', { rows: '5', required: 'required',
    placeholder: 'Describe motion, camera and sound',
    'aria-label': 'Video prompt' });
  const image = element('input', { type: 'file', accept: '.png,image/png',
    required: 'required', 'aria-label': 'Reference PNG' });
  const seed = element('input', { type: 'number', min: '0', max: '4294967295',
    placeholder: 'Random', 'aria-label': 'Seed' });
  const status = element('p', { role: 'status', class: 'muted' });
  const result = element('section', { class: 'panel stack', hidden: true });
  const submit = element('button', { type: 'submit', text: 'Generate video' });
  const background = element('button', { type: 'submit', text: 'Generate in background' });
  const cancel = element('button', { type: 'button', text: 'Cancel job', hidden: true });
  const history = element('section', { class: 'panel stack' });
  const display = (response) => {
    const mp4 = response.data?.[0]?.b64_mp4;
    if (!mp4) throw new Error('The model returned no video');
    const src = `data:video/mp4;base64,${mp4}`;
    result.hidden = false;
    result.replaceChildren(element('h2', { text: 'Generated video' }),
      element('video', { controls: 'controls', src }),
      element('a', { href: src, download: `${model}-${response.seed}.mp4`,
        text: 'Download MP4' }));
    status.textContent = '';
  };
  let refreshHistory = () => {};
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    submit.disabled = true;
    background.disabled = true;
    status.textContent = 'Generating video…';
    try {
      const request = { prompt: prompt.value.trim(),
        image_base64: await readBase64(image.files?.[0], 25 * 1024 * 1024,
          'reference PNG') };
      if (seed.value !== '') request.seed = Number(seed.value);
      const response = event.submitter === background
        ? await runMediaJob(model, 'video-generation', request, status, cancel)
        : await api.video(model, request);
      display(response);
      if (event.submitter === background) refreshHistory();
    } catch (error) { status.textContent = error.message; }
    finally { submit.disabled = false; background.disabled = false; }
  }}, [prompt, element('label', {}, [
    element('span', { text: 'Reference PNG (up to 25 MiB)' }), image]),
    element('label', {}, [element('span', { text: 'Seed' }), seed]),
    ...(options.clip_seconds ? [element('p', { class: 'muted',
      text: `Configured clip length: about ${options.clip_seconds} seconds` })] : []),
    submit, background, cancel, status]);
  target.replaceChildren(element('h1', { text: `Video · ${model}` }), form, result, history);
  refreshHistory = showMediaHistory(history, model, 'video-generation', display, status);
}
