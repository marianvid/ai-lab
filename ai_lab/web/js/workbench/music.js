import { api } from '../api.js';
import { element } from '../format.js';

export function renderMusic(target, model, options = {}) {
  const bucket = options?.duration_kind === 'bucket';
  const prompt = element('textarea', { rows: '4', required: 'required',
    placeholder: 'Describe the song, mood and instruments',
    'aria-label': 'Music description' });
  const lyrics = element('textarea', { rows: '5',
    placeholder: 'Optional lyrics; leave empty for instrumental music',
    'aria-label': 'Lyrics' });
  const duration = element('input', { type: 'number', min: bucket ? '0' : '5',
    max: bucket ? String(options.maximum_bucket) : '180',
    value: bucket ? String(options.default_bucket) : '30',
    'aria-label': bucket ? 'Length bucket' : 'Duration in seconds' });
  const seed = element('input', { type: 'number', min: '0', max: '4294967295',
    placeholder: 'Random', 'aria-label': 'Seed' });
  const status = element('p', { role: 'status', class: 'muted' });
  const result = element('section', { class: 'panel stack', hidden: true });
  const submit = element('button', { type: 'submit', text: 'Generate music' });
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    submit.disabled = true;
    status.textContent = 'Generating music…';
    try {
      const request = {
        prompt: prompt.value.trim(), lyrics: lyrics.value.trim() || '[Instrumental]',
        instrumental: !lyrics.value.trim(),
      };
      request[bucket ? 'length_bucket' : 'duration'] = Number(duration.value);
      if (seed.value !== '') request.seed = Number(seed.value);
      const response = await api.music(model, request);
      const audio = response.data?.[0]?.b64_wav;
      if (!audio) throw new Error('The model returned no audio');
      const src = `data:audio/wav;base64,${audio}`;
      result.hidden = false;
      result.replaceChildren(
        element('h2', { text: 'Generated audio' }),
        element('audio', { controls: 'controls', src }),
        element('a', { href: src, download: `${model}-${response.seed}.wav`,
          text: 'Download WAV' }));
      status.textContent = '';
    } catch (error) { status.textContent = error.message; }
    finally { submit.disabled = false; }
  }}, [prompt, lyrics,
    element('label', {}, [element('span', { text: bucket
      ? 'Length bucket (0 is shortest; actual seconds vary)' : 'Duration (seconds)' }), duration]),
    element('label', {}, [element('span', { text: 'Seed' }), seed]),
    submit, status]);
  target.replaceChildren(element('h1', { text: `Music · ${model}` }), form, result);
}
