import { api } from '../api.js';
import { element } from '../format.js';
import { readBase64 } from './file-encoding.js';
import { runMediaJob, showMediaHistory } from './media-jobs.js';

export function renderMusic(target, model, options = {}) {
  const bucket = options?.duration_kind === 'bucket';
  const hasDuration = options?.duration_kind !== 'none';
  const editableScore = Boolean(options?.editable_score);
  const referenceRequired = Boolean(options?.reference_audio_required);
  const prompt = element('textarea', { rows: '4', required: 'required',
    placeholder: 'Describe the song, mood and instruments',
    'aria-label': 'Music description' });
  const lyrics = element('textarea', { rows: '5',
    ...(options?.lyrics_required ? { required: 'required' } : {}),
    placeholder: options?.lyrics_required ? 'Lyrics required for this model'
      : 'Optional lyrics; leave empty for instrumental music',
    'aria-label': 'Lyrics' });
  const duration = element('input', { type: 'number', min: bucket ? '0' : '5',
    max: bucket ? String(options.maximum_bucket) : '180',
    value: bucket ? String(options.default_bucket) : '30',
    'aria-label': bucket ? 'Length bucket' : 'Duration in seconds' });
  const seed = element('input', { type: 'number', min: '0', max: '4294967295',
    placeholder: 'Random', 'aria-label': 'Seed' });
  const reference = element('input', { type: 'file', accept: '.wav,audio/wav',
    required: 'required', 'aria-label': 'Reference WAV' });
  const score = element('textarea', { rows: '12',
    placeholder: 'Optional ABC score; edit a generated score and run again',
    'aria-label': 'Editable ABC score' });
  const status = element('p', { role: 'status', class: 'muted' });
  const result = element('section', { class: 'panel stack', hidden: true });
  const submit = element('button', { type: 'submit', text: 'Generate music' });
  const background = element('button', { type: 'submit', text: 'Generate in background' });
  const cancel = element('button', { type: 'button', text: 'Cancel job', hidden: true });
  const history = element('section', { class: 'panel stack' });
  const display = (response) => {
    const audio = response.data?.[0]?.b64_wav;
    if (!audio) throw new Error('The model returned no audio');
    const src = `data:audio/wav;base64,${audio}`;
    result.hidden = false;
    result.replaceChildren(
      element('h2', { text: 'Generated audio' }),
      element('audio', { controls: 'controls', src }),
      element('a', { href: src, download: `${model}-${response.seed}.wav`,
        text: 'Download WAV' }));
    if (editableScore && response.score_abc) {
      score.value = response.score_abc;
      result.append(element('a', {
        href: `data:text/plain;charset=utf-8,${encodeURIComponent(response.score_abc)}`,
        download: `${model}-${response.seed}.abc`, text: 'Download ABC score' }));
    }
    status.textContent = response.truncated ? 'The model reached its generation limit; review the ending.' : '';
  };
  let refreshHistory = () => {};
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    submit.disabled = true;
    background.disabled = true;
    status.textContent = 'Generating music…';
    try {
      const request = {
        prompt: prompt.value.trim(), lyrics: lyrics.value.trim() || '[Instrumental]',
        instrumental: !lyrics.value.trim(),
      };
      if (hasDuration) request[bucket ? 'length_bucket' : 'duration'] = Number(duration.value);
      if (editableScore && score.value.trim()) request.abc = score.value.trim();
      if (referenceRequired) {
        request.reference_audio_base64 = await readBase64(
          reference.files?.[0], 25 * 1024 * 1024, 'reference WAV');
      }
      if (seed.value !== '') request.seed = Number(seed.value);
      const response = event.submitter === background
        ? await runMediaJob(model, 'music-generation', request, status, cancel)
        : await api.music(model, request);
      display(response);
      if (event.submitter === background) refreshHistory();
    } catch (error) { status.textContent = error.message; }
    finally { submit.disabled = false; background.disabled = false; }
  }}, [prompt, lyrics,
    ...(hasDuration ? [element('label', {}, [element('span', { text: bucket
      ? 'Length bucket (0 is shortest; actual seconds vary)' : 'Duration (seconds)' }), duration])] : []),
    ...(referenceRequired ? [element('label', {}, [
      element('span', { text: 'Reference WAV (up to 25 MiB)' }), reference])] : []),
    ...(editableScore ? [element('label', {}, [
      element('span', { text: 'ABC score (optional; edit and regenerate)' }), score])] : []),
    element('label', {}, [element('span', { text: 'Seed' }), seed]),
    submit, background, cancel, status]);
  target.replaceChildren(element('h1', { text: `Music · ${model}` }), form, result, history);
  refreshHistory = showMediaHistory(history, model, 'music-generation', display, status);
}
