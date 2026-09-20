import { api } from '../api.js';
import { element } from '../format.js';

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
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    submit.disabled = true;
    status.textContent = 'Generating music…';
    try {
      const request = {
        prompt: prompt.value.trim(), lyrics: lyrics.value.trim() || '[Instrumental]',
        instrumental: !lyrics.value.trim(),
      };
      if (hasDuration) request[bucket ? 'length_bucket' : 'duration'] = Number(duration.value);
      if (editableScore && score.value.trim()) request.abc = score.value.trim();
      if (referenceRequired) {
        const file = reference.files?.[0];
        if (!file) throw new Error('Select a reference WAV');
        if (file.size > 25 * 1024 * 1024) throw new Error('Reference WAV exceeds 25 MiB');
        const bytes = new Uint8Array(await file.arrayBuffer());
        const chunks = [];
        for (let offset = 0; offset < bytes.length; offset += 32768) {
          chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 32768)));
        }
        request.reference_audio_base64 = btoa(chunks.join(''));
      }
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
      if (editableScore && response.score_abc) {
        score.value = response.score_abc;
        result.append(element('a', {
          href: `data:text/plain;charset=utf-8,${encodeURIComponent(response.score_abc)}`,
          download: `${model}-${response.seed}.abc`, text: 'Download ABC score' }));
      }
      status.textContent = response.truncated ? 'The model reached its generation limit; review the ending.' : '';
    } catch (error) { status.textContent = error.message; }
    finally { submit.disabled = false; }
  }}, [prompt, lyrics,
    ...(hasDuration ? [element('label', {}, [element('span', { text: bucket
      ? 'Length bucket (0 is shortest; actual seconds vary)' : 'Duration (seconds)' }), duration])] : []),
    ...(referenceRequired ? [element('label', {}, [
      element('span', { text: 'Reference WAV (up to 25 MiB)' }), reference])] : []),
    ...(editableScore ? [element('label', {}, [
      element('span', { text: 'ABC score (optional; edit and regenerate)' }), score])] : []),
    element('label', {}, [element('span', { text: 'Seed' }), seed]),
    submit, status]);
  target.replaceChildren(element('h1', { text: `Music · ${model}` }), form, result);
}
