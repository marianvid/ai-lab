import { api } from '../api.js';
import { element } from '../format.js';

export function renderSpeech(target, model) {
  const text = element('textarea', { rows: '5', required: 'required',
    'aria-label': 'Text to speak', placeholder: 'Write the text to speak' });
  const instruction = element('textarea', { rows: '3',
    'aria-label': 'Voice instruction',
    placeholder: 'Voice description (required for voice design), or optional delivery instruction' });
  const language = element('input', { value: 'Auto',
    'aria-label': 'Language', placeholder: 'Auto, English, Chinese…' });
  const speaker = element('input', { placeholder: 'Ryan, Vivian…',
    'aria-label': 'Speaker' });
  const status = element('p', { class: 'muted', role: 'status' });
  const result = element('section', { class: 'panel stack', hidden: true });
  const submit = element('button', { type: 'submit', text: 'Generate speech' });
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    submit.disabled = true;
    status.textContent = 'Generating speech…';
    try {
      const response = await api.speech(model, {
        text: text.value.trim(), instruction: instruction.value.trim(),
        language: language.value.trim() || 'Auto', speaker: speaker.value.trim(),
      });
      const audio = response.data?.[0]?.b64_wav;
      if (!audio) throw new Error('The model returned no audio');
      const src = `data:audio/wav;base64,${audio}`;
      result.hidden = false;
      result.replaceChildren(element('h2', { text: 'Generated speech' }),
        element('audio', { controls: 'controls', src }),
        element('a', { href: src, download: `${model}.wav`, text: 'Download WAV' }));
      status.textContent = '';
    } catch (error) { status.textContent = error.message; }
    finally { submit.disabled = false; }
  }}, [text, instruction,
    element('label', {}, [element('span', { text: 'Language' }), language]),
    element('label', {}, [
      element('span', { text: 'Speaker (CustomVoice)' }), speaker]),
    submit, status]);
  target.replaceChildren(element('h1', { text: `Speech · ${model}` }), form, result);
}
