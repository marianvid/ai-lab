import { api } from '../api.js';
import { element } from '../format.js';
import { runMediaJob, showMediaHistory } from './media-jobs.js';

export function renderSpeech(target, model, options = {}) {
  const formOptions = options || {};
  const text = element('textarea', { rows: '5', required: 'required',
    'aria-label': 'Text to speak', placeholder: 'Write the text to speak' });
  const instruction = element('textarea', { rows: '3',
    ...(formOptions.instruction_required ? { required: 'required' } : {}),
    'aria-label': 'Voice instruction',
    placeholder: 'Voice description (required for voice design), or optional delivery instruction' });
  const language = element('input', { value: 'Auto',
    'aria-label': 'Language', placeholder: 'Auto, English, Chinese…' });
  const speaker = element('input', {
    placeholder: formOptions.speaker_hint || 'Optional speaker or voice',
    'aria-label': 'Speaker' });
  const status = element('p', { class: 'muted', role: 'status' });
  const result = element('section', { class: 'panel stack', hidden: true });
  const submit = element('button', { type: 'submit', text: 'Generate speech' });
  const background = element('button', { type: 'submit', text: 'Generate in background' });
  const cancel = element('button', { type: 'button', text: 'Cancel job', hidden: true });
  const history = element('section', { class: 'panel stack' });
  const display = (response) => {
    const audio = response.data?.[0]?.b64_wav;
    if (!audio) throw new Error('The model returned no audio');
    const src = `data:audio/wav;base64,${audio}`;
    result.hidden = false;
    result.replaceChildren(element('h2', { text: 'Generated speech' }),
      element('audio', { controls: 'controls', src }),
      element('a', { href: src, download: `${model}.wav`, text: 'Download WAV' }));
    status.textContent = '';
  };
  let refreshHistory = () => {};
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    submit.disabled = true;
    background.disabled = true;
    status.textContent = 'Generating speech…';
    try {
      const request = {
        text: text.value.trim(), instruction: instruction.value.trim(),
        language: language.value.trim() || 'Auto', speaker: speaker.value.trim(),
      };
      const response = event.submitter === background
        ? await runMediaJob(model, 'speech-synthesis', request, status, cancel)
        : await api.speech(model, request);
      display(response);
      if (event.submitter === background) refreshHistory();
    } catch (error) { status.textContent = error.message; }
    finally { submit.disabled = false; background.disabled = false; }
  }}, [text,
    element('label', { hidden: formOptions.instruction_visible === false ? '' : undefined }, [
      element('span', { text: 'Voice instruction' }), instruction]),
    element('label', { hidden: formOptions.language_visible === false ? '' : undefined }, [
      element('span', { text: 'Language' }), language]),
    element('label', { hidden: formOptions.speaker_visible === false ? '' : undefined }, [
      element('span', { text: formOptions.speaker_label || 'Speaker / voice' }),
      speaker]),
    submit, background, cancel, status]);
  target.replaceChildren(element('h1', { text: `Speech · ${model}` }), form, result, history);
  refreshHistory = showMediaHistory(history, model, 'speech-synthesis', display, status);
}
