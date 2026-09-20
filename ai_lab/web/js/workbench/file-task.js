import { api } from '../api.js';
import { element } from '../format.js';

const TASKS = {
  transcription: { title: 'Transcribe audio', path: '/v1/audio/transcriptions',
    accept: 'audio/*' },
  vad: { title: 'Detect speech', path: '/v1/audio/speech-segments',
    accept: 'audio/*' },
  diarization: { title: 'Identify speakers', path: '/v1/audio/diarizations',
    accept: 'audio/*' },
  alignment: { title: 'Align transcript with audio', path: '/v1/audio/alignments',
    accept: 'audio/*' },
  ocr: { title: 'Read text from image', path: '/v1/images/ocr',
    accept: 'image/png,image/jpeg,image/gif,image/webp' },
};

export function renderFileTask(target, model, task) {
  const config = TASKS[task];
  if (!config) throw new Error(`No file workflow for ${task}`);
  const file = element('input', { type: 'file', accept: config.accept,
    required: 'required', 'aria-label': 'Source file' });
  const language = element('input', { type: 'text', placeholder: 'English, French, Spanish…',
    'aria-label': 'Language' });
  const transcript = element('textarea', { rows: '6', required: 'required',
    'aria-label': 'Transcript', placeholder: 'Paste the exact words in the audio' });
  const status = element('p', { class: 'muted', role: 'status' });
  const result = element('pre', { class: 'panel', 'aria-live': 'polite' });
  const submit = element('button', { type: 'submit', text: 'Run' });
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    if (!file.files?.[0]) return;
    submit.disabled = true;
    status.textContent = 'Processing…';
    result.textContent = '';
    try {
      const fields = (task === 'transcription' || task === 'ocr')
        ? { language: language.value.trim() } : {};
      if (task === 'alignment') {
        fields.text = transcript.value.trim();
        fields.language = language.value.trim() || 'English';
      }
      const output = await api.analyzeFile(config.path, model, file.files[0], fields);
      result.textContent = output.text || JSON.stringify(output, null, 2);
      status.textContent = 'Complete';
    } catch (error) { status.textContent = error.message; }
    finally { submit.disabled = false; }
  }}, [file, ...(task === 'alignment' ? [transcript] : []),
       ...(task === 'transcription' || task === 'ocr' || task === 'alignment'
         ? [language] : []),
       submit, status]);
  target.replaceChildren(element('h1', { text: `${config.title} · ${model}` }),
                         form, result);
}
