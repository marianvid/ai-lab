// One browser contract for task labels and their direct-use surfaces.
export const TASK_ACTIONS = Object.freeze({
  'text-generation': { label: 'Chat', surface: 'chat' },
  'image-generation': { label: 'Create', surface: 'images' },
  'image-edit': { label: 'Edit', surface: 'images' },
  'music-generation': { label: 'Music', surface: 'music' },
  'video-generation': { label: 'Video', surface: 'video' },
  'speech-synthesis': { label: 'Speak', surface: 'speech' },
  alignment: { label: 'Align', surface: 'file' },
  transcription: { label: 'Transcribe', surface: 'file' },
  vad: { label: 'Speech', surface: 'file' },
  diarization: { label: 'Speakers', surface: 'file' },
  ocr: { label: 'Read', surface: 'file' },
});
