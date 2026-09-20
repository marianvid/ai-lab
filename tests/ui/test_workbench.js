import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { installDom, settle } from './support/dom.js';
import { renderChat } from '../../ai_lab/web/js/workbench/chat.js';
import { renderMusic } from '../../ai_lab/web/js/workbench/music.js';
import { renderFileTask } from '../../ai_lab/web/js/workbench/file-task.js';

describe('direct model use', () => {
  it('sends chat through the AI-Lab gateway and keeps the conversation', async () => {
    const context = installDom({
      'POST /v1/chat/completions': { choices: [
        { message: { content: 'Hello back' } },
      ] },
    });
    renderChat(context.view, 'coder-fast');
    const prompt = context.view.querySelector('textarea[aria-label="Message"]');
    prompt.value = 'Hello';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path === '/v1/chat/completions');
    assert.deepEqual(JSON.parse(call.body), {
      model: 'coder-fast', stream: false,
      messages: [{ role: 'user', content: 'Hello' }],
    });
    assert.match(context.view.textContent, /Hello back/);
    prompt.value = 'Again';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const second = JSON.parse(context.calls.at(-1).body);
    assert.equal(second.messages.length, 3);
  });

  it('generates playable music through the gateway', async () => {
    const context = installDom({
      'POST /v1/audio/music/generations': {
        seed: 42, data: [{ b64_wav: 'UklGRg==' }],
      },
    });
    renderMusic(context.view, 'music-xl');
    context.view.querySelector('textarea[aria-label="Music description"]').value =
      'A short ambient cue';
    context.view.querySelector('input[aria-label="Seed"]').value = '42';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path ===
      '/v1/audio/music/generations');
    assert.deepEqual(JSON.parse(call.body), {
      model: 'music-xl', prompt: 'A short ambient cue',
      lyrics: '[Instrumental]', instrumental: true, duration: 30, seed: 42,
    });
    assert.match(context.view.querySelector('audio').getAttribute('src'),
      /^data:audio\/wav;base64,/);
  });

  it('sends an audio upload with the configured model name', async () => {
    const context = installDom({ 'POST /v1/audio/transcriptions': { text: 'Test' } });
    global.FormData = context.window.FormData;
    renderFileTask(context.view, 'canary', 'transcription');
    const file = context.view.querySelector('input[type="file"]');
    Object.defineProperty(file, 'files', { value: [new context.window.File(
      ['audio'], 'sample.wav', { type: 'audio/wav' })] });
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path === '/v1/audio/transcriptions');
    assert.equal(call.body.get('model'), 'canary');
    assert.equal(call.body.get('file').name, 'sample.wav');
    assert.match(context.view.textContent, /Test/);
  });
});
