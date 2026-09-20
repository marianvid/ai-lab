import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { installDom, settle } from './support/dom.js';
import { renderChat } from '../../ai_lab/web/js/workbench/chat.js';
import { renderMusic } from '../../ai_lab/web/js/workbench/music.js';
import { renderVideo } from '../../ai_lab/web/js/workbench/video.js';
import { renderSpeech } from '../../ai_lab/web/js/workbench/speech.js';
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

  it('shows Khala length buckets instead of seconds', async () => {
    const context = installDom({
      'POST /v1/audio/music/generations': {
        seed: 42, data: [{ b64_wav: 'UklGRg==' }],
      },
    });
    renderMusic(context.view, 'new-music-checkpoint', {
      duration_kind: 'bucket', default_bucket: 0, maximum_bucket: 4,
    });
    context.view.querySelector('textarea[aria-label="Music description"]').value =
      'Warm piano';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path ===
      '/v1/audio/music/generations');
    const body = JSON.parse(call.body);
    assert.equal(body.length_bucket, 0);
    assert.equal('duration' in body, false);
    assert.match(context.view.textContent, /Length bucket/);
  });

  it('requires lyrics when the music engine declares them mandatory', () => {
    const context = installDom();
    renderMusic(context.view, 'song-checkpoint', { lyrics_required: true });
    const lyrics = context.view.querySelector('textarea[aria-label="Lyrics"]');
    assert.equal(lyrics.required, true);
    assert.match(lyrics.placeholder, /required/);
  });

  it('edits a YuE2 score and does not send an unsupported duration', async () => {
    const context = installDom({
      'POST /v1/audio/music/generations': {
        seed: 42, score_abc: 'X:1\nT:New song\n',
        data: [{ b64_wav: 'UklGRg==' }],
      },
    });
    renderMusic(context.view, 'song-checkpoint', {
      lyrics_required: true, duration_kind: 'none', editable_score: true,
    });
    context.view.querySelector('textarea[aria-label="Music description"]').value =
      'Soul piano';
    context.view.querySelector('textarea[aria-label="Lyrics"]').value =
      '[Verse] Hello';
    const score = context.view.querySelector('textarea[aria-label="Editable ABC score"]');
    score.value = 'X:1\nT:My edit\n';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path ===
      '/v1/audio/music/generations');
    const body = JSON.parse(call.body);
    assert.equal(body.abc, 'X:1\nT:My edit');
    assert.equal('duration' in body, false);
    assert.equal(score.value, 'X:1\nT:New song\n');
    assert.match(context.view.textContent, /Download ABC score/);
  });

  it('sends a reference WAV for a source-conditioned music engine', async () => {
    const context = installDom({
      'POST /v1/audio/music/generations': {
        seed: 42, data: [{ b64_wav: 'UklGRg==' }],
      },
    });
    renderMusic(context.view, 'cover-model', {
      reference_audio_required: true, lyrics_required: true, duration_kind: 'none',
    });
    context.view.querySelector('textarea[aria-label="Music description"]').value =
      'genre:[soul]';
    context.view.querySelector('textarea[aria-label="Lyrics"]').value =
      '[Verse] Hello';
    const input = context.view.querySelector('input[aria-label="Reference WAV"]');
    Object.defineProperty(input, 'files', { value: [{ size: 8,
      arrayBuffer: async () => Uint8Array.from([82, 73, 70, 70, 0, 0, 0, 0]).buffer,
    }] });
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path ===
      '/v1/audio/music/generations');
    const body = JSON.parse(call.body);
    assert.equal(body.reference_audio_base64, 'UklGRgAAAAA=');
    assert.equal(body.instrumental, false);
    assert.equal('duration' in body, false);
    assert.equal(input.required, true);
  });

  it('generates a playable video from a reference PNG', async () => {
    const context = installDom({
      'POST /v1/videos/generations': {
        seed: 42, data: [{ b64_mp4: 'AAAAGGZ0eXA=' }],
      },
    });
    renderVideo(context.view, 'video-model', { clip_seconds: 6.58 });
    context.view.querySelector('textarea[aria-label="Video prompt"]').value =
      'Gentle motion in a locked shot';
    const input = context.view.querySelector('input[aria-label="Reference PNG"]');
    Object.defineProperty(input, 'files', { value: [{ size: 8,
      arrayBuffer: async () => Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]).buffer,
    }] });
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path === '/v1/videos/generations');
    const body = JSON.parse(call.body);
    assert.equal(body.model, 'video-model');
    assert.equal(body.prompt, 'Gentle motion in a locked shot');
    assert.equal(body.image_base64, 'iVBORw0KGgo=');
    assert.match(context.view.querySelector('video').getAttribute('src'),
      /^data:video\/mp4;base64,/);
    assert.match(context.view.textContent, /Download MP4/);
  });

  it('synthesizes speech and shows a WAV player', async () => {
    const context = installDom({
      'POST /v1/audio/speech/generations': {
        data: [{ b64_wav: 'UklGRg==' }],
      },
    });
    renderSpeech(context.view, 'qwen-voice');
    context.view.querySelector('textarea[aria-label="Text to speak"]').value =
      'Hello there';
    context.view.querySelector('textarea[aria-label="Voice instruction"]').value =
      'Warm voice';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path ===
      '/v1/audio/speech/generations');
    assert.equal(JSON.parse(call.body).model, 'qwen-voice');
    assert.equal(JSON.parse(call.body).text, 'Hello there');
    assert.match(context.view.querySelector('audio').getAttribute('src'),
      /^data:audio\/wav;base64,/);
  });

  it('uses configured speech controls without inspecting the model name', async () => {
    const context = installDom({
      'POST /v1/audio/speech/generations': {
        data: [{ b64_wav: 'UklGRg==' }],
      },
    });
    renderSpeech(context.view, 'any-new-checkpoint', {
      instruction_visible: false, language_visible: false,
      speaker_visible: true, speaker_label: 'Voice',
      speaker_hint: 'Default: bm_george',
    });
    const instruction = context.view.querySelector(
      'textarea[aria-label="Voice instruction"]');
    const speaker = context.view.querySelector('input[aria-label="Speaker"]');
    assert.equal(instruction.closest('label').hidden, true);
    assert.equal(speaker.closest('label').hidden, false);
    assert.equal(speaker.placeholder, 'Default: bm_george');
    speaker.value = 'af_heart';
    context.view.querySelector('textarea[aria-label="Text to speak"]').value =
      'Hello there';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path ===
      '/v1/audio/speech/generations');
    assert.equal(JSON.parse(call.body).speaker, 'af_heart');
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

  it('sends a transcript and audio file for alignment', async () => {
    const context = installDom({ 'POST /v1/audio/alignments': {
      words: [{ text: 'Hello', start: 0.1, end: 0.5 }],
    } });
    global.FormData = context.window.FormData;
    renderFileTask(context.view, 'aligner', 'alignment');
    const file = context.view.querySelector('input[type="file"]');
    Object.defineProperty(file, 'files', { value: [new context.window.File(
      ['audio'], 'sample.wav', { type: 'audio/wav' })] });
    context.view.querySelector('textarea[aria-label="Transcript"]').value = 'Hello';
    context.view.querySelector('form').dispatchEvent(new context.window.Event(
      'submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path === '/v1/audio/alignments');
    assert.equal(call.body.get('model'), 'aligner');
    assert.equal(call.body.get('text'), 'Hello');
    assert.equal(call.body.get('language'), 'English');
    assert.match(context.view.textContent, /0.5/);
  });
});
