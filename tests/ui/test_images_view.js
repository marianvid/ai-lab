import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';
import { installDom, settle } from './support/dom.js';

const PROFILES = [
  { id: 'edit', task: 'edit', model: 'image' },
  { id: 'generate', task: 'generation', model: 'image' },
];

describe('the Images page', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('asks for a source image when an edit profile is selected', async () => {
    const context = installDom({ '/api/image-profiles': PROFILES, '/api/image-jobs': [] });
    const { render } = await import(`../../ai_lab/web/js/views/images.js?${Math.random()}`);
    await render(context.view);
    assert.equal(context.view.querySelector('input[type="file"]').required, true);
    assert.equal(context.view.querySelector('button[type="submit"]').textContent, 'Queue edit');
  });

  it('removes the source-image requirement for generation', async () => {
    const context = installDom({ '/api/image-profiles': PROFILES, '/api/image-jobs': [] });
    const { render } = await import(`../../ai_lab/web/js/views/images.js?${Math.random()}`);
    await render(context.view);
    const select = context.view.querySelector('select');
    select.value = 'generate';
    select.dispatchEvent(new context.window.Event('change'));
    assert.equal(context.view.querySelector('input[type="file"]').required, false);
    assert.equal(context.view.querySelector('button[type="submit"]').textContent,
                 'Queue generation');
  });

  it('sends an edit as multipart data to the edit endpoint', async () => {
    const context = installDom({ '/api/image-profiles': PROFILES, '/api/image-jobs': [],
      'POST /v1/images/edits': { id: 'job-1' } });
    global.FormData = context.window.FormData;
    const { render } = await import(`../../ai_lab/web/js/views/images.js?${Math.random()}`);
    await render(context.view);
    const input = context.view.querySelector('input[type="file"]');
    const file = new context.window.File(['pixels'], 'source.png', { type: 'image/png' });
    Object.defineProperty(input, 'files', { value: [file] });
    context.view.querySelector('textarea').value = 'blue scarf';
    context.view.querySelector('form').dispatchEvent(
      new context.window.Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    const call = context.calls.find((item) => item.path === '/v1/images/edits');
    assert.ok(call);
    assert.equal(call.body.get('profile'), 'edit');
    assert.equal(call.body.get('prompt'), 'blue scarf');
    assert.equal(call.body.get('image').name, 'source.png');
  });

  it('loads a successful result only when View is pressed', async () => {
    const job = { id: 'job-1', profile: 'generate', status: 'succeeded' };
    const context = installDom({ '/api/image-profiles': PROFILES,
      '/api/image-jobs': [job],
      '/api/image-jobs/job-1': { ...job, result: { data: [
        { mime_type: 'image/png', b64_json: 'cGl4ZWxz' },
      ] } } });
    const { render } = await import(`../../ai_lab/web/js/views/images.js?${Math.random()}`);
    await render(context.view);
    assert.equal(context.calls.some((item) => item.path === '/api/image-jobs/job-1'), false);
    [...context.view.querySelectorAll('button')]
      .find((item) => item.textContent === 'View').click();
    await settle();
    assert.equal(context.view.querySelector('img').getAttribute('src'),
                 'data:image/png;base64,cGl4ZWxz');
    assert.ok(context.calls.some((item) => item.path === '/api/image-jobs/job-1'));
  });
});
