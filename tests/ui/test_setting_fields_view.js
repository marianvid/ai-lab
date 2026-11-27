// The settings that once lived only in the configuration file, on the Settings page.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, button, settle } from './support/dom.js';

const LOAD = {
  key: 'limits.runtime.load_s', group: 'loading', label: 'Model load', unit: 's',
  default: 900, low: 1, high: 604800, kind: 'number', restart: true,
  help: 'How long the engine may take to answer once started.',
  restart_note: 'Takes effect when AI-Lab is next started.', value: 900, saved: false,
};
const PATH = {
  key: 'limits.builds.system_path', group: 'updates', label: 'Search path for installs',
  unit: '', default: '/usr/bin:/bin', low: null, high: null, kind: 'text', restart: true,
  help: 'Where the programs are looked for.', restart_note: 'Takes effect when AI-Lab is next started.',
  value: '/usr/bin:/bin', saved: false,
};
const PORT = {
  key: 'limits.first_port', group: 'loading', label: 'First port offered', unit: '',
  default: 8080, low: 1, high: 65535, kind: 'integer', restart: false,
  help: 'The first port suggested.', restart_note: '', value: 8080, saved: false,
};
const ON = {
  key: 'gpu_health.enabled', group: 'gpu_alarms', label: 'Monitor on', unit: '',
  default: true, low: null, high: null, kind: 'boolean', restart: true,
  help: 'Whether the card is watched.', restart_note: 'Takes effect when AI-Lab is next started.',
  value: true, saved: false,
};
const XIDS = {
  key: 'gpu_health.xid_critical', group: 'gpu_alarms', label: 'Critical Xid numbers', unit: '',
  default: [48, 63], low: null, high: null, kind: 'list', restart: true,
  help: 'The serious ones.', restart_note: 'Takes effect when AI-Lab is next started.',
  value: [48, 63], saved: false,
};
const FIELDS = {
  groups: { loading: 'Loading models', updates: 'Engine updates', gpu_alarms: 'GPU health: alarms' },
  fields: [LOAD, PORT, PATH, ON, XIDS],
};

async function renderPage(overrides = {}) {
  const context = installDom({
    '/api/settings': { title: 'AI-Lab', host: {}, accelerator: {}, repositories: [],
                       engines: [], models_root: '/models' },
    '/api/setting-fields': FIELDS,
    'PATCH /api/setting-fields': FIELDS,
    ...overrides,
  });
  const { render } = await import(`../../ai_lab/web/js/views/settings.js?${Math.random()}`);
  await render(context.view);
  await settle();
  return context;
}

function namedSection(view, name) {
  const heading = [...view.querySelectorAll('.section > h3')]
    .find((item) => item.textContent.trim() === name);
  return heading && heading.parentElement;
}

describe('the settings from the configuration file', () => {
  it('shows each group as a section of its own, with every field in it', async () => {
    const { view } = await renderPage();
    const loading = namedSection(view, 'Loading models');
    assert.ok(loading, 'the Loading models section');
    assert.match(loading.textContent, /Model load \(s\)/);
    assert.match(loading.textContent, /First port offered/);
    assert.ok(namedSection(view, 'Engine updates').querySelector('input[type=text]'));
  });

  it('explains a field when pointed at: meaning, default, and when it takes effect', async () => {
    const { view } = await renderPage();
    const field = namedSection(view, 'Loading models').querySelector('label.field');
    assert.match(field.title, /How long the engine may take/);
    assert.match(field.title, /Default: 900 s\./);
    assert.match(field.title, /next started/);
    assert.match(field.textContent, /after a restart/);
  });

  it('says nothing about a restart for a field that takes effect at once', async () => {
    const { view } = await renderPage();
    const fields = [...namedSection(view, 'Loading models').querySelectorAll('label.field')];
    const port = fields.find((item) => item.textContent.includes('First port'));
    assert.doesNotMatch(port.textContent, /after a restart/);
  });

  it('saves only what was changed, and only once something was', async () => {
    const { view, calls, window } = await renderPage();
    const loading = namedSection(view, 'Loading models');
    const save = button(loading, 'Save');
    assert.equal(save.disabled, true);
    const box = loading.querySelector('input[type=number]');
    box.value = '1800';
    box.dispatchEvent(new window.Event('input'));
    assert.equal(save.disabled, false);
    save.click();
    await settle();
    const sent = calls.find((call) => call.method === 'PATCH');
    assert.deepEqual(JSON.parse(sent.body), { 'limits.runtime.load_s': 1800 });
  });

  it('keeps what was typed when the server refuses it', async () => {
    const { view, window } = await renderPage({
      'PATCH /api/setting-fields': { __status: 400, error: 'Model load must be at least 1' },
    });
    const loading = namedSection(view, 'Loading models');
    const box = loading.querySelector('input[type=number]');
    box.value = '0';
    box.dispatchEvent(new window.Event('input'));
    button(loading, 'Save').click();
    await settle();
    assert.equal(namedSection(view, 'Loading models').querySelector('input[type=number]').value, '0');
  });

  it('says so when the fields could not be read, rather than leaving them out', async () => {
    const { view } = await renderPage({ '/api/setting-fields': { __status: 500, error: 'down' } });
    assert.equal(namedSection(view, 'Loading models'), undefined);
    assert.match(view.textContent, /could not be read/);
  });

  it('shows on/off as a box and a list as numbers, and sends them as such', async () => {
    const { view, calls, window } = await renderPage();
    const alarms = namedSection(view, 'GPU health: alarms');
    const on = alarms.querySelector('input[type=checkbox]');
    const list = alarms.querySelector('input[type=text]');
    assert.equal(list.value, '48, 63');
    assert.equal(on.checked, true);
    assert.match(alarms.querySelector('label.field').title, /Default: on\./);
    on.checked = false;
    on.dispatchEvent(new window.Event('change'));
    list.value = '48, 63, 79';
    list.dispatchEvent(new window.Event('input'));
    button(alarms, 'Save').click();
    await settle();
    const sent = calls.find((call) => call.method === 'PATCH');
    assert.deepEqual(JSON.parse(sent.body),
                     { 'gpu_health.enabled': false, 'gpu_health.xid_critical': [48, 63, 79] });
  });
});
