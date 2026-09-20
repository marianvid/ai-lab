import assert from 'node:assert/strict';
import { it } from 'node:test';
import { installDom } from './support/dom.js';

it('keeps a future engine option as text when the user saves it', async () => {
  installDom({});
  const { settingsForm } = await import('../../ai_lab/web/js/form.js');
  const form = settingsForm([{
    key: 'tool_parser', label: 'Tool calling', kind: 'identifier',
    default: '', minimum: null, maximum: null, choices: [],
    group: 'memory', help: '',
  }], {});
  const field = form.node.querySelector('[data-key="tool_parser"]');
  assert.equal(field.type, 'text');
  field.value = 'future_parser_2026';
  assert.equal(form.read().tool_parser, 'future_parser_2026');
  assert.equal(form.changed(), true);
});
