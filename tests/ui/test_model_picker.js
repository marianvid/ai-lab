import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { installDom } from './support/dom.js';
import { modelPicker } from '../../ai_lab/web/js/model-picker.js';

describe('searching models', () => {
  it('keeps the selected model while narrowing the list', () => {
    const { window } = installDom();
    const picker = modelPicker([
      { id: 'qwen-image', label: 'Qwen image · image-generation' },
      { id: 'flux', label: 'Flux · image-generation' },
      { id: 'qwen-text', label: 'Qwen text · text-generation' },
    ], 'qwen-text');
    document.body.append(picker.node);
    assert.equal(picker.select.value, 'qwen-text');
    picker.search.value = 'qwen text';
    picker.search.dispatchEvent(new window.Event('input'));
    assert.deepEqual([...picker.select.options].map((option) => option.value),
                     ['qwen-text']);
    assert.equal(picker.select.value, 'qwen-text');
  });

  it('reports no match and chooses the first result after a new search', () => {
    const { window } = installDom();
    const picker = modelPicker([
      { id: 'qwen', label: 'Qwen' }, { id: 'flux', label: 'Flux' },
    ]);
    document.body.append(picker.node);
    picker.search.value = 'missing';
    picker.search.dispatchEvent(new window.Event('input'));
    assert.equal(picker.select.disabled, true);
    assert.equal(picker.node.querySelector('small').hidden, false);
    picker.search.value = 'flux';
    picker.search.dispatchEvent(new window.Event('input'));
    assert.equal(picker.select.disabled, false);
    assert.equal(picker.select.value, 'flux');
  });
});
