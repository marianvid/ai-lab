// What the browser tab says.
//
// There is usually more than one of these open: one against the Linux box and
// one against the Mac. Two tabs both reading "AI-Lab" are two tabs you have to
// click to tell apart, and the wrong one is the one where a model gets
// unloaded.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { pageTitle } from '../../ai_lab/web/js/format.js';

describe('the browser tab', () => {
  it('says which machine this is', () => {
    assert.equal(pageTitle({ title: 'AI-Lab', host: { operating_system: 'Linux' } }),
                 'AI-Lab · Linux');
    assert.equal(pageTitle({ title: 'AI-Lab', host: { operating_system: 'macOS' } }),
                 'AI-Lab · macOS');
  });

  it('tells two machines apart', () => {
    // The whole point, stated as the thing being avoided.
    const linux = pageTitle({ title: 'AI-Lab', host: { operating_system: 'Linux' } });
    const mac = pageTitle({ title: 'AI-Lab', host: { operating_system: 'macOS' } });
    assert.notEqual(linux, mac);
  });

  it('keeps whatever the installation is called', () => {
    // The name is configuration. Somebody who renamed it did so on purpose.
    assert.equal(pageTitle({ title: 'Private inference lab', host: { operating_system: 'Linux' } }),
                 'Private inference lab · Linux');
  });

  it('says just the name when the machine will not say what it is', () => {
    // Rather than "AI-Lab · undefined", or a dash meaning something else.
    assert.equal(pageTitle({ title: 'AI-Lab', host: {} }), 'AI-Lab');
    assert.equal(pageTitle({ title: 'AI-Lab' }), 'AI-Lab');
  });

  it('has something to say before the settings have arrived', () => {
    // Called on a first draw where the fetch has not returned.
    assert.equal(pageTitle(undefined), 'AI-Lab');
    assert.equal(pageTitle({}), 'AI-Lab');
  });
});
