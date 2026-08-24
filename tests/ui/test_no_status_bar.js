// The page does not narrate itself.
//
// There used to be a line at the foot of every page saying what had just
// happened: "Saving gemma31-nvfp4 done". It was below the fold, it had no
// border and no background, the next action wiped it, and it said things the
// page had already shown — a saved value sitting in its field, a filled bar, a
// row that had changed state.
//
// What replaced it: a failure takes the page with a modal, work in progress
// shows on the button that was pressed, and a result shows where the result
// is. These tests hold that line, because a status bar is the easy thing to
// reach for and it creeps back one message at a time.

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { describe, it } from 'node:test';

const VIEWS = 'ai_lab/web/js';

function everyScript(directory = VIEWS) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) return everyScript(path);
    return entry.name.endsWith('.js') ? [path] : [];
  });
}

describe('the page does not narrate itself', () => {
  it('has no status module for anything to import', () => {
    assert.equal(everyScript().some((path) => path.endsWith('/status.js')), false,
                 'the footer line is back');
  });

  it('has nothing left calling for one', () => {
    const guilty = everyScript().filter((path) =>
      /setStatus|attachStatus|status\.js/.test(readFileSync(path, 'utf8')));
    assert.deepEqual(guilty, [], 'these still want a status line');
  });

  it('leaves no footer in the page for one to be attached to', () => {
    const page = readFileSync('ai_lab/web/index.html', 'utf8');
    assert.equal(/<footer/.test(page), false);
    assert.equal(/id="status"/.test(page), false);
  });

  it('gives anything with an explanation the question-mark cursor', () => {
    // An I-beam over a figure invites dragging across it, which is not what is
    // on offer. And a tooltip nobody knows is there may as well not be
    // written, so the label carries a faint mark too — the same one a setting
    // with an explanation behind it already had.
    const css = readFileSync('ai_lab/web/css/app.css', 'utf8');
    assert.match(css, /\[title\][^{]*\{[^}]*cursor:\s*help/s,
                 'nothing gives a tooltip its own cursor');
    assert.match(css, /underline dotted/,
                 'a tooltip with no mark on the label is one nobody finds');
  });

  it('still has the two things that replaced it', () => {
    // Deleting the bar without these would lose real information: a failure
    // nobody sees, and a button that looks like it did nothing.
    const scripts = everyScript();
    assert.ok(scripts.includes('ai_lab/web/js/confirm.js'), 'no modal to fail into');
    assert.ok(scripts.includes('ai_lab/web/js/working.js'),
              'no way for a button to show it is working');
  });
});
