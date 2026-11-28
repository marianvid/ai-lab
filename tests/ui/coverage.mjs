// The interface's line coverage, with every copy of a module counted as one.
//
// The tests import a page as `views/x.js?<random>` to get a fresh copy of its
// state for each test, and Node's own coverage report lists every copy as a
// file of its own: a page every test draws whole read 31 %. This runs the
// tests with V8's raw coverage (NODE_V8_COVERAGE), drops the query from each
// URL, and counts a line as reached when any copy reached it. A line is one
// that holds code (not blank, not only a comment); it counts as reached when
// the code at its first character ran.
//
//   node tests/ui/coverage.mjs        (npm run coverage)
//   node tests/ui/coverage.mjs views/settings_engines.js   (also its unreached lines)

import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const SOURCE = join(ROOT, 'ai_lab', 'web', 'js');
const raw = mkdtempSync(join(tmpdir(), 'ai-lab-ui-coverage-'));

try {
  const tests = readdirSync(join(ROOT, 'tests', 'ui')).filter((name) => /^test_.*\.js$/.test(name))
    .map((name) => join('tests', 'ui', name));
  execFileSync(process.execPath, ['--test', ...tests], {
    cwd: ROOT, stdio: ['ignore', 'ignore', 'inherit'], env: { ...process.env, NODE_V8_COVERAGE: raw },
  });
  const reached = new Map();                       // file -> Set of reached line numbers
  for (const name of readdirSync(raw)) {
    for (const script of JSON.parse(readFileSync(join(raw, name), 'utf8')).result) {
      if (!script.url.startsWith('file://')) continue;
      const file = fileURLToPath(script.url.split('?')[0]);
      if (!file.startsWith(SOURCE)) continue;
      const text = readFileSync(file, 'utf8');
      const counts = new Int32Array(text.length);
      // Outer ranges come first and inner ones refine them, so applying them in order is enough.
      for (const fn of script.functions) {
        for (const range of fn.ranges) counts.fill(range.count, range.startOffset, range.endOffset);
      }
      const lines = reached.get(file) || new Set();
      let offset = 0;
      text.split('\n').forEach((line, index) => {
        const first = line.search(/\S/);
        if (first >= 0 && counts[offset + first] > 0) lines.add(index + 1);
        offset += line.length + 1;
      });
      reached.set(file, lines);
    }
  }
  // A module no test reaches at all is counted too, at none.
  const walk = (folder) => readdirSync(folder, { withFileTypes: true }).flatMap((entry) => (
    entry.isDirectory() ? walk(join(folder, entry.name))
      : entry.name.endsWith('.js') ? [join(folder, entry.name)] : []));
  for (const file of walk(SOURCE)) if (!reached.has(file)) reached.set(file, new Set());
  let total = 0;
  let hit = 0;
  const rows = [];
  for (const [file, lines] of [...reached].sort()) {
    const code = readFileSync(file, 'utf8').split('\n')
      .map((line, index) => [line.trim(), index + 1])
      .filter(([line]) => line && !line.startsWith('//') && !line.startsWith('*') && !line.startsWith('/*'));
    const got = code.filter(([, number]) => lines.has(number)).length;
    total += code.length;
    hit += got;
    rows.push([relative(ROOT, file), (100 * got / code.length).toFixed(1), code.length]);
  }
  for (const [file, share, count] of rows) console.log(`${share.padStart(6)} %  ${String(count).padStart(4)} lines  ${file}`);
  const asked = process.argv[2];
  if (asked) {
    const file = join(SOURCE, asked);
    const lines = reached.get(file) || new Set();
    const missed = readFileSync(file, 'utf8').split('\n')
      .map((line, index) => [line.trim(), index + 1])
      .filter(([line, number]) => line && !line.startsWith('//') && !line.startsWith('*')
              && !line.startsWith('/*') && !lines.has(number))
      .map(([, number]) => number);
    console.log(`not reached in ${asked}: ${missed.join(' ')}`);
  }
  console.log(`${(100 * hit / total).toFixed(1).padStart(6)} %  ${String(total).padStart(4)} lines  all of ai_lab/web/js`);
} finally {
  rmSync(raw, { recursive: true, force: true });
}
