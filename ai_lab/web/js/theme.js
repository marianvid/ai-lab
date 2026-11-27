// Light or dark.

const KEY = 'ai-lab-theme';
const ORDER = ['light', 'dark'];
// The sun and the moon, which need no translating and no reading. Shown for
// the theme you are *in*, with the tooltip saying what pressing it does —
// otherwise a single symbol has to answer two questions at once and half the
// people read it the other way round.
const ICON = { light: '\u2600', dark: '\u263E' };
const LABEL = { light: 'Light', dark: 'Dark' };

function systemPrefersDark() {
  return window.matchMedia
    && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

// Where storage is refused (a private window, blocked site data) this throws;
// the page must still start, only without remembering (audit L65).
let chosenHere = null;

function read() {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return chosenHere;
  }
}

function keep(choice) {
  chosenHere = choice;
  try {
    localStorage.setItem(KEY, choice);
  } catch {
    // Not remembered after a reload; the page goes on.
  }
}

function stored() {
  const value = read();
  if (ORDER.includes(value)) return value;
  // Nothing chosen yet: start from what the system is already doing, so the
  // first click changes the page rather than confirming it.
  return systemPrefersDark() ? 'dark' : 'light';
}

function apply(choice) {
  document.documentElement.setAttribute('data-theme', choice);
}

export function currentTheme() {
  return stored();
}

export function installTheme(button) {
  const draw = () => {
    const now = stored();
    const next = now === 'dark' ? 'light' : 'dark';
    button.textContent = ICON[now];
    button.title = `Switch to ${LABEL[next].toLowerCase()}`;
    button.setAttribute('aria-label', button.title);
  };

  button.addEventListener('click', () => {
    const next = stored() === 'dark' ? 'light' : 'dark';
    keep(next);
    apply(next);
    draw();
  });

  apply(stored());
  draw();
}

// Applied as this loads, before anything is drawn, so the page never flashes
// the wrong colours.
apply(stored());
