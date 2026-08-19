// Light or dark.

const KEY = 'ai-lab-theme';
const ORDER = ['light', 'dark'];
const LABEL = { light: 'Light', dark: 'Dark' };

function systemPrefersDark() {
  return window.matchMedia
    && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function stored() {
  const value = localStorage.getItem(KEY);
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
  const draw = () => { button.textContent = LABEL[stored()]; };

  button.addEventListener('click', () => {
    const next = stored() === 'dark' ? 'light' : 'dark';
    localStorage.setItem(KEY, next);
    apply(next);
    draw();
  });

  apply(stored());
  draw();
}

// Applied as this loads, before anything is drawn, so the page never flashes
// the wrong colours.
apply(stored());
