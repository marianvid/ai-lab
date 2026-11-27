// Turning numbers and facts into something readable. Shared by every view.

export function bytes(value) {
  if (!value) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = value, unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(size < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

export function seconds(milliseconds) {
  if (milliseconds === undefined || milliseconds === null) return '—';
  return milliseconds < 1000
    ? `${milliseconds} ms`
    : `${(milliseconds / 1000).toFixed(1)} s`;
}

export function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attributes).forEach(([key, value]) => {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null) node.setAttribute(key, value);
  });
  (Array.isArray(children) ? children : [children])
    .filter(Boolean)
    .forEach((child) => node.append(child));
  return node;
}


// Whether someone is in the middle of typing in `container`: a field there has
// the focus, or holds a value that differs from the one it was drawn with
// (typed and not saved). A page that redraws itself on a timer skips the
// redraw while this is true, so what was typed is never thrown away. A field's
// drawn value is its `value` attribute (`element` sets it), so `defaultValue`
// and `defaultChecked` are what was saved.
export function beingEdited(container) {
  const active = document.activeElement;
  if (active && container.contains(active)
      && ['INPUT', 'TEXTAREA', 'SELECT'].includes(active.tagName)) return true;
  return [...container.querySelectorAll('input, textarea')].some((field) => (
    field.type === 'checkbox' || field.type === 'radio'
      ? field.checked !== field.defaultChecked
      : field.value !== field.defaultValue));
}


// What goes in the browser tab.
//
// The machine is part of it, because there is usually more than one of these
// open: one against the Linux box and one against the Mac. Two tabs both
// reading "AI-Lab" are two tabs you have to click to tell apart, and the wrong
// one is the one where a model gets unloaded.
//
// The configured `title` (Settings → This AI-Lab) beside what the machine
// reports (Linux, macOS), so a second machine is told apart at once even
// before somebody gives it a different name.
export function pageTitle(settings) {
  const name = (settings && settings.title) || 'AI-Lab';
  const machine = settings && settings.host && settings.host.operating_system;
  return machine ? `${name} · ${machine}` : name;
}
