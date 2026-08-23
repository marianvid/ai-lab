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


// What goes in the browser tab.
//
// The machine is part of it, because there is usually more than one of these
// open: one against the Linux box and one against the Mac. Two tabs both
// reading "AI-Lab" are two tabs you have to click to tell apart, and the wrong
// one is the one where a model gets unloaded.
//
// Worked out from what the machine reports rather than configured, so a second
// machine is distinguishable the moment it is running rather than after
// somebody remembers to give it a different name.
export function pageTitle(settings) {
  const name = (settings && settings.title) || 'AI-Lab';
  const machine = settings && settings.host && settings.host.operating_system;
  return machine ? `${name} · ${machine}` : name;
}
