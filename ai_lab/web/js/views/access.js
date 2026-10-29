// The Access page (docs/access.md): On/Off, the registered clients with their
// last call and last heartbeat, the operation log, and Security.
//
// Nothing here arrives as an event, so the page refreshes itself on a timer,
// like the Gateway page.

import { api } from '../api.js';
import { element } from '../format.js';
import { showNotice } from '../confirm.js';

const EVERY_MS = 5000;
let timer = null;

function section(title, children) {
  return element('div', { class: 'section' }, [
    element('h3', { text: title }),
    element('div', { class: 'card' }, children),
  ]);
}

function line(label, value, title) {
  return element('div', { class: 'row', ...(title ? { title } : {}) }, [
    element('span', { class: 'muted', text: label }),
    element('span', { text: value }),
  ]);
}

function when(stamp) {
  const value = Number(stamp);
  return Number.isFinite(value) && value > 0 ? new Date(value * 1000).toLocaleString() : '—';
}

async function act(title, work, refresh) {
  try {
    await work();
  } catch (error) {
    await showNotice({ title, body: error.message });
  }
  refresh();
}

function powerSection(power, refresh) {
  const next = !power.on;
  return section('AI-Lab', [
    line('State', power.on ? 'On' : `Off since ${when(power.since)}`,
         'Off: the calls running now end at once, and every new call that would use a model is refused.'),
    element('div', { class: 'row' }, [
      element('button', {
        class: power.on ? 'action danger' : 'action', text: next ? 'Switch On' : 'Switch Off',
        onclick: () => act('Could not switch', () => api.switchPower(next), refresh),
      }),
    ]),
  ]);
}

function clientRow(client, refresh) {
  return element('tr', {}, [
    element('td', { text: client.name }),
    element('td', { text: when(client.last_call_at) }),
    element('td', { text: when(client.heartbeat_at) }),
    element('td', {}, [element('button', {
      class: 'action small', text: 'Remove',
      onclick: () => act('Could not remove the client', () => api.removeClient(client.name), refresh),
    })]),
  ]);
}

function registerForm(refresh) {
  const name = element('input', { type: 'text', placeholder: 'scheduler-linux' });
  const add = element('button', {
    class: 'action', text: 'Register',
    onclick: () => act('Could not register the client', async () => {
      const created = await api.registerClient(name.value.trim());
      await showNotice({ title: `Token for ${created.name}`,
                         body: `${created.token}\n\n${created.note}` });
    }, refresh),
  });
  return element('div', { class: 'row' }, [name, add]);
}

function clientsSection(clients, refresh) {
  const head = element('tr', {}, ['Client', 'Last call', 'Last heartbeat', '']
    .map((text) => element('th', { text })));
  return section('Clients', [
    element('table', { class: 'table' }, [head, ...clients.map((c) => clientRow(c, refresh))]),
    registerForm(refresh),
  ]);
}

function journalSection(journal) {
  const lines = journal.lines.slice(-100).reverse();
  return section('What AI-Lab is doing', lines.length
    ? lines.map((item) => line(new Date(item.at * 1000).toLocaleTimeString(), item.text))
    : [element('p', { class: 'muted', text: 'Nothing yet.' })]);
}

function securitySection(security, refresh) {
  const toggle = (key, label, title) => {
    const box = element('input', { type: 'checkbox', ...(security[key] ? { checked: 'checked' } : {}) });
    box.addEventListener('change', () =>
      act('Could not change the setting', () => api.updateSecurity({ [key]: box.checked }), refresh));
    return element('label', { class: 'row', title }, [box, element('span', { text: label })]);
  };
  const current = element('input', { type: 'password', placeholder: 'current password' });
  const fresh = element('input', { type: 'password', placeholder: 'new password (8+)' });
  const save = element('button', {
    class: 'action', text: security.password_set ? 'Change password' : 'Set password',
    onclick: () => act('Could not set the password',
                       () => api.setPassword(fresh.value, current.value || undefined), refresh),
  });
  return section('Security', [
    toggle('require_token', 'Require a token on every call',
           'Off: a call without a valid token is served and written to the log. On: it is refused with 401.'),
    toggle('login_required', 'Ask for the password (off: no logon)',
           'Needs a password first.'),
    element('div', { class: 'row' }, [...(security.password_set ? [current] : []), fresh, save]),
  ]);
}

export async function render(target) {
  const [power, clients, journal, security] = await Promise.all([
    api.power(), api.clients(), api.journal(0), api.security()]);
  const refresh = () => render(target);
  target.replaceChildren(
    powerSection(power, refresh),
    clientsSection(clients.clients, refresh),
    journalSection(journal),
    securitySection(security, refresh));
  stopRefreshing();
  timer = setTimeout(() => {
    if (!document.activeElement || !target.contains(document.activeElement)) refresh();
  }, EVERY_MS);
}

export function stopRefreshing() {
  if (timer) clearTimeout(timer);
  timer = null;
}
