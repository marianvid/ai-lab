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

// A client's caps (docs/access.md, "Caps"): how many tokens of its text calls
// may be in flight at once, counted as prompt + answer limit, and how many
// calls. Empty is no cap. Over a cap a call is refused at once with 429.
function capsCell(client, refresh) {
  const caps = client.caps || {};
  const box = (key, placeholder, title) => element('input', {
    type: 'number', min: '1', step: '1', size: 8, placeholder, title,
    'data-cap': key, value: caps[key] !== undefined ? String(caps[key]) : '',
  });
  const tokens = box('max_tokens_in_flight', 'tokens',
                     'Tokens in flight at once (prompt + answer limit of every call). Empty: no cap.');
  const calls = box('max_calls_at_once', 'calls', 'Calls at once. Empty: no cap.');
  const number = (node) => (node.value.trim() === '' ? null : parseInt(node.value, 10));
  const save = element('button', {
    class: 'action small', text: 'Save caps',
    onclick: () => act('Could not set the caps', () => api.setClientCaps(client.name, {
      max_tokens_in_flight: number(tokens), max_calls_at_once: number(calls),
    }), refresh),
  });
  return element('td', { class: 'caps' }, [tokens, calls, save]);
}

function inFlight(client) {
  const held = client.in_flight;
  if (!client.caps || !Object.keys(client.caps).length || !held) return '—';
  return `${held.calls} calls, ${held.tokens.toLocaleString()} tokens`;
}

function clientRow(client, refresh) {
  return element('tr', {}, [
    element('td', { text: client.name }),
    element('td', { text: when(client.last_call_at) }),
    element('td', { text: when(client.heartbeat_at) }),
    capsCell(client, refresh),
    element('td', { text: inFlight(client) }),
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
  const head = element('tr', {}, ['Client', 'Last call', 'Last heartbeat',
    'Caps (tokens, calls)', 'In flight now', '']
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
  const fresh = element('input', { type: 'password', placeholder: `new password (${security.password_min ?? 8}+)` });
  const save = element('button', {
    class: 'action', text: security.password_set ? 'Change password' : 'Set password',
    onclick: () => act('Could not set the password',
                       () => api.setPassword(fresh.value, current.value || undefined), refresh),
  });
  return section('Security', [
    toggle('require_token', 'Require a token on every call',
           'Off: a call without a valid token is served and written to the log. On: it is '
           + 'refused with 401. Holds only with login on (below): without it the page hands '
           + 'any visitor a session admitted as the web interface, so AI-Lab refuses that '
           + 'combination. Set a password and turn login on first.'),
    toggle('login_required', 'Ask for the password (off: no logon)',
           'Needs a password first. Turning it off while a token is required is refused: the '
           + 'token requirement would then protect nothing.'),
    element('div', { class: 'row' }, [...(security.password_set ? [current] : []), fresh, save]),
    ...SECURITY_NUMBERS.map((spec) => numberRow(security, spec, refresh)),
    line('Password hash', `scrypt, cost ${security.password_hash?.n ?? '?'} (r ${security.password_hash?.r ?? '?'}, p ${security.password_hash?.p ?? '?'})`,
         'How the password is kept: a salted scrypt hash, never the text. Shown, not changed '
         + 'here: a new cost would need every password set again.'),
    ...(security.token_holds === false ? [element('p', { class: 'error', text: TOKEN_DOES_NOT_HOLD })] : []),
  ]);
}

// Security's numbers (clients/security.py NUMBERS), each with its meaning, its
// default and what too small or too large does (audit m22: the mailbox's two
// were on no page). "Remember me" is the first: how long this browser stays
// logged in, through the browser closing and AI-Lab restarting.
const SECURITY_NUMBERS = [
  { key: 'session_days', label: 'Stay logged in for (days)', min: 1, max: 365, fallback: 30,
    help: 'How long a login holds, in days: the browser keeps it after closing, and AI-Lab '
          + 'keeps it after a restart (only a digest of the session is written). Default 30; '
          + 'from 1 to 365. Shorter means logging in more often; longer means a lost or shared '
          + 'browser stays let in longer. Applies to logins made from then on; a login already open keeps its end.' },
  { key: 'password_min', label: 'Shortest password (characters)', min: 8, max: 128, fallback: 8,
    help: 'The fewest characters a new password may have. Default 8; from 8 to 128. Longer '
          + 'is harder to guess; a length the owner cannot type reliably locks nobody out '
          + 'but makes changing it harder. Checked when a password is set.' },
  { key: 'mailbox_keep', label: 'Mailbox keeps (messages)', min: 10, max: 100000, fallback: 2000,
    help: 'How many messages the agents\' mailbox keeps; the oldest go first. Default 2 000; '
          + 'from 10 to 100 000. Too few: a message is gone before the other side reads it. '
          + 'Too many: a larger file read at every post.' },
  { key: 'mailbox_max_chars', label: 'Mailbox message, longest (characters)', min: 100,
    max: 200000, fallback: 20000,
    help: 'The longest message the mailbox takes. Default 20 000; from 100 to 200 000. Too '
          + 'small: a report has to be sent in many parts. Too large: one message can fill '
          + 'what the mailbox keeps.' },
];

function numberRow(security, spec, refresh) {
  const box = element('input', { type: 'number', min: String(spec.min), max: String(spec.max),
                                 step: '1', 'data-security': spec.key,
                                 value: String(security[spec.key] ?? spec.fallback) });
  const save = element('button', {
    class: 'action', text: 'Save',
    onclick: () => act(`Could not change ${spec.label.toLowerCase()}`,
                       () => api.updateSecurity({ [spec.key]: Number(box.value) }), refresh),
  });
  return element('label', { class: 'row', title: spec.help },
                 [element('span', { text: spec.label }), box, save]);
}

export const TOKEN_DOES_NOT_HOLD = 'The token requirement does not hold on this machine until '
  + 'login is on: set a password on Access, then turn "Ask for the password" on.';

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
