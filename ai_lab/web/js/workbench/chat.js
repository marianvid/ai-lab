import { api } from '../api.js';
import { element } from '../format.js';

export function renderChat(target, model) {
  const history = [];
  const conversation = element('section', { class: 'panel stack',
    'aria-live': 'polite' });
  const system = element('textarea', { rows: '2',
    placeholder: 'Optional system instruction', 'aria-label': 'System instruction' });
  const prompt = element('textarea', { rows: '5', required: 'required',
    placeholder: 'Write a message', 'aria-label': 'Message' });
  const status = element('p', { class: 'muted', role: 'status' });
  const send = element('button', { type: 'submit', text: 'Send' });
  const clear = element('button', { type: 'button', text: 'New conversation',
    onclick: () => { history.length = 0; conversation.replaceChildren(); status.textContent = ''; },
  });
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    const message = prompt.value.trim();
    if (!message) return;
    send.disabled = true;
    status.textContent = 'Waiting for the model…';
    const messages = [
      ...(system.value.trim() ? [{ role: 'system', content: system.value.trim() }] : []),
      ...history, { role: 'user', content: message },
    ];
    try {
      const result = await api.chat(model, messages);
      const answer = result.choices?.[0]?.message?.content;
      if (typeof answer !== 'string') throw new Error('The model returned no text');
      history.push({ role: 'user', content: message },
                   { role: 'assistant', content: answer });
      conversation.append(element('article', { class: 'chat-message' }, [
        element('strong', { text: 'You' }), element('p', { text: message }),
      ]), element('article', { class: 'chat-message' }, [
        element('strong', { text: model }), element('p', { text: answer }),
      ]));
      prompt.value = '';
      status.textContent = '';
    } catch (error) { status.textContent = error.message; }
    finally { send.disabled = false; }
  }}, [system, prompt, element('div', { class: 'inline' }, [send, clear]), status]);
  target.replaceChildren(element('h1', { text: `Chat · ${model}` }),
                         conversation, form);
}
