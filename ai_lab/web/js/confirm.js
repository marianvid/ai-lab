// Asking before something irreversible.
//
// The browser's own confirm() defaults to OK, which is the wrong way round for
// deleting files: the safe answer should be the one you get by pressing Enter
// or Escape, or by clicking without reading. Here Cancel holds the focus and
// Escape closes the dialog, so the destructive button has to be aimed at
// deliberately.

import { element } from './format.js';

export function confirmDestructive({ title, body, confirmLabel = 'Delete' }) {
  return new Promise((resolve) => {
    // The answer is decided before the dialog closes. Closing fires an event
    // that also settles this promise, and relying on which of the two happens
    // first would leave the whole thing one browser quirk away from silently
    // refusing every deletion.
    let answer = false;
    const finish = () => resolve(answer);

    const cancel = element('button', {
      class: 'action', autofocus: 'autofocus', text: 'Cancel',
      onclick: () => { answer = false; dialog.close(); },
    });
    const proceed = element('button', {
      class: 'action danger', text: confirmLabel,
      onclick: () => { answer = true; dialog.close(); },
    });

    const dialog = element('dialog', { class: 'confirm' }, [
      element('h3', { text: title }),
      element('p', {}, body),
      element('div', { class: 'row buttons' }, [cancel, proceed]),
    ]);

    // Closing by Escape, or by any means other than the buttons, leaves the
    // answer at its default, which is no.
    dialog.addEventListener('close', () => { dialog.remove(); finish(); });
    document.body.append(dialog);
    dialog.showModal();
    cancel.focus();
  });
}
