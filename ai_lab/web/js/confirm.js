// Modal interruptions: the two things that must not be missed.
//
// Asking before something irreversible, and telling you something you would
// otherwise scroll past. They share this file because they are one job — a
// dialog that takes the page away until it is answered — and splitting them
// would mean two copies of the mechanics below.
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


// Telling you something you must not miss.
//
// The status line at the foot of the page is the right place for progress and
// for a success. It is the wrong place for a load that failed: the page is
// long, the line is below it, it has no border and no background, and the next
// action wipes it. A model that would not start took forty seconds to say so
// and then said it where nobody was looking.
//
// The message is worth reading in full. An engine that refuses a context says
// which one would have fitted, and that sentence is the whole point of showing
// it at all — so this does not truncate.
export function showNotice({ title, body, dismissLabel = 'Close' }) {
  return new Promise((resolve) => {
    const dismiss = element('button', {
      class: 'action', autofocus: 'autofocus', text: dismissLabel,
      onclick: () => dialog.close(),
    });

    const dialog = element('dialog', { class: 'confirm notice' }, [
      element('h3', { text: title }),
      element('p', { class: 'detail' }, body),
      element('div', { class: 'row buttons' }, [dismiss]),
    ]);

    dialog.addEventListener('close', () => { dialog.remove(); resolve(); });
    document.body.append(dialog);
    dialog.showModal();
    dismiss.focus();
  });
}
