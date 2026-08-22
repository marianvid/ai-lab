// A button that shows it is working.
//
// The feedback for pressing something belongs on the thing that was pressed.
// That is where you are already looking, it cannot be missed, and it disappears
// by itself when the work is done. There used to be a line at the foot of the
// page saying "Saving gemma31-nvfp4 done" — below the fold, wiped by the next
// action, and telling you something the page had already shown you.
//
// While the work runs the button is disabled, so the same action cannot be
// started twice by an impatient second click.

export async function whileWorking(button, label, work) {
  const before = button.textContent;
  const wasDisabled = button.disabled;
  button.textContent = label;
  button.disabled = true;
  try {
    return await work();
  } finally {
    // The page usually redraws over this button, which makes putting it back
    // pointless — but not always: a failure can leave the same button on
    // screen, and it must be pressable again.
    button.textContent = before;
    button.disabled = wasDisabled;
  }
}
