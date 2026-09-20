// Load progress for this Models page instance.
import { onProgress } from '../events.js';
import { seconds } from '../format.js';

export function createProgress() {
    const progress = new Map();
    let subscribed = false;
  function paint(bar, event) {
    bar.className = 'bar thin' + (event.phase === 'ready' ? ' done'
      : event.phase === 'failed' ? ' failed' : '');
    bar.firstChild.style.width = `${Math.round((event.progress || 0) * 100)}%`;
    bar.title = `${Math.round((event.progress || 0) * 100)}% · ${event.phase}`
      + ` · ${seconds(event.elapsed_ms)}`
      + ` · this model ${Math.round(event.memory_used_mb)} MB`;
  }

  function clearBar(bar) {
    bar.className = 'bar thin';
    bar.firstChild.style.width = '0%';
    bar.title = 'nothing loaded';
  }

  // On a fresh page there are no events yet, but the model may well be loaded
  // already. Drawing an empty bar beside the word "ready" puts two contradictory
  // things on screen and makes the reader trust neither.
  function paintFromState(bar, instance) {
    if (instance.ready) {
      bar.className = 'bar thin done';
      bar.firstChild.style.width = '100%';
      bar.title = 'loaded';
    } else if (instance.running) {
      bar.className = 'bar thin';
      bar.firstChild.style.width = '50%';
      bar.title = 'starting';
    } else {
      clearBar(bar);
    }
  }

  function subscribe() {
    if (subscribed) return;
    subscribed = true;
    // The bar and nothing else. These events arrive for every load, including
    // ones an agent caused, and announcing them at the foot of the page meant
    // being told about work you did not start — in a message naming the entry
    // and the file on disk, two names from two places, one of which is on
    // another page. What you pressed yourself is reported by the thing you
    // pressed.
    onProgress((event) => {
      progress.set(event.instance_id, event);
      const bar = document.querySelector(`[data-bar="${CSS.escape(event.instance_id)}"]`);
      if (bar) paint(bar, event);
    });
  }


    return { progress, paint, paintFromState, subscribe };
}
