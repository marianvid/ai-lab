import { api } from '../api.js';
import { element } from '../format.js';

const FINAL = new Set(['succeeded', 'failed', 'cancelled']);

export async function runMediaJob(model, task, input, status, cancel) {
  const created = await api.createMediaJob(model, task, input);
  cancel.hidden = false;
  cancel.onclick = async () => {
    cancel.disabled = true;
    try { await api.cancelMediaJob(created.id); status.textContent = 'Cancelling…'; }
    catch (error) { status.textContent = error.message; cancel.disabled = false; }
  };
  try {
    let job = created;
    while (true) {
      job = await api.mediaJob(created.id);
      status.textContent = `${job.status} · job ${created.id}`;
      if (FINAL.has(job.status)) break;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    if (job.status !== 'succeeded') throw new Error(job.error || job.status);
    return job.result;
  } finally {
    cancel.hidden = true;
    cancel.disabled = false;
  }
}

export function showMediaHistory(panel, model, task, display, status) {
  const refresh = async () => {
    try {
      const jobs = (await api.mediaJobs()).filter(
        (job) => job.model === model && job.task === task);
      panel.replaceChildren(element('h2', { text: 'Previous jobs' }),
        element('button', { type: 'button', text: 'Refresh jobs', onclick: refresh }));
      for (const job of jobs) {
        const actions = [];
        if (job.status === 'succeeded') actions.push(element('button', {
          type: 'button', text: 'Open result', onclick: async () => {
            try { display((await api.mediaJob(job.id)).result); }
            catch (error) { status.textContent = error.message; }
          },
        }));
        if (!FINAL.has(job.status)) actions.push(element('button', {
          type: 'button', text: 'Cancel job', onclick: async () => {
            try { await api.cancelMediaJob(job.id); await refresh(); }
            catch (error) { status.textContent = error.message; }
          },
        }));
        panel.append(element('p', {}, [
          element('span', { text: `${job.status} · ${job.id.slice(0, 8)} ` }),
          ...actions,
        ]));
      }
    } catch (error) { panel.textContent = `Could not load jobs: ${error.message}`; }
  };
  refresh();
  return refresh;
}
