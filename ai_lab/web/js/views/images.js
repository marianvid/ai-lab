import { api } from '../api.js';
import { element } from '../format.js';

export async function render(target) {
  const [profiles, jobs] = await Promise.all([api.imageProfiles(), api.imageJobs()]);
  const select = element('select', { name: 'profile' }, profiles.map((profile) =>
    element('option', { value: profile.id, text: `${profile.id} · ${profile.model}` })));
  const prompt = element('textarea', { name: 'prompt', rows: '5',
                                       placeholder: 'Describe the image' });
  const source = element('input', { name: 'image', type: 'file', accept: 'image/*' });
  const sourceField = element('label', {}, [
    element('span', { text: 'Source image' }), source,
  ]);
  const status = element('p', { class: 'muted' });
  const submit = element('button', { type: 'submit' });
  const selectedProfile = () => profiles.find((profile) => profile.id === select.value);
  const showSelectedTask = () => {
    const editing = selectedProfile()?.task === 'edit';
    sourceField.hidden = !editing;
    source.required = editing;
    submit.textContent = editing ? 'Queue edit' : 'Queue generation';
  };
  select.onchange = showSelectedTask;
  const form = element('form', { class: 'panel stack', onsubmit: async (event) => {
    event.preventDefault();
    submit.disabled = true;
    try {
      const editing = selectedProfile()?.task === 'edit';
      let job;
      if (editing) {
        const body = new FormData();
        body.append('profile', select.value);
        body.append('prompt', prompt.value);
        body.append('async', 'true');
        body.append('image', source.files[0]);
        job = await api.editImage(body);
      } else {
        job = await api.generateImage({profile: select.value,
                                       prompt: prompt.value, async: true});
      }
      status.textContent = `Queued ${job.id}`;
      await render(target);
    } catch (error) {
      status.textContent = error.message;
      status.className = 'error';
    } finally { submit.disabled = false; }
  }}, [element('h2', { text: 'Named image workflow' }), select, prompt,
       sourceField, submit, status]);
  showSelectedTask();
  const preview = element('section', { class: 'panel', hidden: true });
  const showResult = async (job) => {
    const detail = await api.imageJob(job.id);
    const images = (detail.result?.data || []).map((item, index) => {
      const src = `data:${item.mime_type || 'image/png'};base64,${item.b64_json}`;
      return element('figure', {}, [
        element('img', { src, alt: `Result ${index + 1} from ${job.profile}` }),
        element('figcaption', {}, element('a', {
          href: src, download: `${job.id}-${index + 1}.png`, text: 'Download',
        })),
      ]);
    });
    preview.hidden = false;
    preview.replaceChildren(element('h2', { text: `Result · ${job.profile}` }), ...images);
  };
  const rows = jobs.map((job) => element('tr', {}, [
    element('td', { text: job.id.slice(0, 10) }),
    element('td', { text: job.profile }),
    element('td', { text: job.status }),
    element('td', {}, job.status === 'queued' || job.status === 'running'
      ? element('button', { text: 'Cancel', onclick: async () => {
          await api.cancelImageJob(job.id); await render(target);
        }})
      : job.status === 'succeeded'
        ? element('button', { text: 'View', onclick: () => showResult(job) })
        : element('span', { text: job.error || '' })),
  ]));
  const table = element('table', {}, [
    element('thead', {}, element('tr', {}, ['Job', 'Profile', 'Status', 'Result'].map(
      (text) => element('th', { text })))) , element('tbody', {}, rows)]);
  target.replaceChildren(form, element('section', { class: 'panel' }, [
    element('h2', { text: 'Image jobs' }), table]), preview);
}
