// Every request goes through here.
//
// One place for the JSON handling and the error shape, so a view never
// contains transport details. The server always answers with JSON, including
// for errors, which is what makes this short.

async function request(method, path, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${method} ${path} failed`);
  return payload;
}

export const api = {
  settings: () => request('GET', '/api/settings'),
  models: (engine) => request('GET', '/api/models' + (engine ? `?engine=${engine}` : '')),
  formats: () => request('GET', '/api/formats'),
  updateRepository: (id, changes) =>
    request('PATCH', `/api/repositories/${encodeURIComponent(id)}`, changes),
  deleteModel: (id) => request('DELETE', `/api/models/${encodeURIComponent(id)}`),
  instances: () => request('GET', '/api/instances'),
  load: (id) => request('POST', `/api/instances/${encodeURIComponent(id)}/load`),
  unload: (id) => request('POST', `/api/instances/${encodeURIComponent(id)}/unload`),
  // Save the settings and restart the model with them, as one action.
  apply: (id, changes) =>
    request('POST', `/api/instances/${encodeURIComponent(id)}/apply`, changes),
  update: (id, changes) =>
    request('PATCH', `/api/instances/${encodeURIComponent(id)}`, changes),
  newInstanceForm: () => request('GET', '/api/instances/new'),
  createInstance: (payload) => request('POST', '/api/instances', payload),
  deleteInstance: (id) => request('DELETE', `/api/instances/${encodeURIComponent(id)}`),
  transfers: () => request('GET', '/api/downloads'),
  // No destination: the server puts a model where its format lives.
  download: (repo, name) => request('POST', '/api/downloads', { repo, name }),
  cancelDownload: (id) => request('POST', `/api/downloads/${encodeURIComponent(id)}/cancel`),
  builds: () => request('GET', '/api/builds'),
  checkBuild: (engine) => request('POST', `/api/builds/${encodeURIComponent(engine)}/check`),
  updateBuild: (engine) => request('POST', `/api/builds/${encodeURIComponent(engine)}/update`),
  search: (q, all) => request(
    'GET', `/api/hf/search?q=${encodeURIComponent(q)}${all ? '&all=1' : ''}`),
  remoteSets: (repo, all) =>
    request('GET', `/api/hf/sets?repo=${encodeURIComponent(repo)}${all ? '&all=1' : ''}`),
};
