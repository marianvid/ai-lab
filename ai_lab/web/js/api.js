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
  if (!response.ok) {
    // Some refusals are worth acting on and not only reading. "The card is
    // busy" is one: the caller can offer to go ahead anyway, but only if it
    // knows that is why the request failed. The message alone cannot be
    // tested for without matching on its words.
    const error = new Error(payload.error || `${method} ${path} failed`);
    error.status = response.status;
    if (payload.busy) error.busy = payload.busy;
    throw error;
  }
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
  // `force` goes ahead even though a request is being answered on the card.
  // The server refuses without it; only the person looking at the screen can
  // decide that cutting an answer short is the lesser evil.
  load: (id, force) =>
    request('POST', `/api/instances/${encodeURIComponent(id)}/load`,
            force ? { force: true } : undefined),
  unload: (id, force) =>
    request('POST', `/api/instances/${encodeURIComponent(id)}/unload`,
            force ? { force: true } : undefined),
  // Save the settings and restart the model with them, as one action.
  apply: (id, changes, force) =>
    request('POST', `/api/instances/${encodeURIComponent(id)}/apply`,
            force ? { ...changes, force: true } : changes),
  gateway: () => request('GET', '/api/gateway'),
  updateGateway: (changes) => request('PATCH', '/api/gateway', changes),
  logs: (id) => request('GET', `/api/instances/${encodeURIComponent(id)}/logs`),
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
  // What an update would bring. A GET because it changes nothing — it
  // reads a checkout, asks upstream what it wrote, and asks the package
  // manager what it would do.
  buildChanges: (engine) =>
    request('GET', `/api/builds/${encodeURIComponent(engine)}/changes`),
  // Only what this machine can run is listed, and there is no way to ask for
  // the rest. The search answer carries how many were filtered out.
  search: (q) => request('GET', `/api/hf/search?q=${encodeURIComponent(q)}`),
  remoteSets: (repo) =>
    request('GET', `/api/hf/sets?repo=${encodeURIComponent(repo)}`),
};
