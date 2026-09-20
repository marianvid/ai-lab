// The library page composes installed models and remote discovery.
import { api } from '../api.js';
import { element } from '../format.js';
import { setInstalledRedraw, repositorySection, refreshLibrary } from './library_installed.js';
import { createDiscovery } from './library_discovery.js';
const { setDiscoveryContext, searchResults, searchBox, repositoryRow, transferList } = createDiscovery();

// -- the view ---------------------------------------------------------------

export async function render(container) {
  const redraw = () => render(container);
  setInstalledRedraw(redraw);
  const [models, settings, transfers] = await Promise.all([
    api.models(), api.settings(), api.transfers(),
  ]);
  setDiscoveryContext(settings, transfers, redraw);
  const { outcome, results, openRepo } = searchResults();

  // Core and benchmark are physical locations, not different kinds of model.
  // Pair their cloned repositories into one visible section and let each row's
  // storage badge say where that particular model lives.
  const repositoryGroups = new Map();
  const groupByRepository = new Map();
  settings.repositories.forEach((repository) => {
    const key = [repository.name, repository.format, repository.task || ''].join('\u0000');
    if (!repositoryGroups.has(key)) {
      repositoryGroups.set(key, { repository, models: [] });
    }
    groupByRepository.set(repository.id, repositoryGroups.get(key));
  });
  models.forEach((model) => {
    const repositoryId = model.id.split('/')[0];
    const group = groupByRepository.get(repositoryId);
    if (group) group.models.push(model);
  });

  const sections = [...repositoryGroups.values()]
    .filter((group) => group.models.length > 0)
    .map((group) => repositorySection(
      group.repository,
      group.models.sort((left, right) => left.name.localeCompare(right.name)),
    ));

  // Getting models comes first, then what you already have.
  //
  // replaceChildren turns a null into the text "null", so the list is filtered
  // rather than handed straight over — two empty slots showed up on the page
  // as "nullnull".
  const children = [
    searchBox(),
    // Right under the box that was typed into, above where the results would
    // be. Empty whenever there are results to look at instead.
    outcome ? element('p', { class: 'muted outcome', text: outcome }) : null,
    ...results.map(repositoryRow),
    transferList(transfers.filter((transfer) =>
      ['queued', 'running', 'failed'].includes(transfer.state)
      && transfer.repo !== openRepo)),
    element('hr', {}),
    element('div', { class: 'row' }, [
      element('div', { class: 'grow' }, [
        element('h2', { text: 'Installed models' }),
        element('p', { class: 'muted',
                       text: `${models.length} found across core and benchmark storage` }),
      ]),
      element('button', {
        class: 'action',
        text: 'Refresh library',
        title: 'Scan core and benchmark storage again',
        onclick: (event) => refreshLibrary(event.currentTarget),
      }),
    ]),
    ...sections,
  ].filter(Boolean);
  container.replaceChildren(...children);
}
