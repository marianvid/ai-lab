// The small pictures that say what a model can do.
//
// Two of them, and both are the shape everybody else uses for the same thing:
// a wrench for calling tools, and a photograph — a frame with a hill and a sun
// in it — for reading pictures. Recognising them should not require having
// read this project.
//
// They are drawn here rather than loaded, because a page that fetches icons
// has a moment where the row is the wrong width, and because this whole
// interface is served by one small Python process with no asset pipeline.
//
// Every line is `currentColor`, so an icon is whatever colour the text around
// it is and the dark and light themes need no second copy.

const PICTURES = 'images';
const TOOLS = 'tools';

// Drawn on a 24-by-24 grid, which is what these shapes are normally drawn on;
// the size on screen is set in the stylesheet.
const SHAPES = {
  [TOOLS]: {
    title: 'Can call tools',
    // A wrench, at the usual diagonal.
    path: 'M14.7 6.3a4 4 0 0 0 5 5l-9.4 9.4a2.1 2.1 0 0 1-3-3z M14.7 6.3 17.5 3.5'
        + 'a4 4 0 0 1 3 6.8',
  },
  [PICTURES]: {
    title: 'Can read pictures',
    // A frame, a sun in the top left, a hill along the bottom.
    path: 'M3 5h18v14H3z M8.5 9.5a1.2 1.2 0 1 1-2.4 0 1.2 1.2 0 0 1 2.4 0'
        + ' M3 16l5-4.5 4 3.5 3.5-3L21 16',
  },
};

// The order they appear in, so two rows with the same capabilities look the
// same. Alphabetical would put pictures first, which reads oddly next to a
// name; tools is the commoner one and goes first.
const ORDER = [TOOLS, PICTURES];


// One icon, or nothing if the name is not one this knows.
export function capability(name) {
  const shape = SHAPES[name];
  if (!shape) return null;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', `icon capability ${name}`);
  svg.setAttribute('aria-label', shape.title);
  svg.setAttribute('role', 'img');
  // A tooltip inside an SVG is a <title> child; the title attribute is not it.
  const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
  title.textContent = shape.title;
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', shape.path);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', 'currentColor');
  path.setAttribute('stroke-width', '1.7');
  path.setAttribute('stroke-linecap', 'round');
  path.setAttribute('stroke-linejoin', 'round');
  svg.append(title, path);
  return svg;
}


// Every icon for a set of capabilities, in a fixed order.
//
// `taken` is what the settings have switched off. The weights decide what a
// model *can* do; an entry can only take something away — vLLM's "Text only"
// loads a model that can see without the part that sees, and showing the
// picture icon there would be a promise the running model will not keep.
export function capabilities(has, taken = []) {
  const have = new Set(has || []);
  (taken || []).forEach((name) => have.delete(name));
  return ORDER.filter((name) => have.has(name)).map(capability).filter(Boolean);
}


// What a set of settings takes away. Kept next to the icons because it is the
// same question — what will the running model actually do — and there is
// exactly one of these to know about.
export function suppressed(params) {
  return (params && params.language_model_only) ? [PICTURES] : [];
}

export { PICTURES, TOOLS };
