// Model and engine compatibility shared by model rows and the add form.
// Responses from an older manager, and fixtures saved before audio existed,
// have no task field. They are text models: that was the only kind there was.
export const taskOf = (model) => model?.task || 'text-generation';
export const tasksOf = (engine) => engine?.tasks || ['text-generation'];
export const engineCanUse = (engine, model) =>
  engine.formats.includes(model.format) && tasksOf(engine).includes(taskOf(model))
  && (!engine.supported_model_ids || engine.supported_model_ids.includes(model.id));
