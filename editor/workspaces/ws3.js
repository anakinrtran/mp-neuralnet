// Workspace 3: Training methods. Everything here is specific to this workspace; blocks,
// code generation and the page itself are shared (blocks.js, generators.js, app.js).
import { categoryToolbox } from "../blocks.js";
import { outputLayerProblems } from "../checks.js";
import { checkModel as checkCnn } from "./ws2.js";

// Result tabs to show, in order.
export const TABS = ["steps", "training", "compare", "gradients", "onestep", "practice"];

// Compare runs overlays the loss curves of ticked runs and shows each run's training settings.
export const COMPARE_CURVES = true;

export const STEPS = `
  <ol class="steps">
    <li><b>Run it as it is.</b> The model is a working CNN, but its training is broken. Click
      <b>▶ Run</b> and watch <b>Training</b>. What does the loss do in the first moments, and
      after that? Where does validation accuracy end up, and what would random guessing among 36
      characters score?</li>
    <li><b>Diagnose it.</b> Open <b>Gradients</b>. How much does each layer change per step,
      at the very start and later on? What does that tell you about what went wrong? Fix the
      training blocks so the model learns again. Leave the Model alone for now.</li>
    <li><b>Compare optimizers.</b> Try SGD, SGD + momentum and Adam, each at a learning rate that
      suits it. In <b>Compare runs</b>, overlay their loss curves. Which drops fastest? Which
      is the bumpiest?</li>
    <li><b>Batch size.</b> Try a much smaller and a much larger batch. What changes: the run time,
      how bumpy the loss curve is, the final accuracy?</li>
    <li><b>Loss function.</b> Swap cross-entropy for MSE. What happens to the loss curve and to
      accuracy? Why might that be?</li>
    <li><b>Stabilize.</b> Add BatchNorm to your model, and try a Scheduler. Can you use a bigger
      learning rate now? What does <b>Gradients</b> show?</li>
    <li><b>Beat the budget.</b> You only get 2 epochs, so good training choices are what count.
      <b>Practice image</b> shows what your model reads.</li>
    <li><b>Grade.</b> <span id="submit-note"></span></li>
  </ol>`;

export const TOOLBOX = categoryToolbox({
  data: ["load_data"],
  model: ["model", "conv", "relu", "maxpool", "batchnorm", "flatten", "linear"],
  training: ["optimizer", "scheduler", "loss", "train_settings"],
  evaluation: ["evaluate", "decode_practice"],
});

// Only one of each pipeline step is allowed.
export const MAX_INSTANCES = {
  load_data: 1, model: 1, optimizer: 1, scheduler: 1, loss: 1, train_settings: 1,
  evaluate: 1, decode_practice: 1,
};

const chain = (...blocks) => blocks.reduceRight((next, block) => ({ ...block, ...(next && { next: { block: next } }) }), null);

// A working CNN whose training is broken: plain SGD with a learning rate so high that the
// first steps kill the network (the loss flatlines at chance).
export const START_STATE = {
  blocks: {
    languageVersion: 0,
    blocks: [{
      x: 40, y: 40,
      ...chain(
        { type: "load_data" },
        { type: "model", inputs: { LAYERS: { block: chain(
          { type: "conv", fields: { FILTERS: 16 } },
          { type: "relu" },
          { type: "maxpool" },
          { type: "flatten" },
          { type: "linear", fields: { OUTPUTS: 128 } },
          { type: "relu" },
          { type: "linear", fields: { OUTPUTS: 36 } },
        ) } } },
        { type: "optimizer", fields: { KIND: "sgd", LR: 10 } },
        { type: "loss", fields: { KIND: "ce" } },
        { type: "train_settings", fields: { EPOCHS: 2, BATCH: "128" } },
        { type: "evaluate" },
        { type: "decode_practice" },
      ),
    }],
  },
};

/** Workspace 2's checks, with BatchNorm allowed anywhere except last. */
export function checkModel(layers, cfg) {
  const found = checkCnn(layers.filter((b) => b.type !== "batchnorm"), cfg);
  return {
    problems: [...new Set([...found.problems, ...outputLayerProblems(layers, cfg)])],
    // Workspace 2 nudges toward Conv; here a Conv-less model is a fair experiment
    hints: found.hints.filter((h) => !h.startsWith("This model has no Conv block")),
  };
}
