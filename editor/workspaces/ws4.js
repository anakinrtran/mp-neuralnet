// Workspace 4: Generalization and evaluation. Everything here is specific to this workspace;
// blocks, code generation and the page itself are shared (blocks.js, generators.js, app.js).
import { categoryToolbox } from "../blocks.js";
import { outputLayerProblems } from "../checks.js";
import { checkModel as checkWs3 } from "./ws3.js";

// Result tabs to show, in order.
export const TABS = ["steps", "training", "compare", "confusion", "mistakes", "practice"];

// Compare runs overlays the validation accuracy of ticked runs (overfitting shows there, not in the loss).
export const COMPARE_CURVES = "val";

// Load data gets a training-set size field; Optimizer gets weight decay.
export const BLOCK_OPTIONS = { trainSize: true, weightDecay: true };

export const STEPS = `
  <ol class="steps">
    <li><b>Overfit on purpose.</b> This model is big, and it trains on only 100 characters per class.
      Click <b>▶ Run</b> and watch <b>Training</b>: training accuracy and validation accuracy are drawn
      on the same chart. When do they pull apart? Where does each one end up? What is the model
      doing with its training characters?</li>
    <li><b>Try each fix alone.</b> Dropout, weight decay (on the Optimizer), Augment training images
      (rotate or shift) and Early stopping: add one at a time to the starting blocks and run it.
      In <b>Compare runs</b>, tick the runs to overlay their validation accuracy. Which fix closes the
      gap between training and validation the most? Which raises validation accuracy the most?</li>
    <li><b>The flip trap.</b> Tick <i>mirror left↔right</i> on Augment training images. What happens to
      validation accuracy? Picture a few characters in a mirror: why might that be?</li>
    <li><b>Confusion matrix.</b> Add <b>Confusion matrix</b> at the end of your program and run it. In the
      <b>Confusion matrix</b> tab, find your model's three worst confusions. Which of them are look-alike
      pairs?</li>
    <li><b>Worst mistakes.</b> Add <b>Worst mistakes</b> and run it. These are the wrong guesses your model
      was most sure about. Would a person get them right?</li>
    <li><b>Combine.</b> Build the best model you can on the small training set. Do the fixes add up when
      you use them together? You can also change the model itself, and train for up to 60 epochs.</li>
    <li><b>Grade.</b> Save &amp; Grade decodes a held-out test image whose characters aren't in this
      workspace, so you can't tune on it. Validation accuracy is your best guess of how it will go.
      <span id="submit-note"></span></li>
  </ol>`;

export const TOOLBOX = categoryToolbox({
  data: ["load_data"],
  model: ["model", "conv", "relu", "maxpool", "batchnorm", "dropout", "flatten", "linear"],
  training: ["optimizer", "scheduler", "loss", "augment", "early_stopping", "train_settings"],
  evaluation: ["evaluate", "confusion_matrix", "worst_mistakes", "decode_practice"],
});

// Only one of each pipeline step is allowed.
export const MAX_INSTANCES = {
  load_data: 1, model: 1, optimizer: 1, scheduler: 1, loss: 1, augment: 1, early_stopping: 1,
  train_settings: 1, evaluate: 1, confusion_matrix: 1, worst_mistakes: 1, decode_practice: 1,
};

export const chain = (...blocks) =>
  blocks.reduceRight((next, block) => ({ ...block, ...(next && { next: { block: next } }) }), null);

// A big CNN with sensible training on a small training set: the only problem is overfitting.
export const START_STATE = {
  blocks: {
    languageVersion: 0,
    blocks: [{
      x: 40, y: 40,
      ...chain(
        { type: "load_data", fields: { PER_CLASS: 100 } },
        { type: "model", inputs: { LAYERS: { block: chain(
          { type: "conv", fields: { FILTERS: 32 } },
          { type: "relu" },
          { type: "maxpool" },
          { type: "flatten" },
          { type: "linear", fields: { OUTPUTS: 256 } },
          { type: "relu" },
          { type: "linear", fields: { OUTPUTS: 36 } },
        ) } } },
        { type: "optimizer", fields: { KIND: "adam", LR: 0.001, WD: 0 } },
        { type: "loss", fields: { KIND: "ce" } },
        { type: "train_settings", fields: { EPOCHS: 20, BATCH: "128" } },
        { type: "evaluate" },
        { type: "decode_practice" },
      ),
    }],
  },
};

/** Workspace 3's checks, with Dropout allowed anywhere except last. */
export function checkModel(layers, cfg) {
  const found = checkWs3(layers.filter((b) => b.type !== "dropout"), cfg);
  if (!layers.length) return found;  // "The Model is empty" says it all
  return {
    problems: [...new Set([...found.problems, ...outputLayerProblems(layers, cfg)])],
    hints: found.hints,
  };
}
