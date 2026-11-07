// Workspace 4: Generalization, then ship it. It took over the old Workspace 4's content (ws_unused.js is
// kept but unused): the same starting model, blocks and guided steps, plus Your image, the model card and the
// hidden class image. Blocks, code generation and the page itself are shared (blocks.js, generators.js, app.js).
import { categoryToolbox } from "../blocks.js";
import { GENERALIZATION_STEPS, START_STATE as UNUSED_START, checkModel as checkUnused } from "./ws_unused.js";

// Result tabs to show, in order.
export const TABS = ["steps", "training", "compare", "confusion", "mistakes", "practice", "upload", "card"];

export const COMPARE_CURVES = "val";

// The old Workspace 4's small training set and blocks.
export const BLOCK_OPTIONS = { trainSize: true, weightDecay: true };

export const STEPS = `
  <ol class="steps">
${GENERALIZATION_STEPS}    <li><b>Your image.</b> Once your model has trained, choose any picture in <b>Your image</b>. You'll see
      each step of the encoding: your picture shrunk, reduced to a few colors, written in handwritten
      characters, and what your model reads back. Download the result to share it with the class.</li>
    <li><b>Model card.</b> <b>Model card</b> describes your latest finished run: what it is, how it was
      trained, how well it does and where it goes wrong. Read it, then download it.</li>
    <li><b>Decode the hidden class image.</b> Save &amp; Grade decodes the hidden class image with your
      latest finished run. Its characters aren't in this workspace, so you can't tune on them, and you'll
      see only your accuracy: the picture stays secret until the class vote combines everyone's
      predictions. <span id="submit-note"></span></li>
  </ol>`;

export const TOOLBOX = categoryToolbox({
  data: ["load_data"],
  model: ["model", "conv", "relu", "maxpool", "batchnorm", "dropout", "flatten", "linear"],
  training: ["optimizer", "scheduler", "loss", "augment", "early_stopping", "train_settings"],
  evaluation: ["evaluate", "confusion_matrix", "worst_mistakes", "decode_practice"],
});

export const MAX_INSTANCES = {
  load_data: 1, model: 1, optimizer: 1, scheduler: 1, loss: 1, augment: 1, early_stopping: 1,
  train_settings: 1, evaluate: 1, confusion_matrix: 1, worst_mistakes: 1, decode_practice: 1,
};

// The old Workspace 4's start: a big CNN with sensible training on the small set; the only problem is overfitting.
export const START_STATE = UNUSED_START;

export const checkModel = checkUnused;
