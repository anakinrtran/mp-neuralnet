// Workspace 5: Ship it. Everything here is specific to this workspace; blocks, code generation
// and the page itself are shared (blocks.js, generators.js, app.js).
import { categoryToolbox } from "../blocks.js";
import { chain, checkModel as checkWs4 } from "./ws4.js";

// Result tabs to show, in order.
export const TABS = ["steps", "training", "compare", "confusion", "mistakes", "practice", "upload", "card"];

export const COMPARE_CURVES = "val";

// The full training set again, so no size field; Optimizer keeps weight decay.
export const BLOCK_OPTIONS = { weightDecay: true };

export const STEPS = `
  <ol class="steps">
    <li><b>Build your best network.</b> The Model is empty, and this time it trains on the full training
      set. Use what you learned in Workspaces 1–4: which layers, which training settings, which ways to
      keep it from memorizing. A run has at most 3 epochs, so check how long each one takes.</li>
    <li><b>Check it.</b> Look at <b>Training</b>, <b>Confusion matrix</b>, <b>Worst mistakes</b> and the
      <b>Practice image</b>. Where is it still weak?</li>
    <li><b>Your image.</b> In <b>Your image</b>, choose any picture. You'll see each step of the encoding:
      your picture shrunk, reduced to a few colors, written in handwritten characters, and what your model
      reads back. Download the result to share it with the class.</li>
    <li><b>Model card.</b> <b>Model card</b> describes your latest finished run: what it is, how it was
      trained, how well it does and where it goes wrong. Read it, then download it.</li>
    <li><b>Decode the hidden class image.</b> Save &amp; Grade decodes the hidden class image with your
      latest finished run. You'll see only your accuracy: the picture stays secret until the class vote
      combines everyone's predictions. <span id="submit-note"></span></li>
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

// The pipeline with an empty Model: building the network is the student's job.
export const START_STATE = {
  blocks: {
    languageVersion: 0,
    blocks: [{
      x: 40, y: 40,
      ...chain(
        { type: "load_data" },
        { type: "model" },
        { type: "optimizer", fields: { KIND: "adam", LR: 0.001, WD: 0 } },
        { type: "loss", fields: { KIND: "ce" } },
        { type: "train_settings", fields: { EPOCHS: 2, BATCH: "128" } },
        { type: "evaluate" },
        { type: "confusion_matrix" },
        { type: "worst_mistakes" },
        { type: "decode_practice" },
      ),
    }],
  },
};

export const checkModel = checkWs4;
