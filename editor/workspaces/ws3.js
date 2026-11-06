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
      training blocks so the model learns again. Leave the Model alone for now.
      <ul>
        <li>Change a learning rate by factors of ten (10, 1, 0.1, 0.01, 0.001), not by small tweaks:
          a run takes about a minute, and a small tweak rarely shows.</li>
      </ul></li>
    <li><b>Compare optimizers.</b> Try SGD + momentum and Adam. Each needs its own learning rate: start
      SGD + momentum at 0.1 and Adam at 0.001, one run each. In <b>Compare runs</b>, overlay their
      loss curves with your plain SGD run. Which drops fastest? Which is the bumpiest?</li>
    <li><b>Batch size</b> (one or two runs). Try a much smaller and a much larger batch. What changes:
      the run time, how bumpy the loss curve is, the final accuracy?</li>
    <li><b>Loss function</b> (one run). Swap cross-entropy for MSE. What happens to the loss curve and
      to accuracy? Why might that be?</li>
    <li><b>Stabilize.</b> Two tools that make training steadier. Try each as a pair of runs that differ
      in that one thing.
      <ul>
        <li><b>BatchNorm</b> goes right after a Conv, before its ReLU: Conv → BatchNorm → ReLU. (It can
          also go after the hidden Linear, before its ReLU.) Add it to plain SGD at 0.1, which you
          already ran without it. In <b>Compare runs</b>, how do the two loss curves differ? Where does
          validation accuracy end up?</li>
        <li><b>Scheduler</b> goes between the Optimizer and Train. It changes the learning rate during the
          run: big steps early, small steps late. Add cosine to your best Adam or SGD + momentum run and
          change nothing else. Its effect is easiest to see in <b>Gradients</b>: what happens to the
          update sizes toward the end of the run? A scheduler shrinks your steps, so it helps most when
          the learning rate is on the high side. If it made things worse, try it with a learning rate
          ten times higher.</li>
      </ul></li>
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
