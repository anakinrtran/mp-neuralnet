// Workspace 1: Basic network. Everything here is specific to this workspace; blocks,
// code generation and the page itself are shared (blocks.js, generators.js, app.js).
import { categoryToolbox } from "../blocks.js";
import { hiddenSizeProblems, misplacedReluHints, outputLayerProblems } from "../checks.js";

// Result tabs to show, in order.
export const TABS = ["steps", "training", "compare", "weights", "practice"];

export const STEPS = `
  <ol class="steps">
    <li><b>Run the linear model.</b> Click <b>▶ Run</b>. In <b>Training</b>, watch the loss go down
      and the validation accuracy go up.</li>
    <li><b>See how it classifies.</b> Open <b>Weights → Class templates</b>. Each picture shows
      what the model looks for in one character.</li>
    <li><b>Add a hidden layer.</b> From <b>Model</b>, drag a <b>Linear</b> block (it starts at
      128 outputs) and then a <b>ReLU</b> block into the Model, so it reads
      <b>Flatten → Linear 128 → ReLU → Linear 36</b>. Run again, then open <b>Compare runs</b>.</li>
    <li><b>Decode the practice image.</b> From <b>Evaluation</b>, attach <b>Decode practice image</b>
      at the bottom and run. Open <b>Practice image</b> to see what your model read.</li>
    <li><b>Grade.</b> <span id="submit-note"></span></li>
  </ol>`;

export const TOOLBOX = categoryToolbox({
  data: ["load_data"],
  model: ["model", "flatten", "linear", "relu"],
  training: ["train"],
  evaluation: ["evaluate", "decode_practice"],
});

// Only one of each pipeline step is allowed.
export const MAX_INSTANCES = {
  load_data: 1, model: 1, train: 1, evaluate: 1, decode_practice: 1,
};

// What students see when they first open the workspace (or click Reset blocks).
export const START_STATE = {
  blocks: {
    languageVersion: 0,
    blocks: [{
      type: "load_data", x: 40, y: 40,
      next: { block: {
        type: "model",
        inputs: { LAYERS: { block: {
          type: "flatten",
          next: { block: { type: "linear", fields: { OUTPUTS: 36 } } },
        } } },
        next: { block: {
          type: "train", fields: { EPOCHS: 5 },
          next: { block: { type: "evaluate" } },
        } },
      } },
    }],
  },
};

/**
 * Check the layers inside the Model block. Returns problems (block the run) and
 * hints (likely mistakes that still run), in plain language.
 */
export function checkModel(layers, cfg) {
  if (layers.length === 0) {
    return { problems: ["The Model is empty. Put Flatten and Linear blocks inside it."], hints: [] };
  }
  const problems = [];
  if (layers[0].type !== "flatten") {
    problems.push("Put Flatten first in the Model: Linear layers need each image as a flat list of numbers.");
  }
  problems.push(...outputLayerProblems(layers, cfg), ...hiddenSizeProblems(layers, cfg));
  const hints = misplacedReluHints(layers, ["linear"],
    "A ReLU only helps right after a Linear layer. Here it changes nothing " +
    "(after Flatten, every number is already 0 or more). Put a Linear block above it to make a hidden layer.");
  return { problems, hints };
}
