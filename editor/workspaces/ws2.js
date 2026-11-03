// Workspace 2: Convolutional network. Everything here is specific to this workspace; blocks,
// code generation and the page itself are shared (blocks.js, generators.js, app.js).
import { categoryToolbox } from "../blocks.js";
import { hiddenSizeProblems, misplacedReluHints, outputLayerProblems } from "../checks.js";

// Result tabs to show, in order.
export const TABS = ["steps", "training", "compare", "filters", "onestep", "practice"];

export const STEPS = `
  <ol class="steps">
    <li><b>Optional: get a baseline.</b> The Model starts empty this time. Rebuild your network from
      Workspace 1 (no Conv) and run it.
      <b>Compare runs</b> keeps your first run, so you'll see exactly what convolutions add.</li>
    <li><b>Build a convolutional network.</b> Building this one yourself is
      your first real challenge. Use the <b>Model</b> blocks you met in lecture: Conv, ReLU, MaxPool,
      Flatten and Linear. Some questions to think through:
      <ul>
        <li>Which blocks work on the 28×28 image, and which need one flat list of numbers?</li>
        <li>Where does each ReLU belong, and why?</li>
        <li>What must the last layer output?</li>
      </ul>
      The editor points out mistakes before you run. Runs are 2 epochs here, since convolutions
      are slower than Workspace 1's layers.</li>
    <li><b>Watch the filters learn.</b> Open <b>Filters</b> and drag the slider from "before training"
      to the end. Which filters end up finding edges? Which find the character's strokes?</li>
    <li><b>Take one training step apart.</b> Open <b>One step</b> and click <b>Next</b> through a
      single step on a character your model gets wrong: its guess, the loss, backpropagation,
      the weight changes, and the guess afterward.</li>
    <li><b>Go further.</b> Can you beat your first CNN? Try another convolution stage, more
      filters, or a different hidden layer. <b>Compare runs</b> shows your first and latest runs
      side by side, and <b>Practice image</b> shows what your model reads.</li>
    <li><b>Grade.</b> <span id="submit-note"></span></li>
  </ol>`;

export const TOOLBOX = categoryToolbox({
  data: ["load_data"],
  model: ["model", "conv", "relu", "maxpool", "flatten", "linear"],
  training: ["train"],
  evaluation: ["evaluate", "decode_practice"],
});

// Only one of each pipeline step is allowed.
export const MAX_INSTANCES = {
  load_data: 1, model: 1, train: 1, evaluate: 1, decode_practice: 1,
};

// The pipeline with an empty Model: building the network is the student's first challenge.
export const START_STATE = {
  blocks: {
    languageVersion: 0,
    blocks: [{
      type: "load_data", x: 40, y: 40,
      next: { block: {
        type: "model",
        next: { block: {
          type: "train", fields: { EPOCHS: 2 },
          next: { block: {
            type: "evaluate",
            next: { block: { type: "decode_practice" } },
          } },
        } },
      } },
    }],
  },
};

const IMAGE_LAYERS = ["conv", "maxpool"];

/**
 * Check the layers inside the Model block. Returns problems (block the run) and
 * hints (likely mistakes that still run), in plain language.
 */
export function checkModel(layers, cfg) {
  if (layers.length === 0) {
    return { problems: ["The Model is empty. Build your network inside it from the Model blocks."], hints: [] };
  }
  const problems = [];
  const hints = [];
  const types = layers.map((b) => b.type);
  const flat = types.indexOf("flatten");

  if (flat < 0) {
    problems.push("Add a Flatten block between the Conv part and the Linear layers: " +
      "Linear layers need one flat list of numbers.");
  } else if (types.lastIndexOf("flatten") !== flat) {
    problems.push("Use only one Flatten block.");
  }
  if (types.some((t, i) => t === "linear" && (flat < 0 || i < flat))) {
    problems.push("Linear layers go after Flatten.");
  }
  if (flat >= 0 && types.some((t, i) => IMAGE_LAYERS.includes(t) && i > flat)) {
    problems.push("Conv and MaxPool work on images, so they go before Flatten.");
  }
  if (types.filter((t) => t === "maxpool").length > 3) {
    problems.push("Use at most 3 MaxPool blocks: each halves the image (28 → 14 → 7 → 3), " +
      "and a fourth would leave almost nothing.");
  }
  problems.push(...outputLayerProblems(layers, cfg), ...hiddenSizeProblems(layers, cfg));

  if (!types.includes("conv")) {
    hints.push("This model has no Conv block yet, and this workspace is about convolutional networks.");
  } else if (types.some((t, i) => t === "conv" && types[i + 1] !== "relu")) {
    hints.push("Put a ReLU right after each Conv block. Without it, stacked filters can only " +
      "find simple patterns.");
  }
  hints.push(...misplacedReluHints(layers, ["linear", "conv"],
    "A ReLU only helps right after a Linear or Conv layer. Here it changes nothing."));
  return { problems, hints };
}
