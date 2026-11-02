// Workspace 2: Convolutional network. Everything here is specific to this workspace; blocks,
// code generation and the page itself are shared (blocks.js, generators.js, app.js).
import { categoryToolbox } from "../blocks.js";
import { hiddenSizeProblems, misplacedReluHints, outputLayerProblems } from "../checks.js";

// Result tabs to show, in order.
export const TABS = ["steps", "training", "compare", "practice"];

export const STEPS = `
  <ol class="steps">
    <li><b>Start where Workspace 1 ended.</b> Click <b>▶ Run</b> to train the hidden-layer model
      (2 epochs now: convolutions are slower, so every run here is shorter).</li>
    <li><b>Add a convolution.</b> From <b>Model</b>, drag <b>Conv</b>, then <b>ReLU</b>, then
      <b>MaxPool</b> into the Model above Flatten, so it reads
      <b>Conv 16 → ReLU → MaxPool → Flatten → Linear 128 → ReLU → Linear 36</b>.
      Run it, then open <b>Compare runs</b>.</li>
    <li><b>Go deeper.</b> Add a second <b>Conv 32 → ReLU → MaxPool</b> right after the first one.
      Run again and open <b>Practice image</b>.</li>
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

// Workspace 1's final model, with Decode practice image already attached.
export const START_STATE = {
  blocks: {
    languageVersion: 0,
    blocks: [{
      type: "load_data", x: 40, y: 40,
      next: { block: {
        type: "model",
        inputs: { LAYERS: { block: {
          type: "flatten",
          next: { block: {
            type: "linear", fields: { OUTPUTS: 128 },
            next: { block: {
              type: "relu",
              next: { block: { type: "linear", fields: { OUTPUTS: 36 } } },
            } },
          } },
        } } },
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
    return { problems: ["The Model is empty. Put Conv, Flatten and Linear blocks inside it."], hints: [] };
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
    hints.push("This model has no Conv block yet. Add Conv → ReLU → MaxPool above Flatten.");
  } else if (types.some((t, i) => t === "conv" && types[i + 1] !== "relu")) {
    hints.push("Put a ReLU right after each Conv block. Without it, stacked filters can only " +
      "find simple patterns.");
  }
  hints.push(...misplacedReluHints(layers, ["linear", "conv"],
    "A ReLU only helps right after a Linear or Conv layer. Here it changes nothing."));
  return { problems, hints };
}
