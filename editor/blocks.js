// Block library for all workspaces. Each workspace module (workspaces/ws1.js, ...) picks
// which blocks go in its toolbox.
//
// Two kinds of stacking: pipeline steps ("Step") stack under Load data, and layers
// ("Layer") stack inside the Model block. Load data has no top connection, so it is
// always the start of the program; anything not attached under it is greyed out.

// Category colors by pipeline stage. Keep these the same in every workspace.
export const CATEGORY_COLORS = {
  data: "#4a7fb5",
  model: "#8e5bb5",
  training: "#c0763a",
  evaluation: "#3f9a5a",
};

const CATEGORY_NAMES = { data: "Data", model: "Model", training: "Training", evaluation: "Evaluation" };

/** A toolbox with one category per pipeline stage, e.g. categoryToolbox({ data: ["load_data"], ... }). */
export function categoryToolbox(blocksByStage) {
  return {
    kind: "categoryToolbox",
    contents: Object.entries(blocksByStage).map(([stage, types]) => ({
      kind: "category",
      name: CATEGORY_NAMES[stage],
      colour: CATEGORY_COLORS[stage],
      contents: types.map((type) => ({ kind: "block", type })),
    })),
  };
}

export function defineBlocks(cfg) {
  Blockly.common.defineBlocksWithJsonArray([
    {
      type: "load_data",
      message0: "Load data  (handwritten 0–9, A–Z)",
      nextStatement: "Step",
      colour: CATEGORY_COLORS.data,
      tooltip: "Loads the training and validation images. Every program starts here.",
    },
    {
      type: "model",
      message0: "Model  (layers run top to bottom)",
      message1: "%1",
      args1: [{ type: "input_statement", name: "LAYERS", check: "Layer" }],
      previousStatement: "Step",
      nextStatement: "Step",
      colour: CATEGORY_COLORS.model,
      tooltip: "A network made of the layers inside it. Each image passes through them in order.",
    },
    {
      type: "flatten",
      message0: "Flatten  (image → one list of numbers)",
      previousStatement: "Layer",
      nextStatement: "Layer",
      colour: CATEGORY_COLORS.model,
      tooltip: "Turns each image (or a Conv layer's filter outputs) into one long list of numbers, " +
               "which Linear layers need. A 28×28 image becomes 784 numbers.",
    },
    {
      type: "linear",
      message0: "Linear  outputs %1",
      args0: [{ type: "field_number", name: "OUTPUTS", value: cfg.hidden.default,
                min: 1, max: cfg.hidden.max, precision: 1 }],
      previousStatement: "Layer",
      nextStatement: "Layer",
      colour: CATEGORY_COLORS.model,
      tooltip: `Each output is a weighted sum of all the inputs. The last Linear layer needs ` +
               `${cfg.num_classes} outputs, one score per class. A hidden layer can have ` +
               `${cfg.hidden.min}–${cfg.hidden.max}.`,
    },
    {
      type: "conv",
      message0: "Conv  filters %1  (3×3)",
      args0: [{ type: "field_number", name: "FILTERS", value: cfg.conv.default,
                min: cfg.conv.min, max: cfg.conv.max, precision: 1 }],
      previousStatement: "Layer",
      nextStatement: "Layer",
      colour: CATEGORY_COLORS.model,
      tooltip: `Slides small 3×3 pattern detectors (filters) over the image. Each filter makes ` +
               `its own output image showing where its pattern appears. ${cfg.conv.min}–${cfg.conv.max} filters.`,
    },
    {
      type: "maxpool",
      message0: "MaxPool  (halve the size)",
      previousStatement: "Layer",
      nextStatement: "Layer",
      colour: CATEGORY_COLORS.model,
      tooltip: "Shrinks each filter output to half its width and height, keeping the strongest " +
               "value in every 2×2 patch. 28×28 → 14×14 → 7×7 → 3×3.",
    },
    {
      type: "relu",
      message0: "ReLU",
      previousStatement: "Layer",
      nextStatement: "Layer",
      colour: CATEGORY_COLORS.model,
      tooltip: "Keeps positive numbers and turns negative ones into 0. Put it after a hidden Linear layer or a Conv layer.",
    },
    {
      type: "train",
      message0: "Train  for %1 epochs",
      args0: [{ type: "field_number", name: "EPOCHS", value: cfg.epochs.default,
                min: 1, max: cfg.epochs.max, precision: 1 }],
      previousStatement: "Step",
      nextStatement: "Step",
      colour: CATEGORY_COLORS.training,
      tooltip: "Shows the model every training image once per epoch and adjusts its weights " +
               "(Adam optimizer, learning rate 0.001).",
    },
    {
      type: "evaluate",
      message0: "Evaluate  on validation images",
      previousStatement: "Step",
      nextStatement: "Step",
      colour: CATEGORY_COLORS.evaluation,
      tooltip: "Measures accuracy on images the model never trained on.",
    },
    {
      type: "decode_practice",
      message0: "Decode practice image",
      previousStatement: "Step",
      nextStatement: "Step",
      colour: CATEGORY_COLORS.evaluation,
      tooltip: "Reads all 1,024 characters of the practice image and colors each pixel by the predicted class.",
    },
  ]);
}
