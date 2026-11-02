// Toolbox: one category per pipeline stage, colored to match its blocks.
import { CATEGORY_COLORS } from "./blocks.js";

export const TOOLBOX = {
  kind: "categoryToolbox",
  contents: [
    {
      kind: "category", name: "Data", colour: CATEGORY_COLORS.data,
      contents: [{ kind: "block", type: "load_data" }],
    },
    {
      kind: "category", name: "Model", colour: CATEGORY_COLORS.model,
      contents: [
        { kind: "block", type: "model" },
        { kind: "block", type: "flatten" },
        { kind: "block", type: "linear" },
        { kind: "block", type: "relu" },
      ],
    },
    {
      kind: "category", name: "Training", colour: CATEGORY_COLORS.training,
      contents: [{ kind: "block", type: "train" }],
    },
    {
      kind: "category", name: "Evaluation", colour: CATEGORY_COLORS.evaluation,
      contents: [
        { kind: "block", type: "evaluate" },
        { kind: "block", type: "decode_practice" },
      ],
    },
  ],
};
