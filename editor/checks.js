// Layer checks shared by workspace modules. Each takes the blocks inside the Model block
// (top to bottom) and returns plain-language messages.

/** The last layer must be Linear with one output per class. */
export function outputLayerProblems(layers, cfg) {
  const last = layers[layers.length - 1];
  if (!layers.some((b) => b.type === "linear")) return ["The Model needs at least one Linear layer."];
  if (last.type !== "linear" || last.getFieldValue("OUTPUTS") !== cfg.num_classes) {
    return [`The last layer must be Linear with ${cfg.num_classes} outputs (one score per class).`];
  }
  return [];
}

/** Every Linear layer except the last is hidden and must have an allowed size. */
export function hiddenSizeProblems(layers, cfg) {
  return layers.filter((b) => b.type === "linear").slice(0, -1)
    .map((b) => b.getFieldValue("OUTPUTS"))
    .filter((n) => n < cfg.hidden.min || n > cfg.hidden.max)
    .map((n) => `Hidden Linear layers need ${cfg.hidden.min}–${cfg.hidden.max} outputs (one has ${n}).`);
}

/** A ReLU that doesn't follow one of `after` (block types) changes nothing useful. */
export function misplacedReluHints(layers, after, message) {
  const misplaced = layers.some((b, i) =>
    b.type === "relu" && !after.includes(layers[i - 1]?.type) && i < layers.length - 1);
  return misplaced ? [message] : [];
}
