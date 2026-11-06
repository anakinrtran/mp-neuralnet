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

// The generalization steps, shared with Workspace 5 (which keeps Workspace 4's content).
export const GENERALIZATION_STEPS = `
    <li><b>Overfit on purpose.</b> This model is big, and it trains on only 100 characters per class.
      Click <b>▶ Run</b> (about a minute) and watch <b>Training</b>: training accuracy and validation
      accuracy are drawn on the same chart. When do they pull apart? Where does each one end up? What is
      the model doing with its training characters? Keep this run: it's the one to beat.</li>
    <li><b>Try each fix alone.</b> Add one fix to the starting blocks, run, and undo it before trying the
      next. In <b>Compare runs</b>, tick the runs to overlay their validation accuracy with your first
      run. For each fix, ask two things: does it close the gap between training and validation, and does it
      raise the <i>top</i> of the validation curve?
      <ul>
        <li><b>Dropout</b> switches off a random share of units (p) in the layer before it, at every
          training step, so no single unit can memorize a character. Which layer of this model has by far
          the most weights to memorize with? Dropout belongs right after it (after its ReLU). How big a
          share can you switch off and still leave enough units to learn?</li>
        <li><b>Weight decay</b> (on the Optimizer) pulls every weight a little toward zero at each step, so
          the model can't lean on a few huge weights. Small values such as 0.001 are typical: does it
          change the gap much here?</li>
        <li><b>Augment training images</b> shows the model a slightly changed copy of each training
          character every time it sees it, so 100 characters act like many more. The question that
          decides which changes are safe: after the change, is it still the same character? Would a
          person write a 7 slightly tilted, or a little off-center?</li>
        <li><b>The mirroring trap.</b> Now tick <i>mirror left↔right</i> as well. What happens to
          validation accuracy? Picture a few characters (E, J, 3, 7) in a mirror: why might that be?</li>
        <li><b>Early stopping</b> watches validation accuracy and stops once it hasn't improved for
          <i>patience</i> epochs, then keeps the weights from the best epoch. Look at your first run's
          validation curve: in which epoch did it peak? Can early stopping make that peak higher, or only
          keep you from sliding past it?</li>
      </ul></li>
    <li><b>Visualize your model's mistakes.</b> Add <b>Confusion matrix</b> and <b>Worst mistakes</b> at the
      end of your program and run it. In <b>Confusion matrix</b>, find your model's three worst
      confusions: which are look-alike pairs that even you would struggle with? <b>Worst mistakes</b>
      shows the wrong guesses your model was most sure about: would a person get them right? Which kind
      of mistake could a fix from step 2 still remove?</li>
    <li><b>Combine.</b> Build the best model you can on the small training set. Before each run, ask:
      <ul>
        <li>Which fixes raised the top of the validation curve, and which only changed where it stopped?
          Start from the ones that raised it.</li>
        <li>With augmentation, the model sees new versions of every character each epoch, so it can keep
          improving for longer. Is your validation curve still climbing at the last epoch? Then more
          epochs (up to 60) can help, and early stopping makes a long run safe because it keeps the best
          epoch. Every 20 epochs take about a minute.</li>
      </ul></li>
`;

export const STEPS = `
  <ol class="steps">
${GENERALIZATION_STEPS}    <li><b>Grade.</b> Save &amp; Grade decodes a held-out test image whose characters aren't in this
      workspace, so you can't tune on it.
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
