// Turns blocks into the PyTorch code students see and run, and checks the program
// for mistakes before it runs.

const generator = new Blockly.CodeGenerator("PyTorch");
generator.INDENT = "    ";

// A statement block's code is followed by the code of the blocks stacked under it.
generator.scrub_ = function (block, code, thisOnly) {
  const next = block.nextConnection && block.nextConnection.targetBlock();
  return thisOnly ? code : code + generator.blockToCode(next);
};

const HEADER =
  "from torch import nn\n" +
  "from harness import load_data, train, evaluate, decode_practice\n\n";

generator.forBlock.load_data = () =>
  "# Load the handwritten character images\n" +
  "train_data, val_data = load_data()\n\n";

generator.forBlock.model = (block) =>
  "model = nn.Sequential(\n" + generator.statementToCode(block, "LAYERS") + ")\n\n";

generator.forBlock.flatten = () =>
  "nn.Flatten(),  # turn each 28x28 image into a list of 784 numbers\n";

generator.forBlock.linear = (block) => {
  const outputs = block.getFieldValue("OUTPUTS");
  const comment = generator.firstLinear ? "  # Lazy: works out its input size by itself" : "";
  generator.firstLinear = false;
  return `nn.LazyLinear(${outputs}),${comment}\n`;
};

generator.forBlock.relu = () => "nn.ReLU(),\n";

generator.forBlock.train = (block) =>
  `train(model, train_data, val_data, epochs=${block.getFieldValue("EPOCHS")})\n`;

generator.forBlock.evaluate = () => "evaluate(model, val_data)\n";

generator.forBlock.decode_practice = () => "decode_practice(model)\n";

function startBlock(workspace) {
  return workspace.getBlocksByType("load_data", false)[0] || null;
}

/** Python code for the program that starts at Load data (unattached blocks are ignored). */
export function generateCode(workspace) {
  const start = startBlock(workspace);
  if (!start) return HEADER + "# Add a Load data block to start your program.\n";
  generator.init(workspace);
  generator.firstLinear = true;
  return HEADER + generator.blockToCode(start).trimEnd() + "\n";
}

function stackOf(first) {
  const blocks = [];
  for (let b = first; b; b = b.getNextBlock()) blocks.push(b);
  return blocks;
}

/**
 * Check the program before running it. checkModel is the workspace's check of the
 * layers inside the Model block.
 * Returns problems (block the run), hints (likely mistakes that still run) and notes
 * (next steps), all in plain language.
 */
export function checkProgram(workspace, cfg, checkModel) {
  const problems = [];
  const hints = [];
  const notes = [];
  const start = startBlock(workspace);
  if (!start) {
    return { problems: ["Start your program with a Load data block (in Data)."], hints, notes };
  }

  const steps = stackOf(start).map((b) => b.type);
  const at = (type) => steps.indexOf(type);
  if (at("model") < 0) problems.push("Add a Model block under Load data, with layers inside it.");
  if (at("train") < 0) problems.push("Add a Train block after the Model so it can learn.");
  if (at("model") >= 0 && at("train") >= 0 && at("train") < at("model")) {
    problems.push("Move Train below the Model block: the model has to exist before it can be trained.");
  }
  for (const type of ["evaluate", "decode_practice"]) {
    if (at(type) >= 0 && at("train") >= 0 && at(type) < at("train")) {
      const name = type === "evaluate" ? "Evaluate" : "Decode practice image";
      problems.push(`Move ${name} below Train: it needs a trained model.`);
    }
  }

  const modelBlock = workspace.getBlocksByType("model", false)[0];
  if (modelBlock && at("model") >= 0) {
    const found = checkModel(stackOf(modelBlock.getInputTargetBlock("LAYERS")), cfg);
    problems.push(...found.problems);
    hints.push(...found.hints);
  }

  const loose = workspace.getTopBlocks(false).filter((b) => b !== start);
  if (loose.length) notes.push("Greyed-out blocks aren't connected to the program, so they won't run.");
  if (at("decode_practice") < 0 && at("train") >= 0) {
    notes.push("Add Decode practice image (in Evaluation) at the end to see what your model reads.");
  }
  return { problems, hints, notes };
}
