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

generator.forBlock.flatten = (block) =>
  block.getPreviousBlock()?.type === "model"
    ? "nn.Flatten(),  # turn each 28x28 image into a list of 784 numbers\n"
    : "nn.Flatten(),  # turn the filter outputs into one list of numbers\n";

generator.forBlock.conv = (block) => {
  const filters = block.getFieldValue("FILTERS");
  const comment = generator.firstConv ? `  # ${filters} filters, each a 3x3 pattern detector` : "";
  generator.firstConv = false;
  return `nn.LazyConv2d(${filters}, kernel_size=3, padding=1),${comment}\n`;
};

generator.forBlock.maxpool = () => {
  const comment = generator.firstPool ? "  # keep the strongest value in each 2x2 patch" : "";
  generator.firstPool = false;
  return `nn.MaxPool2d(2),${comment}\n`;
};

generator.forBlock.linear = (block) => {
  const outputs = block.getFieldValue("OUTPUTS");
  const comment = generator.firstLinear ? "  # Lazy: works out its input size by itself" : "";
  generator.firstLinear = false;
  return `nn.LazyLinear(${outputs}),${comment}\n`;
};

generator.forBlock.relu = () => "nn.ReLU(),\n";

generator.forBlock.batchnorm = (block) => {
  // BatchNorm2d works on images (before Flatten), BatchNorm1d on lists (after it)
  let flat = false;
  for (let b = block.getPreviousBlock(); b && b.type !== "model"; b = b.getPreviousBlock()) {
    if (b.type === "flatten") flat = true;
  }
  const comment = generator.firstNorm ? "  # keep the numbers in a steady range" : "";
  generator.firstNorm = false;
  return `nn.LazyBatchNorm${flat ? "1d" : "2d"}(),${comment}\n`;
};

const OPTIMIZERS = {
  sgd: (lr) => `optim.SGD(model.parameters(), lr=${lr})`,
  momentum: (lr) => `optim.SGD(model.parameters(), lr=${lr}, momentum=0.9)`,
  adam: (lr) => `optim.Adam(model.parameters(), lr=${lr})`,
};

generator.forBlock.optimizer = (block) =>
  `optimizer = ${OPTIMIZERS[block.getFieldValue("KIND")](block.getFieldValue("LR"))}\n`;

generator.forBlock.scheduler = (block) => {
  // the harness steps the scheduler after every batch, so its length is counted in steps
  const train = block.workspace.getBlocksByType("train_settings", false)[0];
  const epochs = train ? train.getFieldValue("EPOCHS") : 1;
  const batch = Number(train ? train.getFieldValue("BATCH") : generator.cfg.batch_default);
  const perEpoch = Math.ceil(generator.cfg.train_size / batch);
  const n = (v) => v.toLocaleString("en-US");
  return block.getFieldValue("KIND") === "step"
    ? `scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=${perEpoch}, gamma=0.1)` +
      `  # learning rate x0.1 after each epoch (${n(perEpoch)} steps)\n`
    : `scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=${epochs * perEpoch})` +
      `  # learning rate glides to 0 over ${n(epochs * perEpoch)} steps\n`;
};

generator.forBlock.loss = (block) =>
  block.getFieldValue("KIND") === "mse"
    ? "loss_fn = nn.MSELoss()  # compares the probabilities with the right answer\n\n"
    : "loss_fn = nn.CrossEntropyLoss()\n\n";

generator.forBlock.train_settings = (block) => {
  const has = (type) => block.workspace.getBlocksByType(type, false).length > 0;
  const extra = ["optimizer=optimizer", "loss_fn=loss_fn", has("scheduler") && "scheduler=scheduler"]
    .filter(Boolean).join(", ");
  return `train(model, train_data, val_data, epochs=${block.getFieldValue("EPOCHS")}, ` +
    `batch_size=${block.getFieldValue("BATCH")},\n      ${extra})\n`;
};

generator.forBlock.train = (block) =>
  `train(model, train_data, val_data, epochs=${block.getFieldValue("EPOCHS")})\n`;

generator.forBlock.evaluate = () => "evaluate(model, val_data)\n";

generator.forBlock.decode_practice = () => "decode_practice(model)\n";

function startBlock(workspace) {
  return workspace.getBlocksByType("load_data", false)[0] || null;
}

/** Python code for the program that starts at Load data (unattached blocks are ignored). */
export function generateCode(workspace, cfg) {
  const start = startBlock(workspace);
  if (!start) return HEADER + "# Add a Load data block to start your program.\n";
  generator.init(workspace);
  generator.cfg = cfg;
  generator.firstLinear = true;
  generator.firstConv = true;
  generator.firstPool = true;
  generator.firstNorm = true;
  const body = generator.blockToCode(start).trimEnd() + "\n";
  const header = /\boptim\./.test(body) ? HEADER.replace("from torch import nn", "from torch import nn, optim") : HEADER;
  return header + body;
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
  // From Workspace 3: "train_settings", with Optimizer, Loss and an optional Scheduler above it
  const trainType = workspace.getBlocksByType("train_settings", false).length ? "train_settings" : "train";
  const trainAt = at(trainType);
  if (trainAt < 0) problems.push("Add a Train block after the Model so it can learn.");
  if (at("model") >= 0 && trainAt >= 0 && trainAt < at("model")) {
    problems.push("Move Train below the Model block: the model has to exist before it can be trained.");
  }
  if (trainType === "train_settings" && trainAt >= 0) {
    for (const [type, name] of [["optimizer", "Optimizer"], ["loss", "Loss"]]) {
      if (at(type) < 0) problems.push(`Train needs a ${name} block above it.`);
      else if (at(type) > trainAt) problems.push(`Move the ${name} block above Train.`);
    }
    if (at("optimizer") >= 0 && at("model") > at("optimizer")) {
      problems.push("Move the Optimizer below the Model: it adjusts the model's weights, so the model must come first.");
    }
    if (at("scheduler") >= 0 && (at("scheduler") < at("optimizer") || at("scheduler") > trainAt)) {
      problems.push("Put the Scheduler between the Optimizer and Train: it changes the optimizer's learning rate.");
    }
  }
  for (const type of ["evaluate", "decode_practice"]) {
    if (at(type) >= 0 && trainAt >= 0 && at(type) < trainAt) {
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
  if (at("decode_practice") < 0 && trainAt >= 0) {
    notes.push("Add Decode practice image (in Evaluation) at the end to see what your model reads.");
  }
  return { problems, hints, notes };
}
