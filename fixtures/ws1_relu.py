from torch import nn
from harness import load_data, train, evaluate, decode_practice

# Load the handwritten character images
train_data, val_data = load_data()

model = nn.Sequential(
    nn.Flatten(),  # turn each 28x28 image into a list of 784 numbers
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(36),
)

train(model, train_data, val_data, epochs=5)
evaluate(model, val_data)
decode_practice(model)
