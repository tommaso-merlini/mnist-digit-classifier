# MNIST Digit Classifier

A PyTorch implementation of a simple neural network for handwritten digit classification using the MNIST dataset. This project implements a straightforward feedforward neural network architecture to achieve digit recognition.

## Architecture

The model uses a simple yet effective architecture:
```python
Model = nn.Sequential(
    nn.Linear(784, 128),  # 28*28 -> 128
    nn.ReLU(),
    nn.Linear(128, 128),  # 128 -> 128
    nn.ReLU(),
    nn.Linear(128, 10)    # 128 -> 10 classes
)
```

## Dataset

The MNIST dataset consists of:
- 60,000 training images
- 10,000 testing images
- 28x28 grayscale images of handwritten digits (0-9)
- Normalized with mean=0.1307 and std=0.3081

## Features

- Simple feedforward neural network
- Cross-Entropy loss function
- SGD optimizer
- Training and testing loss visualization
- GPU support when available

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

## Model Details

- Input layer: 784 neurons (28x28 flattened images)
- Hidden layers: 2 layers of 128 neurons each
- Output layer: 10 neurons (one for each digit)
- Activation function: ReLU
- Batch size: 64
- Learning rate: 0.01

## Training

The model includes:
- Loss visualization for both training and test sets
- Batch processing
- One-hot encoded targets
- GPU acceleration when available

## Usage

```python
# Train the model
train_losses, test_losses = train(Model, epochs=1)

# Make predictions
x = image.view(-1, 28 ** 2).type(torch.float32)
prediction = torch.argmax(Model(x))
```

## Visualization

The project includes functionality to:
- Display training and test loss curves
- Visualize input images
- Show model predictions

## License

MIT
