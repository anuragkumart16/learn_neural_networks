# Learn Neural Networks From Scratch

A hands-on implementation of a **Perceptron from scratch in Python**, built to understand the mathematics and mechanics behind neural networks rather than relying on libraries such as TensorFlow or PyTorch.

The project currently implements a single perceptron and trains it to solve the **AND logical gate**.

## What I Learned

The project was built by breaking a perceptron down into its fundamental components:

* Inputs
* Weights
* Bias
* Weighted sum
* Activation function
* Learning rate
* Error
* Weight updates
* Bias updates
* Training
* Prediction
* Epochs

The basic computation performed by the perceptron is:

[
z = \sum_i w_i x_i + b
]

The weighted sum is then passed through an activation function:

[
\hat{y} = f(z)
]

For the current implementation, a step activation function is used:

```python
def step(x):
    return 1 if x >= 0 else 0
```

## How Learning Works

The perceptron compares its prediction with the actual output:

[
error = y_{actual} - y_{predicted}
]

The weights are then updated using:

[
\Delta w_i = \eta \times error \times x_i
]

and:

[
w_i = w_i + \Delta w_i
]

The bias is updated using:

[
\Delta b = \eta \times error
]

and:

[
b = b + \Delta b
]

where (\eta) is the learning rate.

In simple terms:

```text
Input
  ↓
Weighted Sum
  ↓
Activation Function
  ↓
Prediction
  ↓
Calculate Error
  ↓
Update Weights
  ↓
Update Bias
  ↓
Repeat
```

## Perceptron Implementation

The `Perceptron` class contains the core functionality:

```python
class Perceptron:
    def __init__(self, no_of_inputs, activation_function, learning_rate=0.1):
        ...
```

It initializes:

* Random weights
* Bias
* Activation function
* Learning rate

The weighted sum is calculated with:

```python
def calulate_weighted_sum(self, inputs):
    weighted_sum = 0

    for index in range(len(inputs)):
        weighted_sum += inputs[index] * self.weights[index]

    weighted_sum += self.bias

    return weighted_sum
```

Training is performed using:

```python
def train(self, inputs, y_actual):
    weighted_sum = self.calulate_weighted_sum(inputs)
    output = self.activation_function(weighted_sum)

    error = y_actual - output

    for index in range(len(inputs)):
        change_in_weight = (
            error * self.learning_rate * inputs[index]
        )

        self.weights[index] += change_in_weight

    self.bias += error * self.learning_rate

    return output
```

Prediction uses the same forward computation without changing the parameters.

## Training the AND Gate

The training dataset is:

| Input 1 | Input 2 | Expected Output |
| ------: | ------: | --------------: |
|       0 |       0 |               0 |
|       0 |       1 |               0 |
|       1 |       0 |               0 |
|       1 |       1 |               1 |

The perceptron is trained over multiple **epochs**.

An epoch is one complete pass through the training dataset:

```text
Epoch 1 → [0,0], [0,1], [1,0], [1,1]
Epoch 2 → [0,0], [0,1], [1,0], [1,1]
...
```

Repeated passes allow the perceptron to gradually adjust its weights and bias.

## Result

After training, the perceptron successfully learned the AND relationship.

For example, the final learned parameters were approximately:

```text
w1 = 0.5394
w2 = 0.1056
b  = -0.6
```

For the input:

```text
[1, 0]
```

the weighted sum becomes approximately:

[
z=(1)(0.5394)+(0)(0.1056)-0.6
]

[
z\approx -0.0606
]

The step activation therefore produces:

```text
0
```

which is the correct AND-gate output.

## Project Structure

The project currently follows this structure:

```text
learn_neural_networks/
├── perceptron.py
├── tests/
│   └── and_gate_test.py
└── venv/
```

The repository also contains a Python virtual environment for running the project.

## Running the Project

From the project root:

```bash
python -m tests.and_gate_test
```

Running the test this way avoids the module import issue encountered when executing the test file directly.

## Current Limitations

This is currently a **single perceptron**, not a multi-layer neural network.

It can learn linearly separable problems such as:

* AND
* OR

but cannot solve problems such as **XOR** with a single perceptron.

The current implementation also uses a simple step activation function and the perceptron learning rule rather than gradient-based backpropagation used in modern multi-layer neural networks.

## Next Steps

The natural progression for this project is:

```text
Single Perceptron
       ↓
AND Gate
       ↓
OR Gate
       ↓
XOR Problem
       ↓
Multiple Perceptrons
       ↓
Layers
       ↓
Multi-Layer Neural Network
       ↓
Backpropagation
       ↓
Gradient Descent
       ↓
MNIST
       ↓
PyTorch / TensorFlow
```

The goal is to progressively build the neural-network machinery from the underlying mathematics instead of treating neural networks as a black box.

## Key Takeaway

A perceptron is fundamentally a mathematical function with **learnable parameters**:

[
\boxed{\hat y=f(w_1x_1+w_2x_2+\cdots+w_nx_n+b)}
]

Learning consists of repeatedly changing those parameters based on the prediction error.

This project is the first step toward implementing a complete neural network from scratch.
