# Physics-Informed Neural Network Model Comparison

An object-oriented Python project for training and comparing physics-informed neural networks (PINNs) for a one-dimensional boundary-value problem.

The program allows users to configure and compare multiple neural-network architectures by specifying parameters such as the number of hidden layers, neurons per layer, activation function, learning rate, and number of training epochs.

The trained PINNs are evaluated against the analytical solution, and the program produces numerical summaries and graphical comparisons of their performance.

## Physical Problem

The project considers a one-dimensional elastic rod of length $begin:math:text$L$end:math:text$ subjected to a linearly distributed load.

The displacement $begin:math:text$u\(x\)$end:math:text$ satisfies

$begin:math:display$
AEu\'\'\(x\) \+ cx \= 0\,
$end:math:display$

with boundary conditions

$begin:math:display$
u\(0\) \= 0\,
$end:math:display$

and

$begin:math:display$
u\'\(L\) \= 0\,
$end:math:display$

where:

- $begin:math:text$E$end:math:text$ is Young's modulus,
- $begin:math:text$A$end:math:text$ is the cross-sectional area,
- $begin:math:text$c$end:math:text$ is the load coefficient,
- $begin:math:text$L$end:math:text$ is the length of the rod.

The analytical solution is

$begin:math:display$
u\(x\)
\=
\\frac\{c\}\{6AE\}
\\left\(
\-x\^3 \+ 3L\^2x
\\right\)\.
$end:math:display$

## Nondimensionalisation

Direct training with the physical parameters leads to quantities on very different numerical scales. To obtain a better-conditioned PINN problem, the differential equation is nondimensionalised.

Using

$begin:math:display$
\\xi \= \\frac\{x\}\{L\}
$end:math:display$

and the characteristic displacement scale

$begin:math:display$
U \= \\frac\{cL\^3\}\{AE\}\,
$end:math:display$

the physical displacement is written as

$begin:math:display$
u\(x\) \= Uv\(\\xi\)\.
$end:math:display$

The dimensionless problem becomes

$begin:math:display$
v\'\'\(\\xi\) \+ \\xi \= 0\,
$end:math:display$

with

$begin:math:display$
v\(0\)\=0\,
\\qquad
v\'\(1\)\=0\.
$end:math:display$

Its analytical solution is

$begin:math:display$
v\(\\xi\)
\=
\\frac\{\-\\xi\^3 \+ 3\\xi\}\{6\}\.
$end:math:display$

The neural networks are trained on this dimensionless formulation, while predictions are converted back to physical displacement values for evaluation and visualization.

## Features

- Object-oriented implementation
- User-configurable neural-network architectures
- Comparison of multiple PINN models
- TensorFlow automatic differentiation
- Physics-informed PDE residual loss
- Boundary-condition losses
- Nondimensionalised training problem
- Analytical reference solution
- Mean and maximum absolute-error evaluation
- Training-loss comparison
- Graphical comparison of predicted and exact solutions

Supported activation functions are:

- `tanh`
- `sigmoid`
- `swish`
- `softplus`

## Project Structure

```text
.
├── main.py
├── rod_problem.py
├── pinn_configuration.py
├── pinn_model.py
├── model_comparison.py
├── user_input.py
├── requirements.txt
└── README.md
```

### `rod_problem.py`

Defines the physical rod problem, its nondimensionalisation, displacement scale, and analytical solution.

### `pinn_configuration.py`

Stores and validates the hyperparameters of individual PINN models.

### `pinn_model.py`

Constructs and trains individual PINNs using TensorFlow. It calculates the PDE and boundary-condition losses using automatic differentiation and evaluates predictions against the analytical solution.

### `model_comparison.py`

Trains, evaluates, and visualizes multiple PINN models and provides numerical comparisons of their performance.

### `user_input.py`

Handles interactive user input and creates validated model configurations.

### `main.py`

Provides the entry point for the program and coordinates model construction, training, evaluation, and visualization.

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run:

```bash
python main.py
```

The program interactively asks how many models should be compared and requests the configuration of each model.

For every model, the user can specify:

- model name,
- number of hidden layers,
- neurons per hidden layer,
- activation function,
- learning rate,
- number of training epochs.

For example:

```text
How many models do you want to compare? 2

Configuration for model 1
Model name [Model 1]: Small_PINN
Number of hidden layers: 2
Neurons per hidden layer: 20
Activation function (tanh, sigmoid, swish, softplus): swish
Learning rate, for example 0.001: 0.001
Number of epochs: 5000

Configuration for model 2
Model name [Model 2]: Deep_PINN
Number of hidden layers: 4
Neurons per hidden layer: 40
Activation function (tanh, sigmoid, swish, softplus): swish
Learning rate, for example 0.001: 0.001
Number of epochs: 5000
```

## Output

After training, the program reports a numerical comparison including:

- number of hidden layers,
- neurons per layer,
- number of trainable parameters,
- final training loss,
- mean absolute error,
- maximum absolute error.

It also produces four graphical comparisons:

1. PINN predictions and the analytical solution,
2. pointwise absolute errors,
3. training-loss histories,
4. mean and maximum absolute-error summary.

## Example Results

An example comparison between a smaller and a deeper PINN shows that both architectures can closely approximate the analytical displacement solution after nondimensionalisation.

Despite having substantially fewer trainable parameters, the smaller network can achieve accuracy comparable to the deeper architecture, illustrating that increasing network complexity does not necessarily improve performance for this relatively simple boundary-value problem.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Purpose

This project explores the application of physics-informed neural networks to differential equations while providing a flexible framework for studying how neural-network architecture and training parameters affect approximation accuracy and convergence behaviour.
