# Digit Classifier: Neural Network from Scratch (NumPy)

A simple neural network framework built with only NumPy and Python, applied to handwritten digit
classification (Kaggle *Digit Recognizer*, MNIST format: 28x28 images, 10 classes).

## What is implemented
- Dense layers with batch (matrix) processing
- ReLU, Sigmoid and Tanh hidden activations, Softmax output
- Cross-entropy loss computed from logits (numerically stable)
- Forward propagation and backpropagation, verified with gradient checking
- Mini-batch gradient descent, with Momentum and Adam optimizers
- Experiments on optimizers, activation functions, and numbers of layers and neurons
- A separate 3-layer, one-neuron-per-layer gradient descent demo (chain rule by hand)

## Results (network 784-128-64-10, ReLU, Adam, 20 epochs)
- Test accuracy: 0.9651
- Cost on dev set: 0.2108
- Cost on test set: 0.2447

## Files
- `Digit_Classifier_Neural_Network.ipynb`: the full notebook, with outputs
- `digit_model.npz`: trained weights, normalization statistics and settings

## How to run
1. Download `train.csv` from the Kaggle *Digit Recognizer* competition (it is not included in this repository).
2. Open the notebook in Google Colab and run the cells from top to bottom.
3. When asked, upload `train.csv`.

## Load the saved model
```python
data = np.load("digit_model.npz")
net = NeuralNetwork(data["layer_dims"].tolist(), data["hidden_activation"].item())
for k in net.params:
    net.params[k] = data[k]
```

## Glossary: abbreviations, names and symbols

### Abbreviations and terms
| Short name | Stands for | What it is |
|---|---|---|
| `np` | NumPy (Numerical Python) | Array and math library that runs the network |
| `pd` | pandas | Reads the CSV and builds tables |
| `plt` | matplotlib.pyplot | Draws plots and images |
| `os`, `shutil` | operating system, shell utilities | Check, rename and delete files |
| CSV | Comma-Separated Values | Text file format of `train.csv` |
| MNIST | Modified National Institute of Standards and Technology | The handwritten-digit dataset |
| ReLU | Rectified Linear Unit | `max(0, x)` |
| tanh | hyperbolic tangent | Squashes values into (-1, 1) |
| SGD | Stochastic Gradient Descent | Gradient descent on mini-batches |
| Adam | Adaptive Moment Estimation | Momentum plus adaptive step sizes |
| RMSprop | Root Mean Square Propagation | The squared-gradient idea inside Adam |
| He init | He initialization | Starting weights scaled by sqrt(2 / n_in) |
| FP / BP | Forward / Back Propagation | Prediction pass and gradient pass |
| GD | Gradient Descent | Update rule: w = w - lr * gradient |
| NaN | Not a Number | An invalid result (for example 0/0) |

### Naming rules used in the code (Andrew Ng's notation)
| Pattern | Meaning | Examples |
|---|---|---|
| Capital letter | A matrix or batch of values | `X`, `W`, `Z`, `A` |
| Lowercase `y`, `t` | Labels or targets | `y_train`, `t` |
| `d` + name | Gradient of the cost with respect to that name | `dW`, `db`, `dZ`, `dA` |
| `_prev` | From the previous layer | `A_prev`, `dA_prev` |
| `_train`, `_dev`, `_test` | Which data split it belongs to | `X_train`, `J_dev`, `y_test` |
| `J` | The cost (average loss) | `J_train`, `J_plus` |
| Number suffix | Layer number | `W1`, `b2`, `z3` |
| `_backward` | The derivative version of a function | `relu_backward` |
| `self.` | A value stored inside a class | `self.params`, `self.lr` |

### Math symbols and their names in the code
| Math | Code | Meaning |
|---|---|---|
| alpha | `lr` | Learning rate (step size) |
| epsilon | `eps` | A tiny number |
| beta | `beta`, `b1`, `b2` | Averaging factor (momentum, Adam) |
| J | `J_train`, `J_dev` | Cost |
| dJ/dw | `dW`, `grads["W1"]` | Gradient |
| y-hat | `probs`, `predict_proba` | Predicted probabilities |
| Sigma | `np.sum`, `.sum()` | Sum |
| w*x + b | `A @ W + b` | Linear step of a layer |

### Main variables
| Name | Meaning | Shape or value |
|---|---|---|
| `df` | Table holding the whole CSV | (42000, 785) |
| `X_train`, `y_train` | Training images and labels (learn from these) | (29400, 784), (29400,) |
| `X_dev`, `y_dev` | Dev images and labels (tune and compare) | about 15% of the data |
| `X_test`, `y_test` | Test images and labels (final score, used once) | about 15% of the data |
| `mean`, `std` | Per-pixel average and spread, from training data only | (784,) each |
| `classes` | The distinct labels | 0 to 9 |
| `n_features`, `n_classes` | Inputs per image, number of classes | 784, 10 |
| `Z` | Layer output before the activation (logits at the last layer) | (batch, units) |
| `A` | Layer output after the activation | (batch, units) |
| `A_prev` | Input to a layer | (batch, n_in) |
| `W1`, `W2`, `W3` | Weights of layers 1, 2, 3 | (784,128), (128,64), (64,10) |
| `b1`, `b2`, `b3` | Biases of layers 1, 2, 3 | (1,128), (1,64), (1,10) |
| `dW`, `db`, `dZ`, `dA_prev` | Gradients (same shape as the value they belong to) | |
| `caches` | Values saved in the forward pass for backprop | list of (A_prev, Z) |
| `params`, `grads` | Dictionaries of all parameters and their gradients | |
| `layer_dims` | Layer sizes | [784, 128, 64, 10] |
| `lr` | Learning rate | 0.001 (Adam), 0.1 (SGD) |
| `beta` | Momentum factor | 0.9 |
| `epochs` | Full passes through the training data | 20 |
| `batch_size` | Examples per mini-batch | 128 |
| `hist` | Per-epoch results: `J_train`, `J_dev`, `acc_train`, `acc_dev` | dictionary |
| `J_train`, `J_dev` | Cost on the training / dev data | |
| `acc_train`, `acc_dev` | Accuracy on the training / dev data | |
| `cm` | Confusion matrix (rows = true digit, columns = predicted digit) | (10, 10) |
| `eps` (gradient check) | Tiny nudge used for the numerical gradient | 1e-5 |
| `summary`, `table` | Results of the bonus experiments | dictionary, table |

### Main functions and classes
| Name | What it does |
|---|---|
| `load_and_prepare` | Loads the CSV, splits 70/15/15, normalizes with training statistics |
| `relu`, `sigmoid`, `tanh`, `softmax` | Activation functions (each hidden one has a `_backward` version) |
| `ACTIVATIONS` | Dictionary linking a name such as "relu" to its forward and backward functions |
| `dense_forward`, `dense_backward` | Forward and backward pass of one dense layer |
| `cost_from_logits` | Cross-entropy cost from the raw scores |
| `cost_backward_from_logits` | Starting gradient of backprop: softmax(Z) - one_hot(y) |
| `NeuralNetwork` | The network: `forward`, `backward`, `predict_proba`, `predict` |
| `gradient_check` | Compares backprop with numerical gradients |
| `SGD`, `Adam` | Optimizers (SGD with beta greater than 0 is momentum) |
| `train` | Mini-batch training loop |
| `confusion_matrix` | Builds the 10 x 10 table of true vs predicted digits |
| `forward_1neuron`, `backward_1neuron`, `cost_1neuron` | The separate one-neuron-per-layer demo |
| `run` | Trains one configuration for the bonus experiments |

### Python and NumPy shorthand
| Symbol | Meaning |
|---|---|
| `@` | Matrix multiplication |
| `.T` | Transpose (swap rows and columns) |
| `axis=1` | Work across each row (one example); `axis=0` works down the columns |
| `keepdims=True` | Keep dimensions so shapes still line up |
| `.shape` | Size of an array |
| `np.argmax` | Position of the largest value (the predicted digit) |
| `np.bincount` | Counts how many of each class (used for the baseline) |
| `.reshape(28, 28)` | Turns 784 numbers back into a 28 x 28 picture |
