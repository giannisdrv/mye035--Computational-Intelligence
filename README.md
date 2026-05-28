# Multi-Layer Perceptron (MLP) Data Classifier

## Overview
This repository contains a custom implementation of a **Multi-Layer Perceptron (MLP) Neural Network** built entirely from scratch in C. Developed as part of the "Computational Intelligence" course at the University of Ioannina, the project focuses on classifying 2-dimensional synthetic geometric data into 4 distinct categories. The network features a deep architecture utilizing three hidden layers, allowing for rigorous experimentation with hyperparameters without the use of external Machine Learning libraries.

## Features
* **Custom Neural Engine (C):** Implements feed-forward, error calculation, and backpropagation mechanics strictly using standard C libraries (`<math.h>`, `<stdlib.h>`). 
* **Dynamic Data Generation:** Includes a standalone data generator (`data.c`) that synthesizes thousands of 2D data points `(x1, x2)` uniformly distributed in a `[0, 2.0]` grid, labeling them based on custom boundary rules.
* **Configurable Architecture:** The network's structure is highly modular via C macros, allowing instant adjustments to the number of input dimensions (`d=2`), output categories (`K=4`), and hidden layer neuron counts (e.g., 20-20-20 or 10-10-10).
* **Activation Function Ablation:** Supports swapping activation functions (such as `tanh`, `ReLU`, and `logistic`/sigmoid) individually per hidden layer to test their impact on gradient flow and convergence.
* **Mini-Batch Gradient Descent:** Features a configurable batch size parameter (`L`) to dynamically slice the dataset into smaller training chunks, optimizing weight updates per epoch.

## Architecture & Technical Details
* **`data.c`**: The data synthesizer. It randomly generates 8,000 data points (4,000 for training, 4,000 for testing), assigns them a category based on spatial coordinates, and exports them to text files (`training_data_with_category.txt`, `testing_data_with_category.txt`).
* **`ex1.c`**: The core MLP model. It reads the generated datasets and maps them onto the input layer. The network uses multi-dimensional arrays for weights (e.g., `w1[d][neurons_H1]`) and propagates signals through three intermediate representations (`h1`, `h2`, `h3`) before outputting probability distributions for the 4 classes. It tracks epoch errors (delta) to monitor convergence.

## Evaluation & Performance
Extensive experiments were conducted to evaluate how hyperparameter tuning affects the classification accuracy:
* **Neuron Distribution:** Testing various neuron configurations (e.g., 20-10-5 vs. 5-10-20) revealed that placing a larger number of neurons closer to the output layer slightly increases the correct prediction rate. A robust `20-20-20` layout yielded highly stable results.
* **Batch Size Impact:** Decreasing the batch size (`L`) dramatically improved training efficiency. For instance, using smaller batches (e.g., `L=20`) allowed the model to update weights more frequently per epoch, resulting in a ~30% increase in accuracy compared to larger batch sizes (`L=400`).
* **Activation Functions:** Testing different combinations of activation functions showed that utilizing `tanh` for the first two hidden layers and `ReLU` for the third layer provided an optimal balance, effectively mitigating vanishing gradients while keeping the output activations sharp.

## Authors
* **Ioannis Drivas** (5216)
* University of Ioannina, Department of Computer Science and Engineering
