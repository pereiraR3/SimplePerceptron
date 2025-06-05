# Artificial Intelligence Activity - Algorithm Implementations

This repository contains Python implementations of two fundamental Artificial Intelligence algorithms, developed as part of an assignment for the AI course. The implemented algorithms are:

1.  **Decision Tree:** A supervised learning algorithm used for classification and regression tasks.
2.  **Perceptron:** A fundamental algorithm for linear classification, serving as a building block for more complex neural networks.

## Repository Structure

The repository is organized as follows:


```
├── README.md # This file, containing information about the repository
└── scripts/
├── decision_tree.py # Python script implementing the Decision Tree
└── perceptron.py # Python script implementing the Perceptron
```


## Script Descriptions

### `decision_tree.py`

This script implements a Decision Tree for classification tasks. Key features of the implementation include:

* **Entropy Calculation:** Used to measure the impurity of a dataset.
* **Data Splitting:** Function to partition the dataset based on an attribute and a threshold value.
* **Information Gain:** Metric used to determine the best attribute for splitting.
* **Recursive Tree Building:** The tree is built recursively by splitting the data until a stopping criterion is met (node purity or max depth).
* **Tree Representation:** Uses a class `DecisionTree` with an internal `_DecisionNode` class to represent tree nodes.
* **Functions for Training, Printing, and Classification:** Methods to train the tree on data, display its structure, and classify new instances.

### `perceptron.py`

This script implements a Perceptron, a linear learning algorithm for binary classification. Key features of the implementation include:

* **Step Activation Function:** Used to produce a binary output.
* **Iterative Training:** The Perceptron's weights are updated iteratively based on the error between predicted and expected outputs.
* **Learning Rate and Threshold:** Parameters controlling the learning process and classification decision.
* **Perceptron Representation:** Uses a `Perceptron` class to encapsulate weights, threshold, and training/prediction methods.
* **Functions for Training and Prediction:** Methods to train the Perceptron on labeled data and to predict the class of new instances.

## How to Run the Scripts

To run the scripts, make sure you have Python 3 installed, along with the `numpy` and `pandas` libraries (required for the Decision Tree).

1.  **Clone the repository** (if you haven't already).
2.  **Navigate to the `scripts` directory:**
    ```bash
    cd scripts
    ```
3.  **Run the scripts with the Python interpreter:**
    ```bash
    python decision_tree.py
    python perceptron.py
    ```

    The scripts will execute the algorithm implementations using sample data and display the results in your terminal.

---

**Author:** Anthony Ricardo Rodrigues Rezende  
**Date:** May 4, 2025  
**Course:** Artificial Intelligence  
**Institution:** Federal University of Mato Grosso
