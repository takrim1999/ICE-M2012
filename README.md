# Advanced Machine Learning & Deep Learning Lab (ICE-M2012)

This repository contains implementations of machine learning and deep learning algorithms from the M. Eng. Advanced Machine Learning syllabus, organized into structured `ML/` and `DL/` directories following the granular coding style from `learn-ai`.

---

## 📂 Repository Structure

```
ICE-M2012/
├── ML/                                      # Classical Machine Learning (Iris Dataset)
│   ├── LinearRegression/
│   │   └── linear_regression.ipynb          # Simple/Multiple Regression, Normal Eq. & Gradient Descent
│   ├── LogisticRegression/
│   │   └── logistic_regression.ipynb        # Binary (Sigmoid) & Multiclass (Softmax) Classification
│   ├── NaiveBayes/
│   │   └── naive_bayes.ipynb                # Gaussian Naïve Bayes & 5-Fold Cross-Validation
│   ├── DecisionTree/
│   │   └── decision_tree.ipynb              # Decision Tree, Gini vs. Entropy, & plot_tree visualization
│   ├── RandomForest/
│   │   └── random_forest.ipynb              # 100-tree ensemble, OOB scoring, & Feature Importances
│   ├── SVM/
│   │   └── svm.ipynb                        # Support Vector Machine with RBF kernel & Support Vectors
│   ├── NeuralNetwork_MLP/
│   │   └── mlp_neural_network.ipynb         # Multi-Layer Perceptron (4 -> 10 -> 10 -> 3) & Loss Curve
│   └── ModelComparison/
│       └── model_comparison.ipynb          # Benchmark comparing all 6 ML classifiers side-by-side
│
├── DL/                                      # Deep Learning (Fashion MNIST Dataset)
│   ├── LSTM/
│   │   └── lstm_fashion_mnist.ipynb         # Recurrent Neural Network / LSTM for sequence data
│   ├── CNN/
│   │   └── cnn_fashion_mnist.ipynb          # 3-Block Convolutional Neural Network (Conv2D/Pooling/Dropout)
│   └── Comparison/
│       └── dl_comparison.ipynb              # Direct CNN vs. LSTM architectural benchmark & empirical study
│
├── demo/                                    # Original instructor reference notebooks
├── pyproject.toml                           # Environment configuration (Python 3.13)
├── uv.lock                                  # Dependency lockfile
└── README.md
```

---

## 🚀 Environment Setup & GPU Execution

This project is managed with `uv` targeting **Python 3.13** with GPU acceleration via `tensorflow[and-cuda]`:

```bash
# Sync dependencies
uv sync

# Launch Jupyter Lab
uv run jupyter lab
```
