# Land Cover Classification Using Aerial Imagery

A machine learning and deep learning project exploring how aerial images can be classified into land-use and land-cover categories. The project implements three approaches, from a raw-pixel baseline to handcrafted image features and a Convolutional Neural Network (CNN).

## Project Overview

This project uses the **UC Merced Land Use Dataset**, consisting of 2,100 aerial images across 21 land-use categories.

Three classification approaches are implemented:

1. **Support Vector Machine (SVM):** Classification using raw pixel features on five selected classes.
2. **HOG + PCA + SVM:** Histogram of Oriented Gradients for feature extraction, Principal Component Analysis for dimensionality reduction, and SVM classification across all 21 classes.
3. **Convolutional Neural Network (CNN):** End-to-end deep learning for classification across all 21 classes.

The objective is to investigate the effectiveness of traditional machine learning techniques and deep learning for aerial image classification.

## Dataset

**Dataset:** UC Merced Land Use Dataset

* Total images: 2,100
* Number of categories: 21
* Image dimensions used: 61 × 61 × 3

**Data splitting:**

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        80% |
| Validation |        10% |
| Testing    |        10% |

The dataset is split using stratified sampling to preserve class distributions.

The notebook expects two NumPy files:

```
DS_Xdata.npy
Ydata.npy
```

These files contain the preprocessed image data and corresponding class labels. They must be placed in the same working directory as the notebook.

The notebook does not include the original dataset preparation script.

## Methodology

### 1. Support Vector Machine (SVM)

A baseline classification model is developed using Support Vector Machines.

**Implementation:**

* Selected five land-use classes.
* Flattened images into one-dimensional feature vectors.
* Applied standardisation using StandardScaler.
* Performed hyperparameter tuning using GridSearchCV.
* Evaluated linear and RBF kernels.
* Used three-fold cross-validation.

This experiment investigates the limitations of raw-pixel features for image classification.

### 2. HOG + PCA + SVM

The second approach combines feature extraction and dimensionality reduction with Support Vector Machines.

**Histogram of Oriented Gradients (HOG)**

HOG extracts important image features by capturing edges, gradients and spatial patterns.

Configuration:

* Orientations: 8
* Pixels per cell: 8 × 8
* Cells per block: 2 × 2

**Principal Component Analysis (PCA)**

PCA reduces the dimensionality of the extracted HOG features to 50 principal components.

This reduces feature complexity and computational requirements.

**SVM Classification**

* Classification across all 21 categories.
* Linear and RBF kernels.
* GridSearchCV for hyperparameter optimisation.
* Five-fold cross-validation.

### 3. Convolutional Neural Network (CNN)

A CNN is developed to automatically learn image features and perform multi-class classification.

**Architecture:**

* Convolutional layers for feature extraction.
* Max-pooling layers for dimensionality reduction.
* Flatten layer.
* Dense layer with 128 neurons.
* Dropout layer with a rate of 0.5.
* Softmax output layer for 21 classes.

**Training configuration:**

| Parameter      | Value                            |
| -------------- | -------------------------------- |
| Optimizer      | Adam                             |
| Learning rate  | 0.001                            |
| Loss function  | Sparse Categorical Cross-Entropy |
| Batch size     | 32                               |
| Maximum epochs | 20                               |
| Regularisation | Dropout and Early Stopping       |

Images are normalised to the range [0, 1] before training.

Early stopping is used to monitor validation performance and restore the best model weights.

## Results

The following results are taken from the saved outputs of the Jupyter notebook.

| Model           | Classes | Training Accuracy | Validation Accuracy | Test Accuracy |
| --------------- | ------: | ----------------: | ------------------: | ------------: |
| SVM             |       5 |            99.50% |              74.00% |        70.00% |
| HOG + PCA + SVM |      21 |            99.23% |              56.67% |        57.14% |
| CNN             |      21 |            89.46% |              59.05% |        60.00% |

**Key observations:**

* The raw-pixel SVM achieved 70% testing accuracy on the selected five-class problem.
* HOG and PCA enabled SVM classification across all 21 categories.
* The CNN achieved 60% testing accuracy on the 21-class dataset.
* Significant differences between training and validation accuracy indicate overfitting.

The baseline SVM uses only five categories, while the other two models use all 21 categories. Therefore, their results are not directly comparable.

The notebook also includes confusion matrices and CNN training and validation accuracy curves.

*Note: Results may vary when retraining the CNN. The accompanying project report contains results from different recorded experiments; the values above represent the notebook's saved outputs.*

## Technologies Used

| Technology         | Purpose                                |
| ------------------ | -------------------------------------- |
| Python             | Programming language                   |
| Jupyter Notebook   | Experimentation and development        |
| NumPy              | Numerical operations and image data    |
| Scikit-learn       | SVM, PCA, preprocessing and evaluation |
| Scikit-image       | HOG feature extraction                 |
| TensorFlow / Keras | CNN development and training           |
| Matplotlib         | Data visualisation                     |
| Seaborn            | Confusion matrix visualisation         |

## Installation and Setup

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Replace the placeholders with your GitHub repository URL and folder name.

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install numpy scikit-learn scikit-image tensorflow matplotlib seaborn notebook
```

**4. Prepare the dataset**

Place the following files alongside the notebook:

```
DS_Xdata.npy
Ydata.npy
```

The original UC Merced images must be preprocessed into the NumPy arrays expected by the notebook if these files are not already available.

**5. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open the project notebook and execute its cells sequentially.

## Limitations and Future Improvements

Although the models successfully perform aerial image classification, several improvements could be explored.

* Implement image augmentation techniques, such as rotation and flipping, to improve generalisation.
* Experiment with transfer learning using pretrained CNN architectures.
* Apply additional regularisation techniques to reduce overfitting.
* Evaluate every model on the same classes for a direct comparison.
* Analyse class-level performance using precision, recall and F1-score.
* Introduce reproducible dataset preparation and dependency configuration.

## Conclusion

This project demonstrates three approaches to aerial image classification using traditional machine learning and deep learning.

The experiments highlight the limitations of raw-pixel features, the role of handcrafted feature extraction and dimensionality reduction, and the ability of convolutional neural networks to learn spatial features automatically.

The project also demonstrates the importance of hyperparameter tuning, cross-validation, model evaluation and identifying overfitting when developing image classification systems.

## Project Materials

* **Jupyter Notebook:** Complete implementation of the models, training processes, evaluations and visualisations.
* **Project Report:** Detailed methodology, experimental results, discussion and references.
