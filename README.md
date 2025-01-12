
# Neural Network Performance for Handwritten Digit and CelebA Classification

This project explores the implementation and optimization of neural networks for classification tasks using datasets such as **MNIST** (handwritten digits) and **CelebA** (celebrity face attributes). The focus of this work is to experiment with different hyper-parameters such as the number of hidden units and lambda values to optimize the performance of the model.


## File Structure and Usage ( to get access to all these files download the zip file that is attached)

### 1. **mnist_all.mat**
   - **Description**: Contains the MNIST dataset with training and test sets of handwritten digits.
   - **Usage**: Used for digit classification, split into training and validation sets.

### 2. **face_all.pickle**
   - **Description**: A subset of the CelebA dataset for classifying faces based on glasses.
   - **Usage**: Used for facial attribute classification (glasses vs no-glasses).

### 3. **nnScript.py**
   - **Description**: Implements the neural network including preprocessing, feedforward, error function, and backpropagation.
   - **Functions**: 
     - `preprocess()`: Prepares data.
     - `sigmoid()`: Activation function.
     - `nnObjFunction()`: Error function with regularization.
     - `nnPredict()`: Makes predictions.
     - `initializeWeights()`: Initializes weights.
   - **Usage**: Central script for training and evaluating the neural network.

### 4. **facennScript.py**
   - **Description**: Runs the neural network on the CelebA dataset, calling functions from `nnScript.py`.
   - **Usage**: Used specifically for face classification tasks.

### 5. **params.pickle**
   - **Description**: Stores the learned model parameters (weights, optimal hidden units, lambda).
   - **Usage**: Used for making predictions with the trained model.


## Key Features:
- **Hyper-parameter Optimization**: A heatmap analysis of the validation set accuracy was conducted to determine the optimal number of hidden units and lambda values.
- **Model Evaluation**: The neural network performance was evaluated on multiple datasets including MNIST and CelebA. The project compared the results of a standard neural network, a deep neural network (using TensorFlow), and a Convolutional Neural Network (CNN).

## Accuracy Results:
- **MNIST**: Achieved a test accuracy of **95.1%** after optimizing the hyper-parameters with lambda = 15 and n_hidden = 50.
- **CelebA**: Achieved a test accuracy of **85.73%** with lambda = 10 and n_hidden = 256.
- **Convolutional Neural Network (CNN)**: Achieved a test accuracy of **98.9%** on the MNIST dataset.

## Training Time:
Noted training times for different models and hyper-parameters. The deep neural network, for instance, showed faster training times compared to the standard neural network.

## Hyper-parameter Tuning:
- **Hidden Units**: It was observed that increasing the number of hidden units, particularly to 16 or 20, generally resulted in better accuracy (above 92%).
- **Lambda**: While lambda values did not exhibit a clear trend, experimenting with values such as 10, 20, and 30 helped improve accuracy when used in combination with the right number of hidden units.

### The optimal hyper-parameters found were:
- **MNIST**: lambda = 15, n_hidden = 50
- **CelebA**: lambda = 10, n_hidden = 256

## Comparison of Neural Networks:
The performance of a standard neural network was compared with a **Deep Neural Network (DNN)** and a **Convolutional Neural Network (CNN)** on the MNIST dataset. The DNN outperformed the standard neural network in terms of test accuracy due to its deeper architecture. Meanwhile, the CNN achieved the highest test accuracy (**98.9%**) for digit classification tasks, showcasing its superior performance in image recognition.

## Confusion Matrix:
The confusion matrix was used to evaluate the model's classification performance. It showed that most predictions were accurate, with some misclassifications visible in the off-diagonal cells. The confusion matrix highlighted the model's ability to classify digits effectively and provided insights into areas for improvement.

## Conclusion:
This project highlights the impact of hyper-parameter tuning on the performance of neural networks and compares the efficiency of different architectures, including standard neural networks, deep neural networks, and convolutional neural networks. The CNN model demonstrated the best performance for digit classification tasks, while the DNN was more efficient in terms of training time.
