# Neural Network from Scratch

This project demonstrates how to build a simple feedforward neural network from scratch using Python and NumPy. The neural network includes forward and backward propagation, weight updates using gradient descent, and basic evaluation on synthetic data. This is a great project to understand the inner workings of neural networks without relying on high-level libraries.

---

## 📚 **Project Structure**

- **`nn_Scratch.py`**: Main script containing the implementation of the neural network.
- **`.gitignore`**: Ensures that unnecessary files (like the virtual environment) are not pushed to the repository.

---

## 🔧 **Requirements**

Before running the project, make sure you have the following installed:

- Python 3.6 or higher
- NumPy
- scikit-learn (for train-test split)

To install the required packages, run:

```bash
pip install numpy scikit-learn
```

---

## 🚀 **How to Run the Project**

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuralNet_Scratch.git
cd NeuralNet_Scratch
```

### Step 2: Set Up a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt  # If available
```

### Step 4: Run the Neural Network Script

```bash
python nn_Scratch.py
```

---

## ⚙️ **Neural Network Details**

- **Input Layer**: Takes in the synthetic input features.
- **Hidden Layer**: 4 neurons with sigmoid activation.
- **Output Layer**: 1 neuron with sigmoid activation for binary classification.
- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Gradient descent.

---

## 🔍 **Key Components**

1. **Data Generation:**
    - Synthetic data with two input features and binary labels.

2. **Forward Propagation:**
    - Calculates activations using the sigmoid function.

3. **Backward Propagation:**
    - Computes gradients using the chain rule.

4. **Training Loop:**
    - Updates weights using gradient descent.

5. **Evaluation:**
    - Tests the model on the test set and prints accuracy.

---

## 📊 **Output**

During training, you will see the cost (loss) printed every 100 iterations. After training, the test accuracy is displayed.

Example output:
```
Iteration 0, Cost: 0.6930
Iteration 100, Cost: 0.6922
Iteration 200, Cost: 0.6919
...
Test Accuracy: 90.00%
```

---

## 📂 **File Descriptions**

- **`nn_Scratch.py`**: Contains the full implementation of the neural network.
- **Synthetic Data:** Generated using NumPy for training and testing.

---

## 💡 **Understanding the Code**

1. **Initialization:** Randomly initializes weights and biases.
2. **Forward Propagation:** Computes the output of the network.
3. **Loss Calculation:** Uses cross-entropy to calculate error.
4. **Backward Propagation:** Updates the weights using gradients.
5. **Prediction:** Classifies the test data and calculates accuracy.

---

## 🌟 **Future Improvements**

- Add support for multi-class classification using softmax.
- Use other optimization techniques such as Adam.
- Experiment with different activation functions and network architectures.

---

## 📬 **Contributing**

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## 🛡️ **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## ✉️ **Contact**

If you have any questions, feel free to contact me:

- **GitHub:** [YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email:** your_email@example.com

