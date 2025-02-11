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
