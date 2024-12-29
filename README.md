# Federated Learning with Deep Belief Network (DBN) for Handwritten Digit Classification

## Overview
This project demonstrates the implementation of a federated learning framework using a custom Deep Belief Network (DBN) to classify handwritten digits from the MNIST dataset. The federated learning approach ensures privacy-preserving machine learning, where the data remains local on clients while collaboratively improving the global model.

## Key Features
- **Federated Learning**: Simulates federated learning across multiple clients, where each client trains a local model on its own data.
- **Custom DBN**: A Deep Belief Network (DBN) is used for the classification task, with multiple hidden layers.
- **Federated Averaging**: Model parameters are averaged across clients after each training round to update the global model.
- **Collaborative Model Training**: Multiple clients train their local models on different data subsets, contributing to a more robust global model.

## Dependencies
This notebook uses the following Python libraries:
- **PyTorch**: For building and training the Deep Belief Network (DBN).
- **Scikit-learn**: For handling the MNIST dataset and splitting the data.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing training progress (optional).

You can install the required dependencies by running:
```
pip install torch scikit-learn numpy matplotlib
```

## How to Use
1. **Clone this repository**:
   ```
   git clone https://github.com/Santhoshcv19/Federated-Deep_Belief_Network.git
   ```

2. **Open the notebook**:
   Open `dbnfed.ipynb` using Jupyter Notebook or JupyterLab.
   ```
   jupyter notebook dbnfed.ipynb
   ```

3. **Run the notebook**:
   Follow the steps in the notebook to:
   - Load the MNIST dataset.
   - Split the dataset into clients.
   - Train local models using a custom Deep Belief Network (DBN).
   - Perform federated averaging to update the global model.
   - Evaluate the model's accuracy on a validation set.

## Notebook Structure
- **Data Loading and Preprocessing**: The notebook loads the MNIST dataset and splits it into `X` (features) and `y` (labels).
- **Federated Training**: The notebook simulates federated learning by training a model on local client data and updating the global model using federated averaging.
- **Model Evaluation**: After completing federated training, the global model is evaluated on a validation set to determine the classification accuracy.

## Federated Learning Flow
1. **Data Split**: The dataset is divided into subsets corresponding to each client.
2. **Local Training**: Each client trains a local model using its own subset of the data.
3. **Federated Averaging**: After training, the models' parameters are averaged across clients to form the global model.
4. **Iteration**: This process is repeated for a set number of rounds, with each round improving the global model through collaborative training.

## Results
The notebook will display the accuracy of the global model after each federated training round. Example output could look like:
```
Round 1/10
Training on client 1
Training on client 2
Training on client 3
...
Final Accuracy: 0.97
```

## Future Improvements
- Implement different optimizers for local training (e.g., Adam, SGD).
- Scale the number of clients to simulate a larger federated learning environment.
- Integrate other datasets for multi-task federated learning.
