# models/accuracy_calculator.py
import torch
import torch.nn.functional as F

def calculate_accuracy(output, labels):
    # Apply softmax to the logits to get probabilities
    probabilities = F.softmax(output, dim=1)
    
    # Get the predicted class with the highest probability
    _, preds = torch.max(probabilities, 1)
    
    # Print the predicted labels for debugging
    print(f'Predicted labels: {preds}')
    print(f'Actual labels: {labels}')
    
    # Count the number of correct predictions
    correct = torch.sum(preds == labels).item()
    
    # Calculate accuracy
    accuracy = correct / labels.size(0)
    return accuracy
