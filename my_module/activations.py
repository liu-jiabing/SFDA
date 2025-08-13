# activations.py
import torch
import torch.nn.functional as F

class Activations:
    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return F.relu(x)

    @staticmethod
    def leaky_relu(x, negative_slope=0.01):
        """Leaky ReLU activation function"""
        return F.leaky_relu(x, negative_slope)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return torch.sigmoid(x)

    @staticmethod
    def tanh(x):
        """Tanh activation function"""
        return torch.tanh(x)

    @staticmethod
    def swish(x):
        """Swish activation function"""
        return x * torch.sigmoid(x)

    @staticmethod
    def gelu(x):
        """GELU activation function"""
        return F.gelu(x)

    @staticmethod
    def softmax(x, dim=-1):
        """Softmax activation function"""
        return F.softmax(x, dim=dim)

# Example of how to use:
if __name__ == "__main__":
    # Create a random tensor for testing
    x = torch.randn(3, 5)

    print("ReLU:", Activations.relu(x))
    print("Leaky ReLU:", Activations.leaky_relu(x))
    print("Sigmoid:", Activations.sigmoid(x))
    print("Tanh:", Activations.tanh(x))
    print("Swish:", Activations.swish(x))
    print("GELU:", Activations.gelu(x))
    print("Softmax:", Activations.softmax(x))
