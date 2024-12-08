import os
import numpy as np
from PIL import Image

def load_images_and_labels(data, image_size=64):
    """
    Load images from `planes` and `cars` subdirectories within `base_path`.
    Convert them to grayscale, resize to image_size x image_size, flatten, and label them.
    
    Labels:
    - planes: 0
    - cars: 1
    """
    X = []
    y = []
    
    # Paths for plane and car images
    plane_dir = os.path.join(data, 'planes')
    car_dir = os.path.join(data, 'cars')

    # Load plane images
    for filename in os.listdir(plane_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(plane_dir, filename)
            img = Image.open(img_path).convert('L') # Convert to grayscale
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

            
            # Convert to numpy array and normalize pixel values to [0,1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Flatten the image into a vector
            img_flat = img_array.flatten()
            
            X.append(img_flat)
            y.append([0])  # plane label

    # Load car images
    for filename in os.listdir(car_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(car_dir, filename)
            img = Image.open(img_path).convert('L') # Convert to grayscale
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

            
            # Convert to numpy array and normalize pixel values to [0,1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Flatten the image
            img_flat = img_array.flatten()
            
            X.append(img_flat)
            y.append([1])  # car label

    X = np.array(X)
    y = np.array(y)
    
    return X, y

# -----------------------------------------
# Example usage and integration with the MLP
# -----------------------------------------
if __name__ == "__main__":
    # Load your prepared images
    base_path = 'data'  # adjust this path accordingly
    X, y = load_images_and_labels(base_path, image_size=64)
    
    # Check data dimensions
    # X should be (num_samples, 256) if image_size=16
    # y should be (num_samples, 1)
    print("Data loaded:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # From here, you can use the MLP code provided earlier.
    # For instance, using the same parameters as the previous code:
    
    input_dim = X.shape[1]   # 256 for 16x16 images
    hidden_dim1 = 128
    hidden_dim2 = 128
    output_dim = 1
    learning_rate = 0.01
    n_epochs = 500

    # Initialize weights and biases
    W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
    b1 = np.zeros((1, hidden_dim1))

    W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01
    b2 = np.zeros((1, hidden_dim2))

    W3 = np.random.randn(hidden_dim2, output_dim) * 0.01
    b3 = np.zeros((1, output_dim))

    # Activation and loss functions
    def relu(z):
        return np.maximum(0, z)

    def relu_derivative(a):
        return np.where(a > 0, 1, 0)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy_loss(y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Training loop
    for epoch in range(n_epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)
        z3 = np.dot(a2, W3) + b3
        a3 = sigmoid(z3)

        # Compute loss
        loss = binary_cross_entropy_loss(y, a3)

        # Backpropagation
        dz3 = (a3 - y)
        dW3 = np.dot(a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, W3.T)
        dz2 = da2 * relu_derivative(a2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update parameters
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate on the training set
    preds = (a3 >= 0.5).astype(int)
    accuracy = np.mean(preds == y)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
import matplotlib.pyplot as plt

# After training is done:
# Pick a random sample from the training set
sample_idx = np.random.randint(0, X.shape[0])
sample_x = X[sample_idx].reshape(1, -1)  # shape: (1, input_dim)
sample_y = y[sample_idx]

# Forward pass with the sample
z1 = np.dot(sample_x, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = relu(z2)
z3 = np.dot(a2, W3) + b3
sample_pred = sigmoid(z3)

# Convert prediction to a binary label: 0 for plane, 1 for car
pred_label = int(sample_pred >= 0.5)
actual_label = int(sample_y[0])

print("----- Sample Prediction -----")
print(f"Sample Index: {sample_idx}")
print(f"Predicted Label: {pred_label} ({'car' if pred_label == 1 else 'plane'})")
print(f"Actual Label: {actual_label} ({'car' if actual_label == 1 else 'plane'})")

# Reshape the vector back to the image form for display
image_size = 64
img = sample_x.reshape(image_size, image_size)

# Plot the image
plt.imshow(img, cmap='gray')
plt.title(f"Actual: {'car' if actual_label == 1 else 'plane'} | Predicted: {'car' if pred_label == 1 else 'plane'}")
plt.axis('off')
plt.show()
