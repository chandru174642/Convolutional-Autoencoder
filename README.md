# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


Problem Statement:

Image denoising is a common problem in image processing and computer vision. Images often get corrupted due to noise during transmission or acquisition. The goal of this project is to design and train a convolutional autoencoder that can learn to remove noise from images and reconstruct clean outputs.
We use the MNIST handwritten digits dataset where Gaussian noise is artificially added to simulate noisy images.

Dataset

Dataset Used: MNIST (Modified National Institute of Standards and Technology)

Training Data: 60,000 grayscale images of handwritten digits (28×28 pixels)

Testing Data: 10,000 grayscale images

Noise: Gaussian noise added to simulate real-world corruption


## DESIGN STEPS
Step 1: Import the required libraries and load the MNIST dataset with proper transformations.

Step 2: Add Gaussian noise to the images to create noisy input data.

Step 3: Design the convolutional autoencoder model with encoder and decoder layers.

Step 4: Define the loss function (MSELoss) and optimizer (Adam).

Step 5: Train the model using noisy images as input and clean images as target output.

Step 6: Test and visualize the results by displaying original, noisy, and denoised images.

Step 7: Evaluate the model performance and conclude the effectiveness of denoising.
Write your own steps

## PROGRAM
### Name:CHANDRU.P
### Register Number:212223110007

```
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # (1,28,28) -> (32,14,14)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32,14,14) -> (64,7,7)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64,7,7) -> (32,14,14)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # (32,14,14) -> (1,28,28)
            nn.Sigmoid()  # Output in range [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
```
# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
```
```
# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:CHANDRU.P")
    print("Register Number: 212223110007")

    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```
```
# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```
## OUTPUT


### Model Summary

Include your model summary


<img width="562" height="354" alt="Screenshot 2025-10-14 112256" src="https://github.com/user-attachments/assets/930d38ed-bab0-4b2f-a9a2-6413904698c2" />


### Original vs Noisy Vs Reconstructed Image

Include a few sample images here.




<img width="1671" height="600" alt="image" src="https://github.com/user-attachments/assets/c0bd5924-5db3-4283-be61-2f09520065a8" />





## RESULT

The convolutional autoencoder was successfully trained to remove noise from MNIST images, effectively reconstructing clean and clear outputs from noisy inputs.
