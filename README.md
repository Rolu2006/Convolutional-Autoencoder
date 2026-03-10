# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
To design and implement a Denoising Autoencoder using PyTorch that learns to remove noise from images. The model is trained on the MNIST handwritten digit dataset, where noise is artificially added to the input images, and the autoencoder learns to reconstruct the original clean images from the noisy inputs.

<img width="1772" height="733" alt="Screenshot 2026-03-10 155652" src="https://github.com/user-attachments/assets/263dc1a7-abc5-4c7e-add5-bb2582c7f738" />





## DESIGN STEPS

### STEP 1:
Load and preprocess the dataset by importing the MNIST dataset, converting images to tensors, and adding random noise to the input images.
### STEP 2:
Design the denoising autoencoder model using an encoder to compress the noisy image and a decoder to reconstruct the clean image.
### STEP 3:

Train and evaluate the model by minimizing reconstruction loss and visualize the original, noisy, and denoised images to analyze performance.

## PROGRAM


### Name: Somalaraju Rohini
### Register Number:212224240156


```
super(DenoisingAutoencoder, self).__init__()

# Encoder
self.encoder = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 7)
)

# Decoder
self.decoder = nn.Sequential(
    nn.ConvTranspose2d(64, 32, 7),
    nn.ReLU(),
    nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    nn.Sigmoid()
)

x = self.encoder(x)
x = self.decoder(x)
return x
```
```
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
model.train()
for epoch in range(epochs):
    running_loss = 0
    for images, _ in loader:
        images = images.to(device)

        noisy_images = add_noise(images)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
```

## OUTPUT

### Model Summary




<img width="898" height="530" alt="Screenshot 2026-03-10 155604" src="https://github.com/user-attachments/assets/4e38211e-f336-4995-9fca-b8f90972abcf" />




### Original vs Noisy Vs Reconstructed Image

<img width="1817" height="789" alt="Screenshot 2026-03-10 160621" src="https://github.com/user-attachments/assets/cfb73e3b-67c8-4fb8-ae35-17e2b979a6f9" />





## RESULT
The denoising autoencoder successfully reconstructs clean images from noisy MNIST inputs after training.
