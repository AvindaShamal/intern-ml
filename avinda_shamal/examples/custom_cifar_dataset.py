from torch.utils.data import DataLoader
from torchvision import transforms
from cvas import CustomImageDataset
import torch
import matplotlib.pyplot as plt


# Define transformations (resize to 32x32 and normalize)
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # # Apply standard Normalization and then convert to [-1, 1] range
        # transforms.ToTensor()
        # # convert each image to a tensor and normalize to [0, 1] range
        # Means and Standard deviations of R,G,B planes are 0.5
        ### no need to apply ToTensor() as read_image() already converts image to tensor
    ]
)

# Load the training and test data
training_data = CustomImageDataset(
    "D:/Intern ML/cifar10/train", transform=transform
)
test_data = CustomImageDataset(
    "D:/Intern ML/cifar10/test", transform=transform
)

print(len(training_data))

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

print(len(train_dataloader))

labels_map = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    img_ = img.permute(1, 2, 0).numpy()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img_)
plt.show()

## Another method to display image and label
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img.permute(1, 2, 0))
plt.show()
print(f"Label: {labels_map[label.item()]}")
