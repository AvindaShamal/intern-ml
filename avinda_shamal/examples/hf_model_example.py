from datasets import load_dataset, load_dataset_builder
from torchvision import transforms
from cvas import HuggingFaceDataset

# check the dataset information
ds_builder = load_dataset_builder("cifar10")
print(f"Description: {ds_builder.info.description}")
print(f"Features: {ds_builder.info.features}")
print(f"Splits info: {ds_builder.info.splits}")

# load the cifar10 dataset
hf_dataset = load_dataset("cifar10")

# initialize the HuggingFaceDataset class
train_data = HuggingFaceDataset(
    hf_dataset, "train", transform=transforms.ToTensor()
)
test_data = HuggingFaceDataset(
    hf_dataset, "test", transform=transforms.ToTensor()
)

print(len(test_data))
print(len(train_data))
print(train_data[0])
print(train_data[0][0].shape)
