from torchvision import datasets, transforms

def get_mnist():
    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    return dataset