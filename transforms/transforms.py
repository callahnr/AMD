import torchvision.transforms as transforms


def compose():
    return transforms.Compose(
        [transforms.ToTensor()])