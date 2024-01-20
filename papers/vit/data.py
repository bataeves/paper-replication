from torchvision import transforms


def get_transforms(img_height: int, img_width: int):
    # Create transform pipeline manually
    return transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )
