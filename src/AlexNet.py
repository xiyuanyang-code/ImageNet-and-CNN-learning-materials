"""
Author: Xiyuan Yang   xiyuan_yang@outlook.com
Date: 2025-04-15 14:40:20
LastEditors: Xiyuan Yang   xiyuan_yang@outlook.com
LastEditTime: 2025-04-15 14:41:31
FilePath: /CNN-tutorial/src/AlexNet.py
Description:
Do you code and make progress today?
Copyright (c) 2025 by Xiyuan Yang, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # Define the convolutional layers
        self.features = nn.Sequential(
            # First convolutional layer: 3 input channels, 64 output channels, kernel size 11x11, stride 4, padding 2
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Second convolutional layer: 64 input channels, 192 output channels, kernel size 5x5, padding 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Third convolutional layer: 192 input channels, 384 output channels, kernel size 3x3, padding 1
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Fourth convolutional layer: 384 input channels, 256 output channels, kernel size 3x3, padding 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Fifth convolutional layer: 256 input channels, 256 output channels, kernel size 3x3, padding 1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # Flattened feature map size is 256 * 6 * 6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # Pass input through the convolutional layers
        x = self.features(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Pass through the fully connected layers
        x = self.classifier(x)
        return x


# Example usage
if __name__ == "__main__":
    # Create an instance of AlexNet with 1000 output classes (default for ImageNet)
    model = AlexNet(num_classes=1000)

    # Print the model architecture
    print(model)

    # Test with a random input tensor (batch size 1, 3 channels, 224x224 image)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Should output torch.Size([1, 1000])
