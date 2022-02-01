from torch import nn


# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential()
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.model(x)
        # [batch_size, 32 * 7 * 7]
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
