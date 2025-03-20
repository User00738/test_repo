import torch
import torch.nn as nn


from utils import count_parameters

cfg_list = [
    (7, 64, 2, 3),   # (kernel_size, num_filters, stride, padding)
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4] # [conv_1, conv_2, num_repeats]
]


class ConvBnActivation(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride, activation):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        activation = 'relu'
        self.conv1 = ConvBnActivation(7, 3, 64, 2, activation)
        self.conv2 = ConvBnActivation(3, 64, 192, 1, activation)
        self.conv3 = ConvBnActivation(1, 192, 128, 1, activation)
        self.conv4 = ConvBnActivation(3, 128, 256, 1, activation)
        self.conv5 = ConvBnActivation(1, 256, 256, 1, activation)
        self.conv6 = ConvBnActivation(3, 256, 512, 1, activation)
        self.conv7_1 = ConvBnActivation(1, 512, 256, 1, activation)
        self.conv7_2 = ConvBnActivation(3, 256, 512, 1, activation)
        self.conv8 = ConvBnActivation(1, 512, 512, 1, activation)
        self.conv9 = ConvBnActivation(3, 512, 1024, 1, activation)
        self.conv10_1 = ConvBnActivation(1, 1024, 512, 1, activation)
        self.conv10_2 = ConvBnActivation(3, 512, 1024, 1, activation)
        self.conv11 = ConvBnActivation(3, 1024, 1024, 1, activation)
        self.conv12 = ConvBnActivation(3, 1024, 1024, 2, activation)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)
        x2 = self.conv2(x1)
        x2 = self.maxpool(x2)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = self.maxpool(x6)
        for i in range(4):
            x6 = self.conv7_1(x6)
            x6 = self.conv7_2(x6)

        x7 = x6
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x9 = self.maxpool(x9)
        for i in range(2):
            x9 = self.conv10_1(x9)
            x9 = self.conv10_2(x9)

        x10 = x9
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv11(x12)
        x14 = self.conv11(x13)
        output = self._create_fcl(x14)

        return output
    

    def _create_fcl(self, x):
        nn_list = nn.Sequential(nn.Flatten(),
                                nn.Linear(1024 * 7 * 7, 496),
                                nn.LeakyReLU(0.1),
                                nn.Linear(496, 7 * 7 * (20 + 5 * 2)))
        
        return nn_list(x)



if __name__ == "__main__":
    model = YOLOv1()
    image = torch.rand(4, 3, 448, 448)
    output = model(image)
    count_parameters(model)
    print(output.shape)