from torch import nn
import torch.nn.functional as F

def conv1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)

class ResBlock(nn.Module):
    '''
    ResBlock: Building block class for the ResNetModel()
    '''
    def __init__(self, in_channels, out_channels, in_relu=True, stride=1, downsample=None):
        super().__init__()
        if in_channels is None:
            in_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.in_relu = in_relu
        self.conv1 = conv1(in_channels, out_channels) 
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv3(out_channels, out_channels, stride=stride)
        self.bn2= nn.BatchNorm1d(out_channels)
        self.conv3 = conv1(out_channels, out_channels)
        self.bn3=nn.BatchNorm1d(out_channels)
     
    def forward(self, x):
        identity = x
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out+=identity
        out = self.relu(out)

        return out


class ResClassifier(nn.Module):
    '''
    ResClassifier: The classifier block of the ResNetModel.
    '''
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.adrop1 = nn.AlphaDropout(p=0.5)
        self.adrop2 = nn.AlphaDropout(p=0.5)
        

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        x = self.adrop1(x)
        x = F.selu(self.fc2(x), inplace=True)
        x = self.adrop2(x)
        x = self.fc3(x)
        return x


class ResNetModel(nn.Module):
    '''
    ResNetModel: The fully built custom ResNet. 
    -   num_classes = The number of classes to classify
    -   in_channels = The number of features from the data
    -   channels = The number of hidden channels in the network
    '''
    def __init__(self, num_classes=24, in_channels=2, channels=32, repeats=1,
                layers=[1,1,1,1,1,1], expansion=[1, 2, 2, 4, 4, 4] ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.make_layer(channels, channels*expansion[0], layers[0], stride=1) #1024->512
        self.stack2 = self.make_layer(channels*expansion[0], channels*expansion[1], layers[1], stride=2) # 512->256
        self.stack3 = self.make_layer(channels*expansion[1], channels*expansion[2], layers[2], stride=2) # 256->128
        self.stack4 = self.make_layer(channels*expansion[2], channels*expansion[3], layers[3], stride=2) # 128->64
        self.stack5 = self.make_layer(channels*expansion[3], channels*expansion[4], layers[4], stride=2) # 64-> 32
        self.stack6 = self.make_layer(channels*expansion[4], channels*expansion[5], layers[5], stride=2) # 32->16

        self.avgpool = nn.AdaptiveAvgPool1d(16)
        self.classifier = ResClassifier(channels*expansion[5]*16, num_classes)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = self.stack5(x)
        x = self.stack6(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def make_layer(self, in_channels, out_channels, blocks, stride):
        if stride != 1:
            downsample = nn.Sequential(
                conv1(in_channels, out_channels, stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            downsample = None
        layers = []
        layers.append(ResBlock(in_channels, out_channels, downsample=downsample, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

def resnet68():
    return ResNetModel(layers=[2, 2, 4, 4, 8, 2]) # 6 Residual Blocks

def resnet152():     ## Name = Sum of Layers * 3 + 2. Total number of learnable layers.
    return ResNetModel(layers = [3, 3, 5, 8, 28, 3])