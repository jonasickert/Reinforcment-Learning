import torch
import torch.nn as nn
import torch.nn.functional as F

class QFunction(nn.Module):
    
    def __init__(self, input_dim, output_dim, input_type='features'):
        
        super(QFunction, self).__init__()
        self.input_type = input_type

        if input_type == 'features':
            # Fully connected layers for features
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, output_dim)
        elif input_type == 'pixels':
            # Convolutional layers for pixels
            self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(7*7*64, 512)
            self.fc2 = nn.Linear(512, output_dim)
        else:
            raise ValueError("Invalid input_type. Must be 'features' or 'pixels'.")

    def forward(self, x):

        if self.input_type == 'features':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.input_type == 'pixels':
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.reshape(x.size(0), -1)  # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x


    """def __init__(self, input_dim, output_dim, input_type='features'):
        
        parameters:
        input_dim: int or tuple
        - Features is int
        - pixels is tuple
        output_dim: int
        - number of actions
        input_type: str
        - features or pixels

        super(QFunction, self).__init__()
        self.input_type = input_type

        if input_type == 'features':
            # Fully connected layers for features
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, output_dim)
        elif input_type == 'pixels':
            # Convolutional layers for pixels
            self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=2)  # Output: 32 x 39 x 39
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # Output: 64 x 18 x 18
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # Output: 128 x 8 x 8
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)  # Output: 128 x 6 x 6
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)  # Output: 256 x 4 x 4
            self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1)  # Output: 256 x 2 x 2

            self.fc1 = nn.Linear(256 * 2 * 2, 2048)  # Flattened size from conv layers
            self.fc2 = nn.Linear(2048, 1024)
            self.fc3 = nn.Linear(1024, 512)
            self.fc4 = nn.Linear(512, 128)
            self.fc5 = nn.Linear(128, 32)
            self.fc6 = nn.Linear(32, output_dim)
        else:
            raise ValueError("Invalid input_type. Must be 'features' or 'pixels'.")


    def forward(self, x):
        
        parameters:
        x: torch.tensor
        - for features its 2D (batch_size, input_dim)
        - for pixels its 4D (batch_size, channels, height, width
        returns:
        torch.Tensor
        
        if self.input_type == 'features':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.input_type == 'pixels':
            x = nn.ReLU()(self.conv1(x))
            x = nn.ReLU()(self.conv2(x))
            x = nn.ReLU()(self.conv3(x))
            x = nn.ReLU()(self.conv4(x))
            x = nn.ReLU()(self.conv5(x))
            x = nn.ReLU()(self.conv6(x))
            x = x.reshape(x.size(0), -1)  # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = self.fc6(x)
        return x"""

