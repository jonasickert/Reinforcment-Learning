import torch
import torch.nn as nn
import torch.nn.functional as F

class QFunction(nn.Module):
    """
    Neural network model for Q-Learning based on pixel or feature inputs
    
    Attributes:
        - input_type : str
        - fc1 : nn.Linear
        - fc2 : nn.Linear
        - fc3 : nn.Linear
        - conv1 : nn.Conv2d
        - conv2 : nn.Conv2d
        -conv3 : nn.Conv2d
    
    Methods:
        forward(x):
            Does the Forward Pass of the input through the network
    """
    
    def __init__(self, input_dim, output_dim, input_type='features'):
        

        """
        parameters:
        
            input_dim: int or tuple
                - Features is int
                - pixels is tuple
            
            output_dim: int 
                - number of actions
            
            input_type: str
                - features or pixels
                     
        """
        super(QFunction, self).__init__()
        self.input_type = input_type

        if input_type == 'features':
            # Fully connected layers for features
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, output_dim)
        elif input_type == 'pixels':
            # Convolutional layers for pixels
            self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Assuming input image size is 84x84
            self.fc2 = nn.Linear(512, output_dim)
        else:
            raise ValueError("Invalid input_type. Must be 'features' or 'pixels'.")

    def forward(self, x):
        """
        parameters:
            x: torch.tensor
                - for features its 2D (batch_size, input_dim)
                - for pixels its 4D (batch_size, channels, height, width)
        
        returns:
            torch.Tensor
        """
        
        if self.input_type == 'features':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.input_type == 'pixels':
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x

