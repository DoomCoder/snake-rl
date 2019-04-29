import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class QNet(torch.nn.Module):
    def __init__(self, D, action_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.D = D
        self.action_size = action_size
        self.conv1 = torch.nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear1 = torch.nn.Linear(32*(int(D/2))**2, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, self.action_size)

    def forward(self, s):
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = self.pool(s)
        s = s.view(-1, 32*(int(self.D/2))**2)
        s = F.relu(self.linear1(s))
        s = F.relu(self.linear2(s))
        s = self.linear3(s)
        return s


class PolicyNet(torch.nn.Module):
    def __init__(self, D, action_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.D = D
        self.action_size = action_size
        self.conv1 = torch.nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear1 = torch.nn.Linear(32*(int(D/2))**2, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, self.action_size)

    def forward(self, s):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # h_relu = self.linear1(x).clamp(min=0)
        # y_pred = self.linear2(h_relu)
        # return y_pred
        # Computes the activation of the first convolution
        # Size changes from (3, D, D) to (16, D, D)
        # s = torch.from_numpy(s).float()
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        # Size changes from (16, D, D) to (16, D/2, D/2)
        s = self.pool(s)
        # Reshape data to input to the input layer of the neural net
        # Size changes from (16, D/2, D/2) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        s = s.view(-1, 32*(int(self.D/2))**2)
        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608+4) to (1, 64)
        s = F.relu(self.linear1(s))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        s = F.tanh(self.linear2(s))
        s = F.softmax(self.linear3(s))
        # s = torch.tanh(self.linear2(s))
        return s


q_loss_fn = torch.nn.MSELoss()


# def pi_loss_fn(q_model, state, action):
#     qs = q_model((state, action))
#     return -qs.mean()

def pi_loss_fn(q_model, state, action):
    # obs_shape = state.shape[1:]
    # batch_size =  state.shape[0]
    # repeat_shape = (batch_size, ) + (4,) + obs_shape
    # repeat_shape = (-1, 4, 1, 1, 1)
    # repeat_states = state.repeat(*repeat_shape)
    # print(repeat_states.shape)
    qs = q_model(state).detach()
    # qs_cpu = qs.cpu()
    # target = np.argmax(qs_cpu, axis=1)
    # target = target.to(device)
    # print(qs)

    # ce = torch.nn.CrossEntropyLoss(size_average=True, reduce='sum')(action, original_actions)
    return -1*torch.mean(action*qs)