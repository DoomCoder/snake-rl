import torch
import torch.nn.functional as F

class QNet(torch.nn.Module):
    def __init__(self, D, action_size):
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
        s = F.tanh(self.linear2(s))
        s = F.softmax(self.linear3(s))
        return s


q_loss_fn = torch.nn.MSELoss()

def pi_loss_fn(q_model, state, action):
    qs = q_model(state).detach()
    return -1*torch.mean(action*qs)
