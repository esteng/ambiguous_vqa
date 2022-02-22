import torch 
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb 

class NeuralNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, dropout):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(n_features, n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, replace_x = None):
        
        if replace_x is not None:
            x = self.sigmoid(self.fc1(x))
            # output = self.sigmoid(self.fc1(replace_x))
            output = F.dropout(replace_x, self.dropout, training=self.training)
            output = self.fc2(output)
            return output, None
        else:
            middle = self.sigmoid(self.fc1(x))
            x = F.dropout(middle, self.dropout, training=self.training)
            x = self.fc2(x)
        return x, middle


import torch.optim as optim
model = NeuralNet(n_features=8,
            n_hidden=32,
            n_classes=1,
            dropout=0.0)

optimizer = optim.SGD(model.parameters(), lr=0.1)
features = torch.Tensor(np.random.rand(10, 8))
labels = torch.Tensor(np.random.rand(10, 1))
loss = nn.MSELoss()
def train(epoch):
    t = time.time()
    model.train()
    
    optimizer.zero_grad()
    output, middle = model(features)
    loss_train = loss(output, labels)
    print(loss_train)
    loss_train.backward()
    optimizer.step()
    return middle 
    # grad_features = loss_train.backward() w.r.t to features
    # features -= 0.001 * grad_features

for epoch in range(10):
    middle = train(epoch)

model.zero_grad()
feature = features[0,:].unsqueeze(0)
answer = labels[0,:].unsqueeze(0)
feature.requires_grad_(True)
model.requires_grad_(False)
model.eval() 
new_optimizer = optim.SGD([feature], lr = 1.0)
random_x = torch.Tensor(np.random.rand(1, 8))
new_feature = middle.detach() 
new_feature.requires_grad_(True)
print(feature.shape)
pdb.set_trace() 
output, middle2 = model(x = random_x, replace_x = new_feature)
loss_val = loss(output, answer)
loss_val.backward()
pdb.set_trace()