import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split



# Lets define our model
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.block = nn.Sequential(
                nn.Linear(16, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                # nn.BatchNorm1d(128),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
    def forward(self, x):
        return self.block(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model().to(device)
loss_function = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


x, y = np.load("x.npy"), np.load("y.npy")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, shuffle=True)
x_train, x_test, y_train, y_test = torch.Tensor(x_train).to(device), torch.Tensor(x_test).to(device), torch.Tensor(y_train).to(device), torch.Tensor(y_test).to(device)
x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1], 16)
y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], 1) 


writer = SummaryWriter('runs/final')


for epoch in range(100):
    for xx, yy in zip(x_train, y_train):
        xx, yy = xx.to(device), yy.to(device)

        outputs = net(xx)
        loss = loss_function(outputs, yy)
        optimizer.zero_grad()
        net.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = net(x_test)
        l = loss_function(output, y_test)
        print(f"training_loss: {loss}, testing_loss: {l}")
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/test", l, epoch)

    writer.flush()
writer.close()

checkpoint = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
torch.save(checkpoint, "weights.pth")
