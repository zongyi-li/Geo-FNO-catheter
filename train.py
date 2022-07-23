"""
r: Zongyi Li and Daniel Zhengyu Huang
"""

import numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from catheter import *
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


################################################################
# configs
################################################################
# load data
PATH = "/groups/esm/dzhuang/Catheter/"

INPUT_X = PATH+"x_2d_structured_mesh.npy"
INPUT_Y = PATH+"y_2d_structured_mesh.npy"
OUTPUT = PATH+"density_1d_data.npy"

ntrain = 800
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 501
step_size = 100
gamma = 0.5

modes = 12
width = 32

r1 = 1
r2 = 2
# nx ny
s1 = int(((101 - 1) / r1) + 1)
s2 = int(((401 - 1) / r2) + 1)
s3 = 2001

################################################################
# load data and data normalization
################################################################
inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float).permute(2,0,1)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float).permute(2,0,1)
print(inputX.shape, inputY.shape)
input = torch.stack([inputX, inputY], dim=-1)

output = np.load(OUTPUT)
output = torch.tensor(output, dtype=torch.float).permute(1,0)
print(input.shape, output.shape)

index = torch.randperm(1000)
train_index = index[:ntrain]
test_index = index[-ntest:]

x_train = input[train_index][:, ::r1, ::r2][:, :s1, :s2]
y_train = output[train_index]
x_test = input[test_index, ::r1, ::r2][:, :s1, :s2]
y_test = output[test_index]
x_train = x_train.reshape(ntrain, s1, s2, 2)
x_test = x_test.reshape(ntest, s1, s2, 2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          shuffle=False)


################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes*2, width).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)

    # plot
    if ep%step_size==0:
        torch.save(model, 'catheter_plain_model_'+str(ep))

