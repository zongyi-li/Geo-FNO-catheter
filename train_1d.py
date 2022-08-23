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
PATH = "/groups/esm/dzhuang/Catheter/allparam/"

INPUT_X = PATH+"x_1d_structured_mesh.npy"
INPUT_Y = PATH+"y_1d_structured_mesh.npy"
INPUT_para = PATH+"data_info.npy"
OUTPUT = PATH+"density_1d_data.npy"

ntrain = 2000
ntest = 1000

batch_size = 20
learning_rate = 0.001

epochs = 501
step_size = 100
gamma = 0.5



n_periods = 4

modes = n_periods*16
width = n_periods*32
# nx ny
s = n_periods*200 + 1
s3 = n_periods*1000 + 1

################################################################
# load data and data normalization
################################################################
inputX = np.load(INPUT_X)
print("inputX.shape", inputX.shape)
inputX = torch.tensor(inputX, dtype=torch.float).permute(1,0)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float).permute(1,0)

# setup inputZ
inputPara = np.load(INPUT_para)


# n_features = 1
# n_data, n_p = inputX.shape
# inputZ = np.zeros((n_data, n_p, n_features))
# for i in range(n_features):
#     inputZ[:,:,i] = np.outer(inputPara[1, :], np.ones(n_p))
# inputZ = torch.tensor(inputZ, dtype=torch.float)

n_features = 3
n_data, n_p = inputX.shape
inputZ = np.zeros((n_data, n_p, n_features))
uf_code = np.array([[1,0,0],[0,1,0],[0,0,1]])
for i in range(n_data):
    # u_f 0, 7.5, 15
    uf_type = 0
    if inputPara[1, i] < (0 + 7.5)/2:
        uf_type = 0
    elif inputPara[1, i] < (7.5 + 15)/2:
        uf_type = 1
    else:
        uf_type = 2
        
    inputZ[i,:,:3] = np.outer(np.ones(n_p), uf_code[uf_type, :])
    
inputZ = torch.tensor(inputZ, dtype=torch.float)



input = torch.stack([inputX, inputY], dim=-1)
input = torch.cat([input, inputZ], dim=2)
print("n_data, n_p, n_features = ", input.shape)



output = np.load(OUTPUT)
output = torch.tensor(output, dtype=torch.float).permute(1,0)
print(input.shape, output.shape)

index = torch.randperm(ntrain+ntest)
train_index = index[:ntrain]
test_index = index[-ntest:]

x_train = input[train_index]
y_train = output[train_index]
x_test = input[test_index]
y_test = output[test_index]
x_train = x_train.reshape(ntrain, s, n_features+2)
x_test = x_test.reshape(ntest, s, n_features+2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          shuffle=False)

if __name__ == "__main__":
    ################################################################
    # training and evaluation
    ################################################################
    model = FNO1d(modes, width).cuda()
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
            torch.save(model, 'catheter_plain_model_1d'+str(ep))

