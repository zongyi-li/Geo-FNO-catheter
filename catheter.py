"""
r: Zongyi Li and Daniel Zhengyu Huang
"""
from scipy import stats
import torch
import numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, output_size=None):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        if output_size == None:
            x = torch.fft.irfft(out_ft, n=x.shape[-1])
        else:
            x = torch.fft.irfft(out_ft, n=output_size)
            
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fc3 = nn.Linear(101, 4)
        self.fc4 = nn.Linear(4*self.width, self.width)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]

        # sum over x-axis
        # x (batch, channel, x, y)  -> (batch, channel, y')
        x = x.permute(0, 3, 1, 2)
        #  x (batch, y, channel, x,)
        x = self.fc3(x)
        x = F.gelu(x)
        #  x (batch, y, channel, 4)
        x = x.flatten(2,3)
        # x(batch, y, channel * 4)
        x = self.fc4(x)
        # x(batch, y, channel)
        x = F.gelu(x)
        x = x.permute(0, 2, 1)
        # x(batch, channel, y)
        x1 = self.conv4(x, 2001)
        x2 = F.interpolate(x, 2001, mode='linear', align_corners=True)
        x = x1 + x2
        # x(batch, channel, 2001)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


    
class FNO1d(nn.Module):
    def __init__(self, modes, width, padding=100, input_channel=2, output_np=2001):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.output_np=output_np
        self.modes1 = modes
        self.width = width
        self.padding = padding # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_channel, self.width) # input channel is 2: (a(x), x)
        
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)

        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
      

    def forward(self, x):
        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x = x[..., :-self.padding]
        # x(batch, channel, y)
        x1 = self.conv4(x, self.output_np)
        x2 = F.interpolate(x, self.output_np, mode='linear', align_corners=True)
        x = x1 + x2
        # x(batch, channel, 2001)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
      
##################################################################
# 
###################################################################

# return the mesh for [-L_p, 0]
# x1, x2 and x3 are positions
def numpy_catheter_mesh_1d_single_period(L_p, x1, x2, x3, h, ncx1, ncx2, ncx3, ncx4):
    # between [-L_p, 0]
    
    # ncx1, ncx2, ncx3, ncx4 = 20, 10, 10, 20
    ncx = ncx1 + ncx2 + ncx3 + ncx4
    xx = np.hstack((np.linspace(-L_p, x1, ncx1,endpoint=False), np.linspace(x1, x2, ncx2,endpoint=False), 
                     np.linspace(x2, x3, ncx3,endpoint=False), np.linspace(x3, 0, ncx4+1)))
    
    yy = np.zeros(ncx+1)
    yy[ncx1:ncx1+ncx2] = (xx[ncx1:ncx1+ncx2] - x1)/(x2 - x1)*h
    yy[ncx1+ncx2:ncx1+ncx2+ncx3+1] = (x3 - xx[ncx1+ncx2:ncx1+ncx2+ncx3+1])/(x3 - x2)*h
    return xx, yy


def numpy_d2xy(d, L_p, x1, x2, x3, h):
    
    p0, p1, p2, p3 = np.array([0.0,0.0]), np.array([x3,0.0]), np.array([x2, h]), np.array([x1,0.0])
    v0, v1, v2, v3 = np.array([x3-0,0.0]), np.array([x2-x3,h]), np.array([x1-x2,-h]), np.array([-L_p-x1,0.0])
    l0, l1, l2, l3 = -x3, np.sqrt((x2-x3)**2 + h**2), np.sqrt((x1-x2)**2 + h**2), L_p+x1
    if d < l0:
        x, y = d*v0/l0 + p0
    elif d < l0 + l1:
        x, y = (d-l0)*v1/l1 + p1 
    elif d < l0 + l1 + l2:
        x, y = (d-l0-l1)*v2/l2 + p2
    else:
        x, y = (d-l0-l1-l2)*v3/l3 + p3

    return x, y

def numpy_Lx2length(L_x, L_p, x1, x2, x3, h):
    assert(L_x < L_p)
    p0, p1, p2, p3 = np.array([0.0,0.0]), np.array([x3,0.0]), np.array([x2, h]), np.array([x1,0.0])
    v0, v1, v2, v3 = np.array([x3-0,0.0]), np.array([x2-x3,h]), np.array([x1-x2,-h]), np.array([-L_p-x1,0.0])
    l0, l1, l2, l3 = -x3, np.sqrt((x2-x3)**2 + h**2), np.sqrt((x1-x2)**2 + h**2), L_p+x1
    assert(L_x <= l0+l1+l2+l3)
    if L_x < -x3:
        l = L_x
    elif L_x < -x2:
        l = l0 + l1*(L_x + x3)/(x3-x2)
    elif L_x < -x1:
        l = l0 + l1 + l2*(L_x + x2)/(x2-x1)
    else:
        l = l0 + l1 + l2 + L_x+x1

    return l

# return the mesh points in [-L_x, 0], each period length is L_p
# x1, x2 and x3 are positions in the first period in the right
def numpy_catheter_mesh_1d_total_length(L_x, L_p, x1, x2, x3, h, N_s):
    # between [-L_p, 0]
    
    n_periods = np.floor(L_x / L_p)
    L_x_last_period = L_x - n_periods*L_p
    L_p_s = ((x1 + L_p) + (0 - x3) + np.sqrt((x2 - x1)**2 + h**2) + np.sqrt((x3 - x2)**2 + h**2))
    L_s = L_p_s*n_periods + numpy_Lx2length(L_x_last_period, L_p, x1, x2, x3, h)
    
    # from 0
    d_arr = np.linspace(0, L_s, N_s)
    period_arr = np.floor(d_arr / L_p_s)
    d_arr -= period_arr * L_p_s
    
    xx = np.zeros(N_s) 
    yy = np.zeros(N_s)
    for i in range(N_s):
        xx[i], yy[i] = numpy_d2xy(d_arr[i], L_p, x1, x2, x3, h)
        xx[i] -= period_arr[i]*L_p
    return xx, yy



def preprocess_period(ind, t,  data_info, file_name, ncx1=50, ncx2=50, ncx3=50, ncx4 = 50, n_periods = 5, bw_method = 1e-1):
    
    
    ncx = ncx1 + ncx2 + ncx3 + ncx4
    # user chooses t and x2 x3 h
    # check t is consistent with data
    sample  = np.int64(  file_name[file_name.find("sample") + len("sample"):  file_name.find("_U")]  )
    uf  = np.float64(  file_name[file_name.find("uf") + len("uf"):  file_name.find("alpha")]  )
    L_p, x2, x3, h, press = data_info[sample - 1, :]

    assert((h > 20 and h < 30) and (15 < x3 and x3 < L_p/4) and (-L_p/4 < x2 and x2 < L_p/4))
    # generate mesh
    x1 = -0.5*L_p
    x2 = x1 + x2
    x3 = x1 + x3
    X, Y = numpy_catheter_mesh_1d_single_period(L_p, x1, x2, x3, h, ncx1, ncx2, ncx3, ncx4)
    x_mesh, y_mesh = np.zeros(n_periods*ncx+1), np.zeros(n_periods*ncx+1)
    x_mesh[-(ncx + 1):], y_mesh[-(ncx + 1):] = X, Y    
    for i_period in range(1,n_periods):
        x_mesh[-((i_period + 1)*ncx + 1):-(i_period*ncx)], y_mesh[-((i_period + 1)*ncx + 1):-(i_period*ncx)] = X - L_p*i_period, Y
    
    # preprocee density
    hf = h5py.File(file_name, "r")
    x_b = hf["config"][str(t+1)]["x"][:]
    y_b = hf["config"][str(t+1)]["y"][:]
    
    if(min(x_b) < -L_p*n_periods):
        print("warning: bacteria out of the domain. ind = ", ind, ", file_name = ", file_name, " loc = ", min(x_b), " end point = ", -L_p*n_periods)
    
    nc = ncx
    N_s = n_periods*nc + 1
    xx = np.linspace(-L_p*n_periods, 0.0, N_s)
    
    bacteria_1d_data = x_b[np.logical_and(x_b <= 0 , x_b >= -L_p*n_periods)]
    n_particle = len(bacteria_1d_data)
    kernel = stats.gaussian_kde(bacteria_1d_data, bw_method = bw_method)
    density_1d_data = kernel(xx)*n_particle

    
    return x_mesh, y_mesh, x_b, y_b, xx, density_1d_data, np.array([sample, uf, L_p, x1, x2, x3, h])






def preprocess_length(ind, t,  data_info, file_name, N_s = 50, L_x = 5, bw_method = 1e-1):

    sample  = np.int64(  file_name[file_name.find("sample") + len("sample"):  file_name.find("_U")]  )
    uf  = np.float64(  file_name[file_name.find("uf") + len("uf"):  file_name.find("alpha")]  )
    L_p, x2, x3, h, press = data_info[sample - 1, :]

    assert((h > 20 and h < 30) and (15 < x3 and x3 < L_p/4) and (-L_p/4 < x2 and x2 < L_p/4))
    # generate mesh
    x1 = -0.5*L_p
    x2 = x1 + x2
    x3 = x1 + x3
    x_mesh, y_mesh = numpy_catheter_mesh_1d_total_length(L_x, L_p, x1, x2, x3, h, N_s)

    # preprocee density
    hf = h5py.File(file_name, "r")
    x_b = hf["config"][str(t+1)]["x"][:]
    y_b = hf["config"][str(t+1)]["y"][:]
    
    if(min(x_b) < -L_x):
        print("warning: bacteria out of the domain. ind = ", ind, ", file_name = ", file_name, " loc = ", min(x_b), " end point = ", -L_x)
    

    xx = np.linspace(-L_x, 0.0, N_s)
    
    bacteria_1d_data = x_b[np.logical_and(x_b <= 0 , x_b >= -L_x)]
    n_particle = len(bacteria_1d_data)
    kernel = stats.gaussian_kde(bacteria_1d_data, bw_method = bw_method)
    density_1d_data = kernel(xx)*n_particle

    
    return x_mesh, y_mesh, x_b, y_b, xx, density_1d_data, np.array([sample, uf, L_p, x1, x2, x3, h])

# ################################################################
# # configs
# ################################################################
# # load data
# PATH = "/groups/esm/dzhuang/Catheter/"

# INPUT_X = PATH+"x_2d_structured_mesh.npy"
# INPUT_Y = PATH+"y_2d_structured_mesh.npy"
# OUTPUT = PATH+"density_1d_data.npy"

# ntrain = 800
# ntest = 200

# batch_size = 20
# learning_rate = 0.001

# epochs = 501
# step_size = 100
# gamma = 0.5

# modes = 12
# width = 32

# r1 = 1
# r2 = 2
# # nx ny
# s1 = int(((101 - 1) / r1) + 1)
# s2 = int(((401 - 1) / r2) + 1)
# s3 = 2001

# ################################################################
# # load data and data normalization
# ################################################################
# inputX = np.load(INPUT_X)
# inputX = torch.tensor(inputX, dtype=torch.float).permute(2,0,1)
# inputY = np.load(INPUT_Y)
# inputY = torch.tensor(inputY, dtype=torch.float).permute(2,0,1)
# print(inputX.shape, inputY.shape)
# input = torch.stack([inputX, inputY], dim=-1)

# output = np.load(OUTPUT)
# output = torch.tensor(output, dtype=torch.float).permute(1,0)
# print(input.shape, output.shape)

# index = torch.randperm(1000)
# train_index = index[:ntrain]
# test_index = index[-ntest:]

# x_train = input[train_index][:, ::r1, ::r2][:, :s1, :s2]
# y_train = output[train_index]
# x_test = input[test_index, ::r1, ::r2][:, :s1, :s2]
# y_test = output[test_index]
# x_train = x_train.reshape(ntrain, s1, s2, 2)
# x_test = x_test.reshape(ntest, s1, s2, 2)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
#                                           shuffle=False)
# test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
#                                           shuffle=False)


# ################################################################
# # training and evaluation
# ################################################################
# model = FNO2d(modes, modes*2, width).cuda()
# print(count_params(model))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# myloss = LpLoss(size_average=False)

# for ep in range(epochs):
#     model.train()
#     t1 = default_timer()
#     train_l2 = 0
#     for x, y in train_loader:
#         x, y = x.cuda(), y.cuda()

#         optimizer.zero_grad()
#         out = model(x)

#         loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
#         loss.backward()

#         optimizer.step()
#         train_l2 += loss.item()

#     scheduler.step()

#     model.eval()
#     test_l2 = 0.0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.cuda(), y.cuda()

#             out = model(x)
#             test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

#     train_l2 /= ntrain
#     test_l2 /= ntest

#     t2 = default_timer()
#     print(ep, t2 - t1, train_l2, test_l2)

#     # plot
#     if ep%step_size==0:
#         torch.save(model, '../catheter_plain_model_'+str(ep))

