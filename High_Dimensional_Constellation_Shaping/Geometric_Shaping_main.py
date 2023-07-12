#%%Importing all the required packages
import torch as tr
import numpy as np
import time
import Functions.Geometric_Shaping as gs
#%% Defining some of the parameters of the optimization
M = 64 # The cardinality size of the constellation
m = int(np.log2(M)) #The amount of bits per symbol
channel_uses = 4 # The dimensionality of the constellation
learning_rate = 0.1 # Learning Rate, 0.1 works very well, and therefore does not need to be changed
EsNo_dB = 7 # The SNR at which to optimise the constellation
optimisation_metric = 'GMI' # The metrix which to optimise for, either 'MI' or 'GMI'
epochs = 100 # The amount of optimisation iterations
bpg = 6 # The amount of bits which are being calculated simultaneously per backward propagation, only used for GMI optimisation in order to limit memory usage
Device = 'cuda' # Determines the device which the optimization is done on, 'cpu' for cpu and 'cuda:0', 'cuda:1' etc. for GPU
Initial_Constellation = 'APSK1' #Look at the Geometric_Shaping.py file for all possible options, only relevant for GMI optimisation
Quad_amount = 128 # The amount of quadratures used for the optimisation using 'RQ', normally we choose 16 for 2D, 128 for 4D, 256 for 8D, and 512 for 12D.
Quad_base =  tr.from_numpy(gs.Quad_initialization(Quad_amount,channel_uses)).to(Device) # Generating the random quadratures
sigma2 = 1/(channel_uses*10**(EsNo_dB/10)) # noise variance per channel use
X_tilde = tr.eye(M).to(Device) # The input to our neural network
#%% Training the model
start_time = time.time()
Constellations = np.zeros((M,channel_uses,epochs)) #For saving the constellations
encoder = tr.nn.Sequential() #Setting up the 1 layer neural network used for optimisation
encoder.add_module('last', tr.nn.Linear(M,channel_uses,bias = False))
encoder.to(Device)

if optimisation_metric == 'GMI':
    X_target,idx_train,Initial_Constellation = gs.Constellation_Initialization(M, channel_uses, Initial_Constellation, Device) #Set up Initial Constellation
    encoder.last.weight = tr.nn.Parameter(X_target.T)
    optimizer = tr.optim.Adam(encoder.parameters(), learning_rate)
    for i in range(1, epochs+1):
        optimizer.zero_grad()
        gs.GMI_RQ(X_tilde, idx_train, EsNo_dB, encoder, Quad_base, Device, bpg)
        Constellations[:,:,i-1] = gs.normalization(encoder(X_tilde)).detach().cpu().numpy()
        if i%10 == 0 or i ==1:
            print('iter ', i, 'time', time.time()-start_time)
        optimizer.step()
        gs.save_GMI(Constellations[:,:,0:i], EsNo_dB, idx_train, epochs, learning_rate, Initial_Constellation, start_time)
else:
    optimizer = tr.optim.Adam(encoder.parameters(), learning_rate)
    for i in range(1,epochs+1):
        optimizer.zero_grad()
        gs.MI_RQ(X_tilde, EsNo_dB, encoder, Quad_base, Device)
        Constellations[:,:,i-1] = gs.normalization(encoder(X_tilde)).detach().cpu().numpy()
        if i%10 == 0 or i ==1:
            print('iter ', i, 'time', time.time()-start_time)
        optimizer.step()
        gs.save_MI(Constellations[:,:,0:i], EsNo_dB, epochs, learning_rate, start_time)
#%% Plotting The figure
X = Constellations[:,:,-1]
if(optimisation_metric == 'MI'):
    gs.plot_constellation(X)
else:
    gs.plot_constellation_labeling(X, idx_train)