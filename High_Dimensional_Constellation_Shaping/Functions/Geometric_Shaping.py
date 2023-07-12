# Importing Packages and defining variables
import numpy as np
import cupy as cp
import torch as tr
import scipy.io as sio
import time
import matplotlib.pyplot as plt
GH =  sio.loadmat('Functions/GaussHermite_J_10.mat')#Loading in the Gauss-Hermite points
#%% Defining the functions

# A function for initializing the quadratures used in the optimization with the RQ estimation method
# Quad_Amount: The amount of quadratures
# channel_uses: The amount of channel_uses (or dimensions) of the constellations
# Output: A matrix containing the quadratures and the associated weights
def Quad_initialization(Quad_Amount, channel_uses):
    xi = np.random.normal(0,1,(Quad_Amount,channel_uses)) #Randomly sample quadratures from Gaussian distribution
    alpha = np.exp(-np.sum(xi**2,1)/2) #Calculate the weights of the quadratures
    return np.column_stack((xi,alpha))

# A function for turning decimal numbers into binary numbers
# d: The decimal numbers to be converted
# m: The amount of bits used in the binary representation
# Output: A matrix containing the binary numbers
def de2bi(d, m):
    d = np.array(d)
    power = 2**np.arange(m)
    return (np.floor((d[:,None]%(2*power))/power))

# A function for turning binary numbers into decimal numbers
# L: A N x m matrix containing the binary numbers, where m is the amount of bits, and N the total amount of binary numbers 
# Output: An array of decimal numbers
def bi2de(L):
    m = np.size(L,1)
    Ldec = np.zeros((int(2**m),1), dtype = int)
    for i in range(int(m)):
        for j in range(int(2**m)):
            Ldec[j] = Ldec[j] + L[j,i]*2**i 
    return Ldec

# A function for generating gray labeloing
# m: The amount of bits used in the gray labeling
# Output: The gray labeling for a constellation with m bits    
def get_labeling(m):
    M  = 2**m
    if m == 1:
        L = np.asarray([[0],[1]])
    else:
        L = np.zeros((M,m))
        L[0:int(M/2),1:m] = get_labeling(m-1)
        L[int(M/2):M, 1:m] = np.flipud(L[0:int(M/2),1:m])
        L[int(M/2):M,0] = 1
    return L

#A function for generating a 2D QAM constellation
# M: The amount of constellation points
# Output: A 2D QAM constellation and its binary labeling
def get_constellation(M):
    Delta = np.sqrt(3/(2*(M-1)))
    Xpam = np.expand_dims(Delta*np.arange(-(np.sqrt(M)-1),np.sqrt(M)+1,2),axis = 1)
    xpamt_2D = np.tile(Xpam,(1,int(np.sqrt(M))))
    xpamt = np.expand_dims(xpamt_2D.flatten(),axis = 1)
    X = np.transpose(np.reshape(np.asarray([xpamt, np.tile(Xpam,(int(np.sqrt(M)),1))]),(2,M)))
    Ltmp = get_labeling(int(np.log2(M)/2))
    Ltmp_dec = bi2de(Ltmp)
    Ltmp_dec2 = np.tile(Ltmp_dec,(1,int(np.sqrt(M))))
    Ltmp_dec3 = np.expand_dims(Ltmp_dec2.flatten(),axis = 1)
    L = np.concatenate((np.reshape(de2bi(Ltmp_dec3,int(np.log2(M)/2)),(M,int(np.log2(M)/2))), np.tile(Ltmp,(int(np.sqrt(M)),1))), axis = 1)
    Ldec = np.reshape(np.asarray(bi2de(np.fliplr(L)),dtype = int),M)
    return [X,Ldec]

#A function for generating an N-D PAM
# D: The amount of dimensions
# symbols_per_dimension: The amount fo symbols per dimension
# Output: A constellation which is an ND PAM
def Generate_ND_PAM(D,symbols_per_dimension):
    X = np.zeros((D,symbols_per_dimension**D))
    Positions = np.arange(-1*symbols_per_dimension+1,symbols_per_dimension,2)
    for i in range(D):
        idx = 0
        for j in range(symbols_per_dimension**(i+1)):
            X[i,idx:idx+symbols_per_dimension**(D-i-1)] = Positions[np.mod(j,symbols_per_dimension)]
            idx = idx + symbols_per_dimension**(D-i-1)
    return X.T

#A function for generating a 2D APSK constellation
# M2: The amount of constellation points
# Amount of Rings: The amount of rings in the APSK constellation, it is assumed each ring has the same amount of points, so M has to be divisible by this number
# Output: A 2D APSK constellation and its binary labeling
def get_APSK(M2,Amount_of_Rings):
    M = int(M2/Amount_of_Rings) * np.ones(Amount_of_Rings, dtype = int)
    X = np.zeros((sum(M),2))
    if Amount_of_Rings > 1:
        idx = 0
        l_r1 = get_labeling(int(np.log2(M[0])))*M.shape[0]
        ldec_r1 = bi2de(l_r1)
        l_rs = get_labeling(int(np.log2(M.shape[0])))
        ldec_rs = bi2de(l_rs)
        l_apsk = np.zeros((sum(M),1),dtype = int)
        for i in range(M.shape[0]):
            R = np.sqrt(-np.log(1-(i+0.5)*1/M.shape[0]))
            for j in range(M[i]):
                X[idx+j,:] = [R*np.cos(j*2*np.pi/M[i]), R*np.sin(j*2*np.pi/M[i]) ]
            l_apsk[idx:idx+M[i],] = ldec_r1 + ldec_rs[i]
            idx = idx + M[i]
        l_apsk = np.squeeze(l_apsk)
    else:
        Lbin = get_labeling(int(np.log2(M[0])))
        l_apsk = np.squeeze(bi2de(Lbin))
        for j in range(M2):
             X[j,:] = [np.cos(j*2*np.pi/M2), np.sin(j*2*np.pi/M2)]
    return [X, l_apsk]

#A function for generating a 4D QAM constellation
# M: The amount of constellation points
# Output: A 4D QAM constellation and its binary labeling
def get_constellation_4D(M):
    m = int(np.log2(M))
    m_pd = int(m/4)
    Delta = np.sqrt(np.sqrt(3/(2*(M-1))))
    Xpam = np.expand_dims(Delta*np.arange(-(np.sqrt(np.sqrt(M))-1),np.sqrt(np.sqrt(M))+1,2),axis = 1)
    Lpam = get_labeling(m_pd)
    X1 =  (np.expand_dims(np.ones(int(M/2**m_pd)),axis= 1)*np.transpose(Xpam)).flatten('F')
    X2 =  np.tile((np.expand_dims(np.ones(int(M/2**(m_pd*2))),axis= 1)*np.transpose(Xpam)).flatten('F'), 2**m_pd)
    X3 =  np.tile((np.expand_dims(np.ones(int(M/2**(m_pd*3))),axis= 1)*np.transpose(Xpam)).flatten('F'), 2**(m_pd*2))
    X4 =  np.tile((np.expand_dims(np.ones(int(M/2**(m_pd*4))),axis= 1)*np.transpose(Xpam)).flatten('F'), 2**(m_pd*3))
    L1 =  np.reshape(np.tile(Lpam,(int(M/2**m_pd))), (M,m_pd))
    L2 =  np.tile(np.reshape(np.tile(Lpam,(int(M/2**(m_pd*2)))), (int(M/2**m_pd),m_pd)),(2**m_pd,1))
    L3 =  np.tile(np.reshape(np.tile(Lpam,(int(M/2**(m_pd*3)))), (int(M/2**(m_pd*2)),m_pd)),(2**(m_pd*2),1))
    L4 =  np.tile(np.reshape(np.tile(Lpam,(int(M/2**(m_pd*4)))), (int(M/2**(m_pd*3)),m_pd)),(2**(m_pd*3),1))
    X = np.transpose(np.asarray([X1,X2,X3,X4]))
    Lbin =np.asarray(np.concatenate((L1,L2,L3,L4),axis = 1), dtype = int)
    Ldec = np.squeeze(bi2de(Lbin))
    return [X,Ldec]  

#A function for generating an 8D APSK constellation
# M: The amount of constellation points
# Amount of Rings: The amount of rings in the APSK constellation per 2D, it is assumed each ring has the same amount of points
# Output: An 8D APSK constellation and its binary labeling
def get_APSK_8D(M, Amount_of_Rings):
    m = int(np.log2(M))
    M2 = int(np.sqrt(M))
    X_4D, l_apsk_4D = get_APSK_4D(M2,Amount_of_Rings)
    Lbin_4D = de2bi(l_apsk_4D, int(m/2))
    X1 = np.repeat(X_4D,M2,axis = 0)
    X2 = np.reshape(np.repeat(X_4D,M2,axis = 1), (M,4), 'F')
    L1 = np.repeat(Lbin_4D, M2, axis = 0)
    L2 = np.reshape(np.repeat(Lbin_4D,M2,axis = 1), (M,int(m/2)), 'F')
    X = np.concatenate([X1,X2], axis = 1)
    Lbin = np.concatenate([L1,L2], axis = 1)
    Ldec = np.squeeze(bi2de(Lbin))
    return[X,Ldec]

#A function for generating a 4D APSK constellation
# M2: The amount of constellation points
# Amount of Rings: The amount of rings in the APSK constellation per 2D, it is assumed each ring has the same amount of points
# Output: A 4D APSK constellation and its binary labeling
def get_APSK_4D(M, Amount_of_Rings):
    m = int(np.log2(M))
    M2 = int(np.sqrt(M))
    X_2D, l_apsk_2D = get_APSK(M2,Amount_of_Rings)
    Lbin_2D = de2bi(l_apsk_2D, int(m/2))
    X1 = np.repeat(X_2D,M2,axis = 0)
    X2 = np.reshape(np.repeat(X_2D,M2,axis = 1), (M,2), 'F')
    L1 = np.repeat(Lbin_2D, M2, axis = 0)
    L2 = np.reshape(np.repeat(Lbin_2D,M2,axis = 1), (M,int(m/2)), 'F')
    X = np.concatenate([X1,X2], axis = 1)
    Lbin = np.concatenate([L1,L2], axis = 1)
    Ldec = np.squeeze(bi2de(Lbin))
    return[X,Ldec]

# A function for generating a SP 12D BPSK with a total of 11 bits
# Output: A SP-12D-BPSK constellation with a total of 11 bits and its binary labeling
def get_BPSK_11bits_12D():
    X = Generate_ND_PAM(12, 2) #Generate a 12D BPSK
    X = X[0:2048,:] #Remove half of the points
    Ldec = np.arange(0,2048) #Generate the binary labeling
    return X, Ldec

#A function for generating a higher dimensional constellation from combining lower dimensional constellations
# X: The input lower dimensional constellation
# idx: The input lower dimensional labeling in decimal
# D: The target dimension of the output constellation
# Output: A higher dimensional constellation with its labeling in decimal based on a lower dimensional constellation
def get_Optimised_Lower_Dims(X,Ldec,D):
    D_lower = np.size(X,1)
    D_ratio =int(D/D_lower)
    M_lower = np.size(X,0)
    M_higher = int(M_lower**(D_ratio))
    m_lower = int(np.log2(M_lower))
    X_out = np.zeros((M_higher,D))
    Lbin = de2bi(Ldec, m_lower)
    Lbin_out = np.zeros((M_higher,int(np.log2(M_higher))))
    for i in range(M_higher):
        for j in range(D_ratio):
            X_out[i,j*D_lower:(j+1)*D_lower] = X[int(np.mod(np.floor(i/(M_lower**j)),M_lower)),:]
            Lbin_out[i,j*m_lower:(j+1)*m_lower] = Lbin[int(np.mod(np.floor(i/(M_lower**j)),M_lower)),:]
    Lbin_out_dec = bi2de(Lbin_out)
    return X_out, Lbin_out_dec

#A function for normalizing the power of a constellation, used for tensors
# x: The constellation
# Output: A normalised constellation with total average power 1
def normalization(x): # E[|x|^2] = 1
    channel_uses = x.size(1)
    return x / tr.sqrt((channel_uses*(x**2)).mean())

#A function for normalizing the power of a constellation, used for numpy arrays
# x: The constellation
# Output: A normalised constellation with total average power 1
def normalization_np(x):
    D = np.size(x,1)
    return x / np.sqrt((D*(x**2)).mean())

#A function for normalizing the power of a constellation, used for cupy arrays
# x: The constellation
# Output: A normalised constellation with total average power 1
def normalization_cp(x):
    D = cp.size(x,1)
    return x / cp.sqrt((D*(x**2)).mean())

# A function for generating the initial constellation for use in the optimisation
# M: the cardinality of the constellation
# channel_uses: The dimensiolity of the constellation
# Initial_Constellations: The name of the initial constellation, there are the following options:
    # 'QAM': a standard QAM constellation, available for 2D and 4D
    # 'APSK' + 'N': an APSK constellation with N rings per 2D, so if you want to have an APSK constellation with 4 rings per 2D you would input 'APSK4'
    # 'SP-BPSK': A 12D BPSK constellation set-partioned to have 11 bits
    # 'Path to initial constellation of lower dimensionality': The last choice is to give a path to a lower dimensional constellation saved using the save function. The initial constellation will be a combination of lower dimensional optimised constellations
# Output: The initial constellation and its decimal labeling
def Constellation_Initialization(M,channel_uses,Initial_Constellation, Device):
    if channel_uses == 2:
        if Initial_Constellation == 'QAM':
            [X_target,idx_train]= get_constellation(M)
        else:
            Amount_of_Rings = int(Initial_Constellation[4::])
            [X_target,idx_train] = get_APSK(M,Amount_of_Rings)
            Initial_Constellation = Initial_Constellation + '_' + str(Amount_of_Rings) + 'Rings'
    elif channel_uses == 4:
        if Initial_Constellation == 'QAM':
            [X_target,idx_train]= get_constellation_4D(M)
        elif Initial_Constellation[:4] == 'APSK':
            Amount_of_Rings = int(Initial_Constellation[4::])
            [X_target,idx_train] = get_APSK_4D(M,Amount_of_Rings)
            Initial_Constellation = Initial_Constellation + '_' + str(Amount_of_Rings) + 'Rings'
        else:
            Optimized_Constellation_LD = np.load(Initial_Constellation, allow_pickle = True)
            X_LD = Optimized_Constellation_LD[0][:,:,-1]
            l_LD =  Optimized_Constellation_LD[2]
            [X_target,idx_train] = get_Optimised_Lower_Dims(X_LD, l_LD, channel_uses)
            Initial_Constellation = 'Optimized_LD'
    elif channel_uses == 8:
        if(Initial_Constellation[:4]) == 'APSK':
            Amount_of_Rings = int(Initial_Constellation[4::])
            [X_target,idx_train] = get_APSK_8D(M,Amount_of_Rings)
            Initial_Constellation = Initial_Constellation + '_' + str(Amount_of_Rings) + 'Rings'
        else:
            Optimized_Constellation_LD = np.load(Initial_Constellation, allow_pickle = True)
            X_LD = Optimized_Constellation_LD[0][:,:,-1]
            l_LD =  Optimized_Constellation_LD[2]
            [X_target,idx_train] = get_Optimised_Lower_Dims(X_LD, l_LD, channel_uses)
            Initial_Constellation = 'Optimized_LD'
    else:
        if(Initial_Constellation == 'SP-BPSK'):
            [X_target,idx_train] = get_BPSK_11bits_12D()
        else:
            Optimized_Constellation_LD = np.load(Initial_Constellation, allow_pickle = True)
            X_LD = Optimized_Constellation_LD[0][:,:,-1]
            l_LD =  Optimized_Constellation_LD[2]
            [X_target,idx_train] = get_Optimised_Lower_Dims(X_LD, l_LD, channel_uses)
            Initial_Constellation = 'Optimized_LD'
    X_target = tr.tensor(X_target,dtype = tr.float32).to(Device)
    X_target = normalization(X_target)
    return X_target, idx_train, Initial_Constellation

#A function for saving the variables during the optimisation for GMI optimisation
# Constellations: A matrix containing the constellations during each step of the optimisation
# EsNo_dB: The SNR for which the constellation was optimised
# idx_train: The labeling of the constellation
# learning rate: The learning rate during the optimisation
# Estimation_Type: The estimation used for calculating the loss function
# Initial Constellation: The name of the initial constellation used
# start_time: The time at which the optimisation started
def save_GMI(Constellations, EsNo_dB, idx_train, epochs, learning_rate, Initial_Constellation, start_time):
    M = np.size(Constellations,0)
    channel_uses = np.size(Constellations,1)
    np.save('./Data/GMI/' + str(channel_uses) + 'D/' + str(M) + '/GMI_' + str(channel_uses) + 'D_' + str(M) + '_' + str(EsNo_dB) + 'dB_'+ str(learning_rate)+'lr' + Initial_Constellation + str(round(start_time)), [
      Constellations,
      EsNo_dB,
      idx_train,
      epochs,
      learning_rate,
      time.time()-start_time], allow_pickle=True)


#A function for saving the variables during the optimisation for MI optimisation
# Constellations: A matrix containing the constellations during each step of the optimisation
# EsNo_dB: The SNR for which the constellation was optimised
# learning rate: The learning rate during the optimisation
# Estimation_Type: The estimation used for calculating the loss function
# Initial Constellation: The name of the initial constellation used
# start_time: The time at which the optimisation started
def save_MI(Constellations, EsNo_dB, epochs, learning_rate, start_time):
    M = np.size(Constellations,0)
    channel_uses = np.size(Constellations,1)
    np.save('./Data/MI/' + str(channel_uses) + 'D/' + str(M) + '/MI_' + str(channel_uses) + 'D_' + str(M) + '_' + str(EsNo_dB) + 'dB_'+ str(learning_rate)+'lr' + str(round(start_time)), [
      Constellations,
      EsNo_dB,
      epochs,
      learning_rate,
      time.time()-start_time], allow_pickle=True)
   
#A function for rotating the entire set of quadratures randomly
# Quad: The set of quadratures with their associated weights
# M: The amount of constellation points of the constellations, for GMI set it to the actual value, for MI set it to 2
# Output: The rotated set of quadratures
def Quad_Rotation_Random(Quad,M, Device):
    M = int(M)
    m = int(np.log2(M))
    D_complex = int((Quad.size(1)-1)/2)
    W = tr.zeros(Quad.size(0),Quad.size(1),m).to(Device)
    angles = tr.pi/2*tr.rand(m,D_complex)
    for i in range(m):
        for j in range(D_complex):
            W[:,2*j,i] = Quad[:,2*j] * tr.cos(angles[i,j]) - Quad[:,2*j+1] * tr.sin(angles[i,j])
            W[:,2*j+1,i] = Quad[:,2*j+1] * tr.cos(angles[i,j]) + Quad[:,2*j] * tr.sin(angles[i,j])
        W[:,-1,i] = Quad[:,-1]
    return W

#A function for estimating the AIR usin a Monte-Carlo approximation for a given constellation using cupy
# X: The constellation
# idx: the labeling of the constellation in decimal
# EsN0_dB: The signal-to-noise ratio at which the AIR should be evaluated
# samples: The amount of samples per constellation point
# Output: The MI and GMI of the constellation
def AIR_MC_Cupy(X,idx,EsN0_dB,samples):
    #Setting up variables
    M = int(np.size(X,0)) #The amount of constellation points
    m = int(np.log2(M)) # The amount of bits per symbol
    channel_uses = np.size(X,1) # The amount of dimension of the constellation
    idx2 = np.zeros(M, dtype = int) # Reordering the constellation according to the labeling
    for i in range(M):
        idx2[idx[i]] = i 
    X = X[idx2,:]
    X = normalization_cp(X) #Normalising the constellation
    
    #Determining the labeling
    labeling =de2bi(np.arange(M),m)
    Ik1 = np.zeros([int(M/2),int(m)],dtype = int) #Find the pointers to the subconstellations
    Ik0 = np.zeros([int(M/2),int(m)],dtype = int)
    for kk in range(int(m)): 
        Ik1[:,kk] = np.where(labeling[:,kk] == 1)[0]
        Ik0[:,kk] = np.where(labeling[:,kk] == 0)[0]
        
    #Calculating distances between points
    Dmat = cp.zeros((M,M,channel_uses))
    for i in range(channel_uses):
        Dmat[:,:,i] = cp.expand_dims(X[:,i],1) - cp.expand_dims(X[:,i],1).T #Calculate the distances between constellation points
    
    Dmatnorm = cp.sum(Dmat**2,2)
    
    #Calculating SNR variables
    Es = cp.sum(X**2,1).mean() #Calculate the signal energy
    EsN0lin = 10**(EsN0_dB/10)  #Turn the SNR value from dB to a linear value
    SigmaZ2 = (Es/(EsN0lin)) #Calculate the noise 
    
    #Generate the random noise
    z = cp.random.normal(0,np.sqrt(SigmaZ2/channel_uses),(channel_uses,samples))
    z[1:2:-1,:] = z[1:2:-1,:]*-1
    
    #Setting up variables
    sum0 = 0
    sum1 = 0
    sum_temp = 0
    #Calculating MI and GMI
    for i in range(samples):
        num = cp.exp((-(Dmatnorm+2*cp.sum(z[:,i]*Dmat,2)))/(2*SigmaZ2/channel_uses))
        for k in range(m):
            num0 = num[Ik0[:,k],:]
            num1 = num[Ik1[:,k],:]
            den0 = num0[:,Ik0[:,k]]
            den1 = num1[:,Ik1[:,k]]
            sum0 = sum0+cp.sum(cp.log2(cp.sum(num0,1)/cp.sum(den0,1)))
            sum1 = sum1+cp.sum(cp.log2(cp.sum(num1,1)/cp.sum(den1,1)))
        sum_temp = sum_temp + cp.sum(cp.log2(cp.sum(num,1)))
    GMI = m-(sum0+sum1)/(M*samples)
    MI = m - sum_temp/(M*samples)
    return GMI,MI

# A function for calculating the MI of a given constellation using a Monte-Carlo Approximation using cupy
# X: The constellation
# EsN0_dB: The signal-to-noise-ratio at which to evaluate the constellation
# samples: The amount of samples per constellation point
# Output: The MI of the constellation
def MI_MC_Cupy(X,EsN0_dB,samples):
    #Setting up variables
    M = int(np.size(X,0)) #The amount of constellation points
    m = int(np.log2(M)) # The amount of bits per symbol
    channel_uses = np.size(X,1) # The amount of dimension of the constellation
    X = normalization_cp(X) # Normalising the constellation power

        
    #Calculating distances between points
    Dmat = cp.zeros((M,M,channel_uses))
    for i in range(channel_uses):
        Dmat[:,:,i] = cp.expand_dims(X[:,i],1) - cp.expand_dims(X[:,i],1).T #Calculate the distances between constellation points
    
    Dmatnorm = cp.sum(Dmat**2,2)
    
    #Calculating SNR variables
    Es = cp.sum(X**2,1).mean() #Calculate the signal energy
    EsN0lin = 10**(EsN0_dB/10)  #Turn the SNR value from dB to a linear value
    SigmaZ2 = (Es/(EsN0lin)) #Calculate the noise 
    
    # Generating the random noise
    z = cp.random.normal(0,np.sqrt(SigmaZ2/channel_uses),(channel_uses,samples))
    z[1:2:-1,:] = z[1:2:-1,:]*-1
    
    #Setting up the variables
    sum_temp = 0
    #Calculating MI
    for i in range(samples):
        num = cp.exp((-(Dmatnorm+2*cp.sum(z[:,i]*Dmat,2)))/(2*SigmaZ2/channel_uses))
        sum_temp = sum_temp + cp.sum(cp.log2(cp.sum(num,1)))
    MI = m - sum_temp/(M*samples)
    return MI

#A function for calculating the loss function for use in optimising GMI based constellations using the RQ method
# X_tilde: An eye matrix representing the one-hot vectors used as the input for the neural network
# idx: The labeling of the constellation in decimal
# EsNo_dB: The signal to noise ratio at which the optimisation whould be done
# encoder: The neural network object used in the optimisation
# Quad_base: The base set of quadratures used for the estimation
# Device: The device on which the tensors are loaded
# bpg: The amount of bits which are calculated per gradient, done to split up the memory to prevent memory overload. Should divide the amount of bits per symbol
def GMI_RQ(X_tilde,idx, EsNo_dB, encoder, Quad_base, Device, bpg):
    X = encoder(X_tilde) #Get X from the encoder
    channel_uses = X.size(1) #The dimensionality of the constellation
    M = X.size(0) #The cardinality of the constellation
    m = np.log2(M) #The amount of bits per symbol
    Quad = Quad_Rotation_Random(Quad_base, M, Device) #Randomly rotating the set of quadratures
    for i in range(int(channel_uses)): #Changing the sign of each odd dimension of the quadrature, used to simplify the complex inner product calculation
        if i%2 == 0:
            Quad[:,i,:] = Quad[:,i,:]*-1
    X = normalization(X) #Normalising the constellation
    idx2 = np.zeros(M) #Reordering the constellation according to the labeling
    for i in range(M):
        idx2[idx[i]] = i 
    X = X[idx2,:]
    Dmat = tr.zeros(M,M,channel_uses,requires_grad=True).to(Device)
    for i in range(channel_uses):
        Dmat[:,:,i] = X[:,i].unsqueeze(1) -(X[:,i].unsqueeze(1)).t() #Calculate the distances between constellation points
    
    labeling = de2bi(np.arange(M), m)
    
    Ik1 = np.zeros([int(M/2),int(m),1]) #Find the pointers to the subconstellations
    Ik0 = np.zeros([int(M/2),int(m),1])
    Ikden1 = np.zeros([int(M/2),int(M/2),int(m)])
    Ikden0 = np.zeros([int(M/2),int(M/2),int(m)])
    for kk in range(int(m)): 
        Ik1[:,kk,0] = np.where(labeling[:,kk] == 1)[0]
        Ik0[:,kk,0] = np.where(labeling[:,kk] == 0)[0]
        Ikden1[:,:,kk] = Ik1[:,kk,:] + M*Ik1[:,kk,:].T + kk*M**2
        Ikden0[:,:,kk] = Ik0[:,kk,:] + M*Ik0[:,kk,:].T + kk*M**2
        Ik1[:,kk,0] = Ik1[:,kk,0]  + kk*M
        Ik0[:,kk,0] = Ik0[:,kk,0]  + kk*M
    Ik1 = tr.tensor(np.squeeze(Ik1), dtype = tr.int64).to(Device)
    Ik0 = tr.tensor(np.squeeze(Ik0), dtype = tr.int64).to(Device)
    Ikden1 = tr.tensor(Ikden1, dtype = tr.int64).to(Device)
    Ikden0 = tr.tensor(Ikden0, dtype = tr.int64).to(Device)
    
    Es = tr.sum(X**2,1).mean() #Calculate the signal energy
    EsN0lin = 10**(EsNo_dB/10)  #Turn the SNR value from dB to a linear value
    SigmaZ2 = (Es/(EsN0lin)) #Calculate the noise 
    
    Dmatnorm = tr.sum(Dmat**2,2).unsqueeze(2)
    Dmat = Dmat.unsqueeze(3).repeat([1, 1, 1,bpg])
    loops = int(m/bpg)
    for l1 in range(Quad.size(0)): #Dimension 1
        for k in range(loops):
            num = tr.exp(-(Dmatnorm + 2/np.sqrt(channel_uses/2)*tr.sqrt(SigmaZ2)*(tr.sum(Quad[l1,0:channel_uses,k*bpg:(k+1)*bpg]*Dmat,2)))/(2*SigmaZ2/channel_uses)).T
            a = num.sum(1)
            num0 = a.take(Ik0[:,k*bpg:(k+1)*bpg])
            den0 = num.take(Ikden0[:,:,k*bpg:(k+1)*bpg]).sum(1)
            den1 = num.take(Ikden1[:,:,k*bpg:(k+1)*bpg]).sum(1)
            num1 = a.take(Ik1[:,k*bpg:(k+1)*bpg])
            sum_0 = tr.sum(tr.abs(Quad[l1,channel_uses,k*bpg:(k+1)*bpg])*tr.log(num0/den0)/tr.log(tr.tensor(2, dtype = tr.float32))) 
            sum_1 = tr.sum(tr.abs(Quad[l1,channel_uses,k*bpg:(k+1)*bpg])*tr.log(num1/den1)/tr.log(tr.tensor(2, dtype = tr.float32)))
            loss = sum_0 + sum_1
            loss.backward(retain_graph=True)

#A function for calculating the loss function for use in optimising MI based constellations using the RQ method
# X_tilde: An eye matrix representing the one-hot vectors used as the input for the neural network
# EsNo_dB: The signal to noise ratio at which the optimisation whould be done
# encoder: The neural network object used in the optimisation
# Quad_base: The base set of quadratures used for the estimation
# Device: The device on which the tensors are loaded
def MI_RQ(X_tilde, EsNo_dB, encoder, Quad_base, Device):
    X = encoder(X_tilde) #Get X from the encoder
    channel_uses = X.size(1) #The dimensionality of the constellation
    M = X.size(0) #The cardinality of the constellation
    Quad = Quad_Rotation_Random(Quad_base, 2, Device) #Randomly rotating the set of quadratures
    for i in range(int(channel_uses)): #Changing the sign of each odd dimension of the quadrature, used to simplify the complex inner product calculation
        if i%2 == 0:
            Quad[:,i] = Quad[:,i]*-1
    X = normalization(X)
    Dmat = tr.zeros(M,M,channel_uses,requires_grad=True).to(Device)
    for i in range(channel_uses):
        Dmat[:,:,i] = X[:,i].unsqueeze(1) -(X[:,i].unsqueeze(1)).t() #Calculate the distances between constellation points
    
    Es = tr.sum(X**2,1).mean() #Calculate the signal energy
    EsN0lin = 10**(EsNo_dB/10)  #Turn the SNR value from dB to a linear value
    SigmaZ2 = (Es/(EsN0lin)) #Calculate the noise 
    
    Dmatnorm = tr.sum(Dmat**2,2).unsqueeze(2)
    Dmat = Dmat.unsqueeze(3)
    for l1 in range(Quad.size(0)): 
        num = tr.exp(-(Dmatnorm + 2/np.sqrt(channel_uses/2)*tr.sqrt(SigmaZ2)*(tr.sum(Quad[l1,0:channel_uses]*Dmat,2)))/(2*SigmaZ2/channel_uses)).T
        loss = tr.sum(tr.abs(Quad[l1,channel_uses])*tr.log(num.sum(1)))/tr.log(tr.tensor(2, dtype = tr.float32)) 
        loss.backward(retain_graph=True)

#A function for calculating the GMI using GH
# X_tilde: The constellation
# idx: The decimal labeling of the constellation
# EsNo_dB: The signal-to-noise ratio
# Output: GMI
def GMI_GH_Numpy(X_tilde,idx, EsNo_dB):
    X = normalization_np(X_tilde)
    M = int(np.size(X,0))
    m = int(np.log2(M))
    channel_uses = np.size(X,1)
    idx2 = np.zeros(M, dtype = int)
    labeling = de2bi(np.arange(M), m)
    for i in range(M):
        idx2[idx[i]] = i 
    X = X[idx2,:]
    Dmat = np.zeros((M,M,channel_uses))
    for i in range(channel_uses):
        Dmat[:,:,i] = np.expand_dims(X[:,i],1) - np.expand_dims(X[:,i],1).T #Calculate the distances between constellation points
    
    Ik1 = np.zeros([int(M/2),1,int(m)],dtype = int) #Find the pointers to the subconstellations
    Ik0 = np.zeros([int(M/2),1,int(m)],dtype = int)
    for kk in range(int(m)): 
        Ik1[:,0,kk] = np.where(labeling[:,kk] == 1)[0]
        Ik0[:,0,kk] = np.where(labeling[:,kk] == 0)[0]
    Ikden1 = Ik1 + M*np.transpose(Ik1,(1,0,2))
    Ikden0 = Ik0 + M*np.transpose(Ik0,(1,0,2))
    
    GH_xi = np.array(GH['xi'])#Load in the Gauss-Hermite points
    GH_alpha = np.array(GH['alpha'])#Load in the Gauss-Hermite weigths
    
    xi = np.zeros((10**channel_uses,channel_uses))
    alpha = np.ones(10**channel_uses)
    for i in range(channel_uses):
        for j in range(10**i):
            xi[j*10**(channel_uses-i):(j+1)*10**(channel_uses-i),i] = np.repeat(GH_xi,10**(channel_uses-i-1))
            alpha[j*10**(channel_uses-i):(j+1)*10**(channel_uses-i)] = alpha[j*10**(channel_uses-i):(j+1)*10**(channel_uses-i)]*np.repeat(GH_alpha,10**(channel_uses-i-1))
    Es = np.sum(X**2,1).mean() #Calculate the signal energy
    EsN0lin = 10**(EsNo_dB/10)  #Turn the SNR value from dB to a linear value
    SigmaZ2 = (Es/(EsN0lin)) #Calculate the noise 
    
    sum_0 = np.zeros((int(M/2),m))
    sum_1 = np.zeros((int(M/2),m))
    l_2 = np.log(2)
    GMI = 0
    Dmatnorm = np.sum(Dmat**2,2)
    for l1 in range(10**channel_uses): #Dimension 1
        num = np.exp(-(Dmatnorm + 2*np.sqrt(SigmaZ2)/np.sqrt(channel_uses/2)*(np.sum(xi[l1]*Dmat,2)))/(2*SigmaZ2/channel_uses))
        a = num.sum(1)
        num0 = np.squeeze(a.take(Ik0))
        den0 = np.squeeze(num.take(Ikden0).sum(1))
        den1 = np.squeeze(num.take(Ikden1).sum(1))
        num1 = np.squeeze(a.take(Ik1))
        sum_0 = alpha[l1]*np.log(num0/den0)/l_2 +sum_0
        sum_1 = alpha[l1]*np.log(num1/den1)/l_2 +sum_1
    sum_0 = np.sum(sum_0)
    sum_1 = np.sum(sum_1)
    GMI = m-1/M/(np.pi**(channel_uses/2))*(sum_0 + sum_1)
    return GMI

#A function for calculating the loss function for MI optimisation using GH for 2D Constellations
# X_tilde: An eye matrix represnting the one hot vector input for the neural network
# EsNo_dB: The signal-to-noise-ratio at which to optimise the constellation
# encoder: The neural network object used in the optimisation
# Device: The device on which the tensors are stored
def MI_GH(X_tilde, EsNo_dB, encoder, Device):
    X = normalization(encoder(X_tilde))
    M = X.size(0)
    m = tr.log2(M)
    channel_uses = X.size(1)
    GH_xi = tr.tensor(GH['xi'], dtype  = tr.float32).to(Device)#Load in the Gauss-Hermite points
    GH_alpha = tr.tensor(GH['alpha'], dtype = tr.float32).to(Device)#Load in the Gauss-Hermite weigths
    
    Dmat = tr.zeros(M,M,channel_uses).to(Device)
    Dmat[:,:,0] = X[:,0].unsqueeze(1) -(X[:,0].unsqueeze(1)).t() #Calculate the distances between constellation points
    Dmat[:,:,1] = X[:,1].unsqueeze(1) -(X[:,1].unsqueeze(1)).t()

    Es = (X[:,0]**2 + X[:,1]**2).mean() #Calculate the signal energy
    EsN0lin = 10**(EsNo_dB/10)  #Turn the SNR value from dB to a linear value
    SigmaZ2 = (Es/(EsN0lin)) #Calculate the noise variance
    sum_0 = 0 #Initialize the sum 

    for l1 in range(10): #Dimension 1
        for l2 in range(10): #Dimension 2
             num = tr.exp(-((Dmat[:,:,0]**2 + Dmat[:,:,1]**2) + 2*tr.sqrt(SigmaZ2)*(GH_xi[l1]*Dmat[:,:,0] - GH_xi[l2]*Dmat[:,:,1]))/SigmaZ2)
             sum_0 = GH_alpha[l1]*GH_alpha[l2]*tr.log(tr.sum(num,1))/tr.log(tr.tensor(2, dtype = tr.float32))  + sum_0
    sum_0 = tr.sum(sum_0)
    MI = m-1/M/np.pi*sum_0
    loss = -MI
    loss.backward()

#A function for calculating the loss function for MI optimisation using GH for 4D Constellations
# X_tilde: An eye matrix represnting the one hot vector input for the neural network
# EsNo_dB: The signal-to-noise-ratio at which to optimise the constellation
# encoder: The neural network object used in the optimisation
# Device: The device on which the tensors are stored
def MI_GH_4D(X_tilde, EsNo_dB, encoder, Device):
    X = normalization(encoder(X_tilde)) 
    M = X.size(0)
    m = tr.log2(M)
    channel_uses = X.size(1)
    GH_xi = tr.tensor(GH['xi'], dtype  = tr.float32).to(Device)#Load in the Gauss-Hermite points
    GH_alpha = tr.tensor(GH['alpha'], dtype = tr.float32).to(Device)#Load in the Gauss-Hermite weigths
    
    Dmat = tr.zeros(M,M,channel_uses).to(Device)
    Dmat[:,:,0] = X[:,0].unsqueeze(1) -(X[:,0].unsqueeze(1)).t() #Calculate the distances between constellation points
    Dmat[:,:,1] = X[:,1].unsqueeze(1) -(X[:,1].unsqueeze(1)).t()
    Dmat[:,:,2] = X[:,2].unsqueeze(1) -(X[:,2].unsqueeze(1)).t()
    Dmat[:,:,3] = X[:,3].unsqueeze(1) -(X[:,3].unsqueeze(1)).t()
    Dmatnorm = Dmat[:,:,0]**2 + Dmat[:,:,1]**2+ Dmat[:,:,2]**2 + Dmat[:,:,3]**2
    Es = (X[:,0]**2 + X[:,1]**2 + X[:,2]**2 + X[:,3]**2).mean() #Calculate the signal energy
    EsN0lin = 10**(EsNo_dB/10)  #Turn the SNR value from dB to a linear value
    SigmaZ2 = (Es/(EsN0lin)) #Calculate the noise variance
    sum_0 = 0 #Initialize the sum 
    for l1 in range(10): #Dimension 1
        for l2 in range(10): #Dimension 2
            for l3 in range(10):
                for l4 in range(10):
                     num = tr.exp(-(Dmatnorm + np.sqrt(2)*tr.sqrt(SigmaZ2)*(GH_xi[l1]*Dmat[:,:,0] - GH_xi[l2]*Dmat[:,:,1] + GH_xi[l3]*Dmat[:,:,2] - GH_xi[l4]*Dmat[:,:,3]))/(0.5*SigmaZ2))
                     sum_0 = GH_alpha[l1]*GH_alpha[l2]*GH_alpha[l3]*GH_alpha[l4]*tr.log(tr.sum(num,1))/tr.log(tr.tensor(2, dtype = tr.float32))  + sum_0
    sum_0 = tr.sum(sum_0)
    MI = m-1/M/(np.pi**2)*sum_0
    loss = -MI
    return loss

#A function for plotting a constellation with labeling
# X: The constellation to be plotted
# idx: The binary labeling for the constellation
def plot_constellation_labeling(X, idx):
    M = np.size(X,0)
    m = int(np.log2(M))
    channel_uses = np.size(X,1)
    idx2 = np.zeros(M, dtype = int)
    labeling = de2bi(np.arange(M), m)
    for i in range(M):
        idx2[idx[i]] = i 
    X = X[idx2,:]
    max_axis =np.round(np.max(np.abs(X))*110)/100
    fig = plt.figure(figsize=(m*3.1,channel_uses*2.9))
    for j in range(int(channel_uses/2)):
        x = X[:, 2*j+0]
        y = X[:, 2*j+1] 
        for i in range(int(m)):
            ax1 = fig.add_subplot(int(channel_uses/2), m, j*m+i+1)
            ax1.scatter(x[labeling[:,i] == 0], y[labeling[:,i] == 0], c='tab:blue', marker='.')
            ax1.scatter(x[labeling[:,i] == 1], y[labeling[:,i] == 1], c='tab:red', marker='.')
            ax1.spines['left'].set_position('center')
            ax1.spines['bottom'].set_position('center')
            ax1.spines['right'].set_color('none')
            ax1.spines['top'].set_color('none')
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(-max_axis, max_axis)
            plt.ylim(-max_axis, max_axis)
            fig.subplots_adjust(hspace=0, wspace=0.1)
    plt.show()

#A function for plotting a constellation without labeling
# X: The constellation to be plotted
# idx: The binary labeling for the constellation
def plot_constellation(X):
    channel_uses = np.size(X,1)
    max_axis =np.round(np.max(np.abs(X))*110)/100
    fig = plt.figure()
    for j in range(int(channel_uses/2)):
        x = X[:, 2*j+0]
        y = X[:, 2*j+1] 
        ax1 = fig.add_subplot(1,int(channel_uses/2), j + 1)
        ax1.scatter(x, y, c='tab:blue', marker='.')
        ax1.spines['left'].set_position('center')
        ax1.spines['bottom'].set_position('center')
        ax1.spines['right'].set_color('none')
        ax1.spines['top'].set_color('none')
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-max_axis, max_axis)
        plt.ylim(-max_axis, max_axis)
        fig.subplots_adjust(hspace=0, wspace=0.1)
    plt.show()