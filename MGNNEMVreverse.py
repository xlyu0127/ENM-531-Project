import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(531)

class MGNNE(torch.nn.Module):
    
    def __init__(self, data,atom_feature = 8, bond_feature = 8,  small_layer = 32, large_layer = 128 , out_feature = 1):
        self.data = data
        super(MGNNE, self).__init__()
#        if torch.cuda.is_available() == True:
#            self.dtype = torch.cuda.FloatTensor
#        else:
        self.dtype = torch.FloatTensor
        
        self.V = torch.zeros([1,atom_feature]).type(self.dtype)
        self.V[0] = 1.
        
        self.atom_feature = atom_feature
        self.small_layer = small_layer
        self.large_layer = large_layer
        self.bond_feature = bond_feature
        self.initial_bond = torch.zeros([1,self.bond_feature]).type(self.dtype)
        self.out_feature = out_feature
        self.loss_func = torch.nn.MSELoss()
        self.loss_func.cuda()
        self.encoder_layer = int(3*self.bond_feature/2)
        self.depth = 4
        
        
        self.W11 = nn.Linear(self.atom_feature + self.bond_feature,self.small_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W11.weight, gain=1.0)
        self.W12 = nn.Linear(self.small_layer,self.small_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W12.weight, gain=1.0)
        self.W13 = nn.Linear(self.small_layer,self.bond_feature,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W13.weight, gain=1.0)
        
        self.W21 = nn.Linear(self.atom_feature + self.bond_feature,self.small_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W21.weight, gain=1.0)
        self.W22 = nn.Linear(self.small_layer,self.small_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W22.weight, gain=1.0)
        self.W23 = nn.Linear(self.small_layer,self.bond_feature,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W23.weight, gain=1.0)
        
        self.W31 = nn.Linear(self.atom_feature + self.bond_feature,self.small_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W21.weight, gain=1.0)
        self.W32 = nn.Linear(self.small_layer,self.small_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W22.weight, gain=1.0)
        self.W33 = nn.Linear(self.small_layer,self.bond_feature,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.W23.weight, gain=1.0)
        
        self.WN1 = nn.Linear(self.bond_feature*self.depth + self.atom_feature,self.large_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.WN1.weight, gain=1.0)
        self.WN2 = nn.Linear(self.large_layer,self.large_layer,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.WN2.weight, gain=1.0)
        self.WN3 = nn.Linear(self.large_layer,self.out_feature,bias=True).type(self.dtype)
        nn.init.xavier_uniform_(self.WN3.weight, gain=1.0)
        

        
        self.Tanh = nn.Tanh()
    
    # Information transfer functions for single, double, triple bonds
    def W1(self,V):
        V = self.Tanh(self.W11(V))
        V = self.Tanh(self.W12(V))
        return self.W13(V)
    
    def W2(self,V):
        V = self.Tanh(self.W21(V))
        V = self.Tanh(self.W22(V))
        return self.W23(V)
    
    def W3(self,V):
        V = self.Tanh(self.W31(V))
        V = self.Tanh(self.W32(V))
        return self.W33(V)
    # Predict chemical shift basing on chemical environment vector up to 3 layers
    # Bayesian neural network here.
    def WN(self,V):
        V = self.Tanh(self.WN1(V))
        V = self.Tanh(self.WN2(V))
        return self.WN3(V)

    
    def vectorize(self,A):
        
        W = [self.W1,self.W2,self.W3]
        #Vc represents the vectorized chemical environment. Vc[i,j,:] is the jth adjacent layer of ith carbon, up to 3 layers.
        Vc = torch.zeros(A.shape[0],self.bond_feature,self.depth).type(self.dtype)
        size = A.shape[0]
        y = torch.zeros([size, self.bond_feature*self.depth + self.atom_feature]).type(self.dtype)
        #unsaturated = (A.sum(0) - 4).nonzero()
        
        # Breadth First Processing of vicinity chemical environment information from bonding matrix
        # Mixed feature represents combination of information flown from last atom and the inherent information of the current atom
        # In this case since all the atoms are carbon, the atom feature is set to a fixed value.
        for n in range(size):
            
            for i in range(size):
                if A[n,i] != 0 and not (i in [n]):
                    mixed_first_feature = W[int(A[n,i])-1](torch.cat((self.initial_bond,self.V), dim = 1))
                    Vc[n,:,0] = Vc[n,:,0] + mixed_first_feature.reshape(1,-1)
                    
                    for j in range(A.shape[0]):
                        
                        if A[i,j] != 0 and not (j in [i,n]):
                            mixed_first_feature = W[int(A[i,j])-1](torch.cat((self.initial_bond,self.V), dim = 1))
                            mixed_second_feature = W[int(A[i,n])-1](torch.cat((mixed_first_feature,self.V), dim = 1))
                            Vc[n,:,1] = Vc[n,:,1] + mixed_second_feature.reshape(1,-1)
                            
                            for k in range(A.shape[0]):
                                
                                if A[j,k] != 0 and not (k in [i,j,n]):
                                    mixed_first_feature = W[int(A[j,k])-1](torch.cat((self.initial_bond,self.V), dim = 1))
                                    mixed_second_feature = W[int(A[i,j])-1](torch.cat((mixed_first_feature,self.V), dim = 1))
                                    mixed_third_feature = W[int(A[i,n])-1](torch.cat((mixed_second_feature,self.V), dim = 1))

                                    Vc[n,:,2] = Vc[n,:,2] + mixed_third_feature.reshape(1,-1)
                                    
                                    for l in range(A.shape[0]):
                                        if A[k,l] != 0 and not (l in [i,j,k,n]):
                                            mixed_first_feature = W[int(A[k,l])-1](torch.cat((self.initial_bond,self.V), dim = 1))
                                            mixed_second_feature = W[int(A[j,k])-1](torch.cat((mixed_first_feature,self.V), dim = 1))
                                            mixed_third_feature = W[int(A[i,j])-1](torch.cat((mixed_second_feature,self.V), dim = 1))
                                            mixed_fourth_feature = W[int(A[i,n])-1](torch.cat((mixed_third_feature,self.V), dim = 1))
                                            Vc[n,:,3] = Vc[n,:,3] + mixed_fourth_feature.reshape(1,-1)
                                            
#                                            for m in range(A.shape[0]):
#                                                if A[l,m] != 0 and not (m in [i,j,k,n,l]):
#                                                    mixed_first_feature = W[int(A[m,l])-1](torch.cat((self.initial_bond,self.V), dim = 1))
#                                                    mixed_second_feature = W[int(A[l,k])-1](torch.cat((mixed_first_feature,self.V), dim = 1))
#                                                    mixed_third_feature = W[int(A[k,j])-1](torch.cat((mixed_second_feature,self.V), dim = 1))
#                                                    mixed_fourth_feature = W[int(A[i,j])-1](torch.cat((mixed_third_feature,self.V), dim = 1))
#                                                    mixed_fifth_feature = W[int(A[i,n])-1](torch.cat((mixed_fourth_feature,self.V), dim = 1))
#                                                    Vc[n,:,4] = Vc[n,:,4] + mixed_fifth_feature.reshape(1,-1)
                                    
                                    
            y[n] = torch.cat((Vc[n,:,:].reshape(1,-1),self.V), dim = 1)
            

        return y

    def forward(self,A):
        return self.WN(self.vectorize(A))

    def encoder_loss(self,A):
        mu = A.mean(0).type(self.dtype)
        AN = A - mu
        return -self.loss_func(AN, torch.zeros(AN.shape).type(self.dtype))
    def encoder_train(self, num_iter, lr = 1e-3, verbose = True):
        data = self.data
        size = data.shape[0]
        start_time = timeit.default_timer()
    
        optimizer = torch.optim.Adam([self.W11.weight,
                                      self.W12.weight,
                                      self.W13.weight,
                                      self.W21.weight,
                                      self.W22.weight,
                                      self.W23.weight,
                                      self.W31.weight,
                                      self.W32.weight,
                                      self.W33.weight
                                        ], lr = lr)
        loss_stor = []
        
        
        for i in tqdm(range(num_iter)):
            X = self.vectorize(torch.tensor(data[(size*i)%size,1]))
            
            for j in range(size - 1):
                X = torch.cat([X,self.vectorize(torch.tensor(data[j,1]))])

#            reconstruction = self.decoder(self.encoder(X))
            loss = self.encoder_loss(X)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            elapsed = timeit.default_timer() - start_time
            if verbose:
                print('Time: %.4f, It: %5d, Loss: %.3f' % (elapsed,i, loss.cpu().data.numpy()))
            loss_stor.append(loss.cpu().data.numpy())

        return loss_stor
    
    def train(self, batch_size, num_iter, lr = 5e-4, verbose = True):
        data = self.data
        size = data.shape[0]
        start_time = timeit.default_timer()
    
        optimizer = torch.optim.Adam([self.WN1.weight,
                                      self.WN2.weight,
                                      self.WN3.weight], lr = lr)
        criterion = nn.MSELoss()
        loss_stor = []
        
        batch_size = batch_size
        
        for i in tqdm(range(num_iter)):
            X = self(torch.tensor(data[(batch_size*i)%size,1]))
            Y = torch.tensor(data[(batch_size*i)%size,2]).reshape(-1,1).type(self.dtype)
            
            for j in range(batch_size-1):
                X = torch.cat([X,self(torch.tensor(data[((batch_size*i)%size + j + 1)%size,1]))])
                Y = torch.cat([Y, torch.tensor(data[((batch_size*i)%size + j + 1)%size,2]).reshape(-1,1).type(self.dtype)])
                
            loss = criterion(X,Y)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
    
            elapsed = timeit.default_timer() - start_time
            if verbose:
                print('Time: %.4f, It: %5d, Loss: %.3f' % (elapsed,i, loss.cpu().data.numpy()))
            loss_stor.append(loss.cpu().data.numpy())
            # The dataset is permuted every epoch
            if (size - (i*batch_size)%size) < batch_size:
                data = np.random.permutation(data)
        if verbose:
            plt.hist(loss_stor[200:], bins = 100)
            plt.show()
            plt.plot(np.arange(len(loss_stor)),loss_stor)
        return loss_stor
#    
#    
if __name__ == "__main__":
    

    traindata = np.load('traindata.npy',allow_pickle = True)
    testdata = np.load('testdata.npy',allow_pickle = True)
    model = MGNNE(traindata)
    model.encoder_train(300, verbose = False)
    
   
#    Validate on test data every 20 batch    
    validation = []
    loss_stor = []
    for j in tqdm(range(20)):
        loss_stor.append(model.train(16,20, verbose = False))
        total_loss = 0
        for i in np.arange(len(testdata)):
            indiv_loss = np.mean(model(torch.tensor(testdata[i,1])).cpu().data.numpy().reshape(1,-1) - testdata[i,2])**2
#            print(indiv_loss)
            total_loss +=  indiv_loss
        validation.append(total_loss)
        
    plt.plot(validation)
    plt.show()
    
#    torch.save(model.state_dict(), 'r4')
