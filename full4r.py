import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import MGNNEMVreverse
import bnn2
np.random.seed(531)

traindata = np.load('traindata.npy', allow_pickle = True)
testdata = np.load('testdata.npy', allow_pickle = True)



x_train = torch.load('r4traincode')
x_train = x_train[:,0:32]
y_train = torch.load('r4trainY')

x_test = torch.load('r4testcode')
x_test = x_test[:,0:32]
y_test = torch.load('r4testY')

GCNNAFP = MGNNEMVreverse.MGNNE(traindata)
GCNNAFP.load_state_dict(torch.load('r4e'))

batch_size = len(x_train)
BNN = bnn2.BBP_Homoscedastic_Model_Wrapper(input_dim = 32, output_dim = 1, no_units = 128,batch_size = batch_size, no_batches = 1, learn_rate = 8e-3, init_log_noise = 0)

BNN.load_data(x_train,y_train)

#kernel_list = []
#mean_list = []
#for i in range(5):
#    BNN.train(num_epochs=400)
#    _,mean = BNN.test(x_test,y_test)
#    mean_list.append(mean[2])
#    kernel_list.append(BNN.distribution(x_test[2]))
#    
#    
#    
#
#mean_density = []
#for i in range(5):
#    mean_density.append(float(kernel_list[i](mean_list[i])))
#    plt.plot(np.linspace(0, 10, 200),kernel_list[i](np.linspace(0, 10, 200)), alpha = 0.75, label = str(400*(i+1)) + ' epochs')
#    
#plt.plot(y_test[2]*np.ones(100),np.linspace(0,15,100),'k--',label = 'actual value')
##plt.plot(mean_list,mean_density,'--',label = 'change of mean')
#plt.xlim(10, 0)
#plt.ylim(0,6)
#plt.xlabel('ppm')
#plt.ylabel('density')
#plt.legend(loc='upper left')
#plt.show()
    

BNN.train(num_epochs=4000)
plotset = []
for i in range(len(testdata)):
    if testdata[i,0] in [4310,97]:
        plotset.append(i)


for k in plotset:
    A = testdata[k,1]
    kernel_list = []
    vec1 = GCNNAFP.vectorize(testdata[k,1])
    num_h = np.zeros(len(A))
    for i in range(len(A)):
        num_h[i] = 4 - sum(A[:,i])
    
    for i in range(len(A)):
        kernel_list.append(BNN.distribution(vec1[i,0:32]))
    kernel = np.zeros(200)
    for i in range(len(A)):
        kernel += kernel_list[i](np.linspace(0, 10, 200))*num_h[i]
        plt.plot(testdata[k,2][i]*np.ones(100), np.linspace(0,15,100),'k--')
    plt.plot(np.linspace(0, 10, 200),kernel, c = 'k')
    plt.xlim(11, -1)
    plt.ylim(0,15)
    plt.xlabel('ppm')
    plt.ylabel('density')
    plt.title('SDBS: ' + str(testdata[k,0]))
    plt.show()
    
for k in plotset:
    A = testdata[k,1]
    kernel_list = []
    vec1 = GCNNAFP.vectorize(testdata[k,1])
    num_h = np.zeros(len(A))
    for i in range(len(A)):
        num_h[i] = 4 - sum(A[:,i])
    
    for i in range(len(A)):
        kernel_list.append(BNN.distribution(vec1[i,0:32]))
    kernel = np.zeros(200)
    for i in range(len(A)):
        plt.plot(np.linspace(0, 10, 200), kernel_list[i](np.linspace(0, 10, 200))*num_h[i], label = 'carbon '+ str(int(i+1)))
        plt.plot(testdata[k,2][i]*np.ones(100), np.linspace(0,15,100),'k--')
    
    plt.xlim(11, -1)
    plt.ylim(0,15)
    plt.xlabel('ppm')
    plt.ylabel('density')
    plt.title('SDBS: ' + str(testdata[k,0]))
    plt.legend(loc = 'upper left')
    plt.show()
    
mse, means = BNN.test(x_test[:,0:32],y_test)
    
plt.figure(figsize=(18,6))
plt.scatter(y_test, means - y_test, marker = '*', c = 'k')
plt.plot(np.linspace(-1,11,200),np.zeros(200), 'k--')
plt.xlim([11,-1])
plt.ylim([-3,3])
plt.xlabel('ppm')
plt.ylabel('Error in chemical shift')
plt.show()


plt.scatter(y_test,means, marker = '*', c = 'k')
plt.plot(np.linspace(-1,11,200),np.linspace(-1,11,200), 'k--')
plt.xlim([-1,11])
plt.ylim([-1,11])
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.show()
    