import torch
import numpy as np
import torch.utils.data
import MGNNEMVreverse
np.random.seed(531)

#data1 = np.load('263_4291.npy', allow_pickle = True)
#data2 = np.load('654_211.npy', allow_pickle = True)
#data3 = np.load('8893_9726.npy', allow_pickle = True)
#data4 = np.load('4292_5387.npy', allow_pickle = True)
#data5 = np.load('7940_10424.npy', allow_pickle = True)
#data6 = np.load('9437_226.npy', allow_pickle = True)
#data7 = np.load('97_2051.npy', allow_pickle = True)
#
#data = np.vstack((data1,data2,data3,data4,data5,data6,data7))
#data = np.random.permutation(data)
#
#traindata = data[0:int(len(data)*.8)]


traindata = np.load('traindata.npy', allow_pickle = True)
testdata = np.load('testdata.npy', allow_pickle = True)

model = MGNNEMVreverse.MGNNE(traindata)
model.load_state_dict(torch.load('r4e35'))

X = model.vectorize(torch.tensor(traindata[0,1]))
Y = torch.tensor(traindata[0,2]).reshape(-1,1)

for i in range(len(traindata) - 1):
    X = torch.cat([X,model.vectorize(torch.tensor(traindata[i+1,1]))])
    Y = torch.cat([Y,torch.tensor(traindata[i+1,2]).reshape(-1,1)])

Xcl = X[(Y > 0).squeeze(), :]
Ycl = Y[(Y > 0).squeeze(), :]

torch.save(Xcl,'r4e35traincode')
torch.save(Ycl, 'r4e35trainY')


Xtest = model.vectorize(torch.tensor(testdata[0,1]))
Ytest = torch.tensor(testdata[0,2]).reshape(-1,1)

for i in range(len(testdata) - 1):
    Xtest = torch.cat([Xtest,model.vectorize(torch.tensor(testdata[i+1,1]))])
    Ytest = torch.cat([Ytest,torch.tensor(testdata[i+1,2]).reshape(-1,1)])

Xtestcl = Xtest[(Ytest > 0).squeeze(), :]
Ytestcl = Ytest[(Ytest > 0).squeeze(), :]

torch.save(Xtestcl,'r4e35testcode')
torch.save(Ytestcl, 'r4e35testY')
total_loss = 0
for i in range(len(testdata)):
    indiv_loss = np.mean(model(torch.tensor(testdata[i,1])).cpu().data.numpy().reshape(1,-1) - testdata[i,2])**2
#            print(indiv_loss)
    total_loss +=  indiv_loss