
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer
from scipy import stats

torch.cuda.device(0)
torch.cuda.get_device_name(torch.cuda.current_device())

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out

def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)
    
    return - (log_coeff + exponent).sum()


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)
    
    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()
    
    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
        return (exponent + log_coeff).sum()

class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        
        scale = (2/self.input_dim)**0.5
        rho_init = np.log(np.exp((2/self.input_dim)**0.5) - 1)
        self.weight_mus = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.05, 0.05))
        self.weight_rhos = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-2, -1))
        
        self.bias_mus = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.05, 0.05))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-2, -1))
        
    def forward(self, x, sample = True):
        
        if sample:
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            bias_epsilons =  Variable(self.bias_mus.data.new(self.bias_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            bias_stds = torch.log(1 + torch.exp(self.bias_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            bias_sample = self.bias_mus + bias_epsilons*bias_stds
            
            output = torch.mm(x, weight_sample) + bias_sample
            
            # computing the KL loss term
            prior_cov, varpost_cov = self.prior.sigma**2, weight_stds**2
            KL_loss = 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*weight_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.weight_mus - self.prior.mu)**2/prior_cov).sum()
            
            prior_cov, varpost_cov = self.prior.sigma**2, bias_stds**2
            KL_loss = KL_loss + 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*bias_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.bias_mus - self.prior.mu)**2/prior_cov).sum()
            
            return output, KL_loss
        
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, KL_loss
        
    def sample_layer(self, no_samples):
        all_samples = []
        for i in range(no_samples):
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            
            all_samples += weight_sample.view(-1).cpu().data.numpy().tolist()
            
        return all_samples

class BBP_Homoscedastic_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_log_noise, sigma = 1):
        super(BBP_Homoscedastic_Model, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # network with two hidden and one output layer
        self.layer1 = BayesLinear_Normalq(input_dim, no_units, gaussian(0, sigma))
        self.layer2 = BayesLinear_Normalq(no_units, output_dim, gaussian(0, sigma))
        
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace = True)
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

    
    def forward(self, x):
        
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)
        
        x, KL_loss = self.layer1(x)
        KL_loss_total = KL_loss_total + KL_loss
        x = self.activation(x)
        
        x, KL_loss = self.layer2(x)
        KL_loss_total = KL_loss_total + KL_loss
        
        return x, KL_loss_total

class BBP_Homoscedastic_Model_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, init_log_noise):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches
        
        self.network = BBP_Homoscedastic_Model(input_dim = input_dim, output_dim = output_dim,
                                               no_units = no_units, init_log_noise = init_log_noise)
        self.network.cuda()
        
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr = self.learn_rate) 
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learn_rate)
        self.loss_func = log_gaussian_loss
    
    def fit(self, x, y, no_samples):
        x, y = to_variable(var=(x, y), cuda=True)
        
        # reset gradient and total loss
        self.optimizer.zero_grad()
        fit_loss_total = 0
        
        for i in range(no_samples):
            output, KL_loss_total = self.network(x)

            # calculate fit loss based on mean and standard deviation of output
            fit_loss_total = fit_loss_total + self.loss_func(output, y, self.network.log_noise.exp(), self.network.output_dim)
        
        KL_loss_total = KL_loss_total/self.no_batches
        total_loss = (fit_loss_total + KL_loss_total)/(no_samples*x.shape[0])
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        return fit_loss_total/no_samples, KL_loss_total
    
    def load_data(self,x_train,y_train):
        
        # standardize the data
        self.x_mean, self.x_std = x_train.mean(axis=0), x_train.std(axis=0)
        self.y_mean, self.y_std = y_train.mean(axis=0), y_train.std(axis=0)
        self.x_train = (x_train - self.x_mean)/self.x_std
        self.y_train = (y_train - self.y_mean)/self.y_std
        
        print('Train data loaded and normalized.')
        
    def train(self,num_epochs = 2000):
        fit_loss_train = np.zeros(num_epochs)
        KL_loss_train = np.zeros(num_epochs)
        total_loss = np.zeros(num_epochs)
        
        self.best_net, best_loss = None, float('inf')
        for i in range(num_epochs):
    
            fit_loss, KL_loss = self.fit(self.x_train, self.y_train, no_samples = 10)
            fit_loss_train[i] += fit_loss.cpu().data.numpy()
            KL_loss_train[i] += KL_loss.cpu().data.numpy()
            
            total_loss[i] = fit_loss_train[i] + KL_loss_train[i]

            # choose the network 'fit' best the data
            if fit_loss < best_loss:
                best_loss = fit_loss
                self.best_net = copy.deepcopy(self.network)
                
            if i % 100 == 0 or i == num_epochs - 1:
                
                print("Epoch: %5d/%5d, Fit loss = %8.3f, KL loss = %8.3f, noise = %6.3f" %
              (i + 1, num_epochs, fit_loss_train[i], KL_loss_train[i], self.network.log_noise.exp().cpu().data.numpy()))
        return self.best_net.cuda()
    
    def test(self,x_test,y_test,sample_num = 500):
        x_test = (x_test - self.x_mean)/self.x_std
        y_test = (y_test - self.y_mean)/self.y_std
        samples = []
        for i in range(500):
          preds = (self.best_net.forward(x_test.cuda())[0]*self.y_std.cuda()) + self.y_mean.cuda()
        samples.append(preds.cpu().data.numpy()[:, 0])
        samples = np.array(samples)
        means = samples.mean(axis = 0).reshape(len(y_test),1)
        MSEloss_test = torch.mean((torch.tensor(means) - y_test)**2)
        
        return MSEloss_test, torch.tensor(means)

    def sample(self,X,sample_num = 500):
        X = (X - self.x_mean)/self.x_std
        samples = []

        # record the samples
        for i in range(sample_num):
          preds = (self.best_net.forward(X.cuda())[0]*self.y_std.cuda()) + self.y_mean.cuda()
          samples.append(preds.cpu().data.numpy()[:, 0])
        samples = np.array(samples)
        means = samples.mean(axis = 0).reshape(-1,1)
        
        return samples,means
    def distribution(self,X,sample_nums = 500):
        samples,_ = self.sample(X,sample_num = sample_nums)
        return stats.gaussian_kde(samples.reshape(1,-1))

        

if __name__ == "__main__":
    x_train = torch.load('r4traincode')
    x_train = x_train[:,0:32]
    y_train = torch.load('r4trainY')
    
    x_test = torch.load('r4testcode')
    x_test = x_test[:,0:32]
    y_test = torch.load('r4testY')
    
    batch_size, nb_train = len(x_train), len(x_train)
    
    net = BBP_Homoscedastic_Model_Wrapper(input_dim = 32, output_dim = 1, no_units = 128, learn_rate = 1e-2, batch_size = batch_size, no_batches = 1, init_log_noise = 0)
    
    net.load_data(x_train,y_train)
    net.train(num_epochs=1000)

    # find the predictive distribution
    kernel = net.distribution(x_test[2])

    plt.plot(np.linspace(0, 3, 200),kernel(np.linspace(0, 3, 200)))
    plt.plot(y_test[2]*np.ones(100),np.linspace(0,15,100),'r--')


