
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import os

if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('GPU')
else:
  device = torch.device('cpu')
  print('CPU')

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--directed', default=True)
  parser.add_argument('--time_step', default=40)
  parser.add_argument('--num_node', default=50)
  parser.add_argument('--CP_true', default=[10,20,30])

  parser.add_argument('--latent_dim', default=3)
  parser.add_argument('--w_dim', default=3)

  parser.add_argument('--shared_layer', default=[128, 256, 512])
  parser.add_argument('--output_layer', default=[32, 128, 256])
  parser.add_argument('--num_samples', default=100)
  parser.add_argument('--langevin_K', default=20)
  parser.add_argument('--langevin_s', default=0.5)

  parser.add_argument('--decoder_lr', default=0.01)
  parser.add_argument('--decay_rate', default=0.01)
  parser.add_argument('--penalty', default=1)
  parser.add_argument('--mu_lr', default=0.01)
  parser.add_argument('--epoch', default=200)
  
  parser.add_argument('--output_dir', default='./output')

  parser.add_argument('-f', required=False) # needed in Colab

  return parser.parse_args()

###################
args = parse_args()


os.makedirs(args.output_dir, exist_ok=True)
output_dir = os.path.join(args.output_dir)
#print(output_dir)




torch.manual_seed(0)
rho = 0.0
n = args.num_node
K = 3
v = args.CP_true
data = torch.zeros(args.time_step, n, n)
sum_holder =[]

for t in range(args.time_step):
    if t == 0 or t == v[1]:
        P = torch.full((n, n), 0.3)
        P[:n // K, :n // K] = 0.5
        P[n // K:2 * (n // K), n // K:2 * (n // K)] = 0.5
        P[2 * (n // K):n, 2 * (n // K):n] = 0.5
        torch.diagonal(P).zero_()
        A = torch.bernoulli(P)

    if t == v[0] or t == v[2]:
        Q = torch.full((n, n), 0.2)
        Q[:n // K, :n // K] = 0.45
        Q[n // K:2 * (n // K), n // K:2 * (n // K)] = 0.45
        Q[2 * (n // K):n, 2 * (n // K):n] = 0.45
        torch.diagonal(Q).zero_()
        A = torch.bernoulli(Q)

    if (t > 0 and t < v[0]) or (t > v[1] and t < v[2]):
        aux1 = (1 - P) * rho + P
        aux2 = P * (1 - rho)
        aux1 = torch.bernoulli(aux1)
        aux2 = torch.bernoulli(aux2)
        A = aux1 * A + aux2 * (1 - A)

    if (t > v[0] and t < v[1]) or (t > v[2] and t <= args.time_step):
        aux1 = (1 - Q) * rho + Q
        aux2 = Q * (1 - rho)
        aux1 = torch.bernoulli(aux1)
        aux2 = torch.bernoulli(aux2)
        A = aux1 * A + aux2 * (1 - A)

    torch.diagonal(A).zero_()
    data[t,:,:] = A.clone()
    sum_holder.append(torch.sum(A))

#print(data.shape)
#plt.plot(np.arange(0, args.time_step), sum_holder)  
#plt.show()



class CPD(nn.Module):
  def __init__(self, args):
    super(CPD, self).__init__()
    self.l1 = nn.Linear( args.latent_dim, args.output_layer[0] )
    self.left1 = nn.Linear( args.output_layer[0], args.num_node * args.w_dim ) 
    self.middle1 = nn.Linear( args.output_layer[0], args.w_dim * args.w_dim ) 
    self.right1 = nn.Linear( args.output_layer[0], args.num_node * args.w_dim ) 

  def forward(self, z):
    
    output = self.l1(z).tanh()
    w_left = self.left1(output).tanh()
    w_middle = self.middle1(output).tanh()
    w_right = self.right1(output).tanh() 

    w_left = w_left.reshape(args.num_samples, args.num_node, args.w_dim)
    w_middle = w_middle.reshape(args.num_samples, args.w_dim, args.w_dim)
    w_right = w_right.reshape(args.num_samples, args.num_node, args.w_dim)
    output = torch.bmm(torch.bmm(w_left, w_middle),torch.transpose(w_right, 1, 2)).sigmoid() # n by n

    return output
    
  def infer_z(self, z, adj_gt_vec, mu_t):
    '''
    z:          m by d
    adj_gt_vec: m*n*n (with repetition)
    mu_t_mat:   d
    '''

    criterion = nn.BCELoss(reduction='sum') # take the sum ???? divided by m

    for k in range(args.langevin_K):

      z = z.detach().clone()
      z.requires_grad = True
      assert z.grad is None

      adj_prob = self.forward(z) # m by (n*n)
      nll = criterion( adj_prob.view(-1), adj_gt_vec ) # both are m*n*n
      z_grad_nll = torch.autograd.grad(nll, z)[0] # m by d 

      z = z - args.langevin_s * (z_grad_nll + (z-mu_t)) + \
          torch.sqrt(2*torch.tensor(args.langevin_s)) * torch.randn(args.num_samples, args.latent_dim).to(device)

    z = z.detach().clone()
    return z
    
  def cal_log_lik(self, mu):
  
    log_lik = 0.0
    
    for t in range(args.time_step):
      mu_t = mu[t,:].detach().clone() # d
      z = torch.randn(args.num_samples, args.latent_dim).to(device) + mu_t.to(device)  # m by d
      adj_prob = self.forward(z) # m by (n*n)
      adj_prob = torch.mean(torch.prod(adj_prob, dim=1)) # product over dyads dim=1, average over m # should be a scalar
      log_lik += torch.log(adj_prob)
    
    return log_lik
    
    
    
    
    
    
    

data = data.to(device)
T = data.shape[0]
mu = torch.randn(T, args.latent_dim).to(device) * 0.01 # initialize as random, divided by norm of row diff (cannot be identical)

mu_old = mu.detach().clone()
loss_holder = []
log_lik_holder = []

model = CPD(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.decoder_lr, weight_decay=args.decay_rate) 
criterion = nn.BCELoss(reduction='sum') # sum for expectation, later divided by m
model.train()

for learn_iter in range(args.epoch):

  loss = 0.0

  for t in range(T):
    mu_t = mu[t,:].detach().clone() # d

    adj_gt = data[t,:,:].detach().clone() # n by n
    adj_gt_vec = adj_gt.view(-1).repeat(args.num_samples) # m*n*n (with repetition)
    
    # sample from posterior
    init_z = torch.randn(args.num_samples, args.latent_dim).to(device) # m by d, starts from N(0,1)
    sampled_z = model.infer_z(init_z, adj_gt_vec, mu_t) # m by d, m samples of z from langevin

    adj_prob = model(sampled_z) # m by (n*n) # m samples of adj_prob from the decoder
    loss += criterion(adj_prob.view(-1), adj_gt_vec) / args.num_samples  # both are m*n*n

  loss_holder.append(loss.detach().cpu().numpy())

  # update decoder
  for param in model.parameters():
    param.grad = None
  loss.backward()
  optimizer.step()
   
  
  #r = list(range(T))
  #random.shuffle(r)
  #for t in r:
  for t in range(T):
    mu_t = mu[t,:].detach().clone() # d

    adj_gt = data[t,:,:].detach().clone() # n by n
    adj_gt_vec = adj_gt.view(-1).repeat(args.num_samples) # m*n*n (with repetition)
    
    # sample from posterior
    init_z = torch.randn(args.num_samples, args.latent_dim).to(device) # m by d, starts from N(0,1)
    sampled_z = model.infer_z(init_z, adj_gt_vec, mu_t) # m by d, m samples of z from langevin

    if t == 0:  
      grad_mu_t = -(sampled_z - mu_t).mean(dim=0) - args.penalty * (1/torch.norm(mu[1,:] - mu[0,:],p=2)) * (mu[1,:] - mu[0,:])
    elif t == T-1:
      grad_mu_t = -(sampled_z - mu_t).mean(dim=0) + args.penalty * (1/torch.norm(mu[t,:] - mu[t-1,:],p=2)) * (mu[t,:] - mu[t-1,:])
    else:
      grad_mu_t = -(sampled_z - mu_t).mean(dim=0) - args.penalty * (1/torch.norm(mu[t+1,:] - mu[t,:],p=2)) * (mu[t+1,:] - mu[t,:]) \
                                                  + args.penalty * (1/torch.norm(mu[t,:] - mu[t-1,:],p=2)) * (mu[t,:] - mu[t-1,:]) 

    mu = mu.detach().clone()
    mu[t,:] -=  args.mu_lr * grad_mu_t # gradient descent 

  # early stopping
  log_lik = model.cal_log_lik(mu)
  log_lik_holder.append(log_lik.detach().cpu().numpy())
  
  
  if learn_iter > args.epoch+2:
    if log_lik_holder[learn_iter-2] < log_lik_holder[learn_iter-1] and log_lik_holder[learn_iter-1] > log_lik_holder[learn_iter]:
      plt.plot(np.arange(len(log_lik_holder)), log_lik_holder)
      plt.savefig( os.path.join(output_dir, 'log_lik.png') )
      plt.close()
    
      signal = 0.5*torch.norm(-torch.diff(mu, dim=0), p=2, dim=1)**2 
      signal = signal.cpu().detach().numpy()
      plt.plot(np.arange(0, args.time_step-1), signal)  
      plt.savefig( os.path.join(output_dir, 'mu_diff_result.png') )
      plt.close()
      print('EARLY STOPPING')
      break
  
  
  if (learn_iter+1) % 10 == 0:
    print('\n')
    print('learning iter =', learn_iter)
    print('decoder loss =',loss)
    print('mu residual =',torch.mean((mu-mu_old)**2))
    print('mu relative difference =',torch.norm(mu-mu_old,  p='fro') / torch.norm(mu_old,  p='fro'))
    print('log likelihood =', log_lik)
    
    mu_old = mu.detach().clone()
    signal = 0.5*torch.norm(-torch.diff(mu, dim=0), p=2, dim=1)**2 
    signal = signal.cpu().detach().numpy()
    
    plt.plot(np.arange(0, args.time_step-1), signal)  
    plt.savefig( os.path.join(output_dir, 'mu_diff_{}.png'.format(learn_iter)) )
    plt.close()





plt.plot(np.arange(len(loss_holder)), loss_holder)
plt.savefig( os.path.join(output_dir, 'decoder_loss.png') )
plt.close()

plt.plot(np.arange(len(log_lik_holder)), log_lik_holder)
plt.savefig( os.path.join(output_dir, 'log_lik.png') )
plt.close()


'''
signal = torch.norm(mu, p=2, dim=1)**2 
signal = signal.cpu().detach().numpy()
plt.plot(np.arange(0, args.time_step), signal)  
plt.show()

#mu, loss_holder = main(args, data)

# torch.diff: second row - first row (then minus sign)# torch.norm: 2-norm of each row# take a squared
#signal = 0.5*torch.norm(-torch.diff(mu, dim=0), p=2, dim=1)**2 
#signal = signal.cpu().detach().numpy()
#plt.plot(np.arange(0, args.time_step-1), signal)  
#plt.show()  

#print(len(loss_holder))
plt.plot(np.arange(0,len(loss_holder)), loss_holder)  
plt.show()
'''



