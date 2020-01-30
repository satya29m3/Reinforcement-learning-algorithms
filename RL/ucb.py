import matplotlib.pyplot as plt
import numpy as np
from ep_greedy import run_exp
from opt_val import run_exp_opt
class bandit:
    def __init__(self,m):
        self.m = m
        self.mean = 0
        self.N = 1
    
    def pull(self):
        return np.random.randn() + self.m
    
    def update(self,x):
        self.N += 1
        self.mean = (1-1/self.N)*self.mean + 1/self.N*x

def ucb(mean,n,nj):
    if nj==0:
        return float('inf')
    return mean + np.sqrt(2*np.log(n)/nj)

def run_exp_ucb(m1,m2,m3,N):
    bandits= [bandit(m1),bandit(m2),bandit(m3)]
    data = np.empty(N)

    for i in range(N):
        j = np.argmax([ucb(b.mean,i+1,b.N) for b in bandits])
        # print(j)
        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x
    cum_avg = np.cumsum(data) / (np.arange(N)+1)

    plt.plot(cum_avg)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)
    
    return cum_avg

if __name__ == '__main__':
    c1 = run_exp_ucb(1.0,2.0,3.0,100000)
    eps = run_exp(1,2,3,0.1,100000)
    opt = run_exp_opt(1,2,3,100000)
    plt.plot(c1)
    plt.plot(eps)
    plt.plot(opt)

    plt.show()


