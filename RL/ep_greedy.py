import matplotlib.pyplot as plt
import numpy as np

class bandit:
    def __init__(self,m):
        self.m = m
        self.mean = 0
        self.N = 0
    
    def pull(self):
        return np.random.randn() + self.m
    
    def update(self,x):
        self.N += 1
        self.mean = (1-1/self.N)*self.mean + 1/self.N*x


def run_exp(m1,m2,m3,eps,N):
    bandits= [bandit(m1),bandit(m2),bandit(m3)]
    data = np.empty(N)

    for i in range(N):

        p = np.random.random()
        if p < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])
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
    c1 = run_exp(1.0,2.0,3.0,0.1,20)
    c2 = run_exp(1.0,2.0,3.0,0.05,20)
    c3 = run_exp(1.0,2.0,3.0,0.01,20)

    plt.plot(c1,label = 'eps = 0.1')
    plt.plot(c2,label = 'eps = 0.05')
    plt.plot(c3,label = 'eps = 0.01')
    plt.show()


