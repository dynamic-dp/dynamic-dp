import math 
import numpy as np
def cal_step_decay_rate(target_percent,T):
    return np.exp(np.log(target_percent)/T)

def mu0_search(mu,T,decay_rate,p,mu_t=None):
    low_mu = 0.1*mu
    if mu_t:
        high_mu = mu_t
    else:
        high_mu = 50*mu
    for i in range(1000):
        mus = []
        mu_0 = (low_mu+high_mu)/2
        for t in range(T):
            mus.append(mu_0/decay_rate**(t))
        if(p*math.sqrt(sum(np.exp(np.array(mus)**2)-1))>mu):
            high_mu = mu_0
        else:
            low_mu = mu_0
    return mu_0