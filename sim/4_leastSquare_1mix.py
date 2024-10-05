import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from simulate_1mix import simulate

def simulate0(t, par):    
    y = simulate(t, v, V0, M0, A0, par)
    return y
 
def residuals(par):
    yP = simulate0(t,par)
    yyE = yE
    print(((yP-yyE)**2).mean())
    return yP-yyE
   
def leastSq(t, yE, v, V0, M0, A0, par):
    """
    t - time poitns, yE - exp data,
    initial value
    """            
    return least_squares(residuals, par, method='lm')


if __name__ == "__main__":
    import time
    data = pd.read_csv('data/4_Final Figure.csv')
    t = data['time (s)']
    yE = data['DP(μcal/s)']

    v, V0, M0, A0 = 0.1, 206.1, 20, 200
    n = 1
    Tmix, Titc = 5, 7.5
    pkoff = -2
    pKd = -1 # uM
    H = -10/10**3  #(cal/umol)
    Hdil = -1/10**2

    par =(n, Tmix, Titc, pkoff, pKd, H, Hdil) # parameter with 5 bulk index

    start = time.time()    
    mod = leastSq(t, yE, v, V0, M0, A0, par)
    end = time.time()
    print("The time of simulation with Jacobian is :",\
          (end-start) * 10**3, "ms")

    pp = mod['x']

    sim = simulate(t,v, V0, M0, A0, pp)
    end = time.time()
    print("The time of simulation with Jacobian is :",
      (end-start) * 10**3, "ms")

     
    plt.plot(t,yE, 'k-',label="Exp")
    plt.plot(t,sim, 'r-', alpha=0.5, label="Fit")
    plt.xlabel('Time (s)')
    plt.ylabel('DP(μcal/s)')
    plt.legend()
    plt.savefig("result/fitting_4_1mix.png")
    plt.show()

    dof = len(data)-pp.size
    resV = np.sum(mod.fun**2)/dof
    pcov = np.linalg.inv(np.dot(mod.jac.T, mod.jac))*resV
    cv = np.sqrt(np.diag(pcov))/pp
    diag = np.sqrt(np.diag(np.diag(pcov)))
    gaid = np.linalg.inv(diag)
    pcor = gaid @ pcov @ gaid

    plt.imshow(np.log10(np.abs(pcor)))
    plt.xticks(range(len(pp)),labels=["n", "Tmix", "Titc", "pkoff", "pKd", "H", "Hdil"])
    plt.yticks(range(len(pp)), labels=["n", "Tmix", "Titc", "pkoff", "pKd", "H", "Hdil"])    
    plt.colorbar()
    plt.savefig("result/pcor_4_1mix.png")
    plt.close()


    rnames = ["n", "Tmix", "Titc", "pkoff", "pKd", "H", "Hdil"]

    df = pd.DataFrame({'coef':pp, 'se':np.sqrt(np.diag(pcov))}, index=rnames)
    df = df.join(pd.DataFrame(pcor,columns=rnames,index=rnames))
    df.to_csv("result/parameters_4_1mix.csv")   
        

##    print("chi2 = ", np.sum((np.array(np.concatenate(yP).flat)-np.array(np.concatenate(yE).flat))**2))
##
####    cv = np.sqrt(np.diag(pcov))/popt
####    diag = np.sqrt(np.diag(np.diag(pcov)))
####    gaid = np.linalg.inv(diag)
####    pcor = gaid @ pcov @ gaid
####
####    plt.imshow(np.log10(np.abs(pcor)))
####    plt.colorbar()
####    plt.savefig("221216pcor.png")
####    plt.show()
####
####    rnames = ['ptc','pka','pkd','alpha','ri1','ri2','ri3','ri4','ri5']
####    cnames = rnames
####    df = pd.DataFrame(pcor,columns=cnames, index=rnames)
####    df.to_csv("correlation.csv")
####
