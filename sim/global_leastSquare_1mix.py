import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from simulate_1mix import simulate

def simulate0(t, par):
    y =[]
    for i in range(len(t)):
        parT = tuple(par[(i*4):(i*4+3)])+tuple(par[-3:])+(par[i*4+3],)
        y.append(simulate(t[i], v, V0, M0, A0, parT))
    return np.array(y).flatten()
 
def residuals(par):
    yP = simulate0(t,par)
    yyE = np.array(yE).flatten()
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
    files = ['data/'+str(i)+'_Final Figure.csv' for i in range(3,6)]
    t, yE = [], []
    for f in files:
        data = pd.read_csv(f)
        t.append(data['time (s)'].values)
        yE.append(data['DP(μcal/s)'].values)

    v, V0, M0, A0 = 0.1, 206.1, 20, 200
    
    n1, n2, n3 = 1.2,1,1
    Tmix1, Titc1, Tmix2, Titc2, Tmix3, Titc3 = 3, 7.5, 3, 7.5, 3, 7.5
    Hdil1, Hdil2, Hdil3 = -1/10**3, -1/10**3, -1/10**3
    pkoff = -4
    pKd = -2 # uM
    H = -13/10**3  #(cal/umol)

    par =(n1, Tmix1, Titc1, Hdil1,\
          n2, Tmix2, Titc2, Hdil2,\
          n3, Tmix3, Titc3, Hdil3,\
          pkoff, pKd, H) # parameter with 5 bulk index

    start = time.time()    
    mod = leastSq(t, yE, v, V0, M0, A0, par)
    end = time.time()
    print("The time of simulation with Jacobian is :",\
          (end-start) * 10**3, "ms")

    pp = mod['x']

    end = time.time()
    print("The time of simulation with Jacobian is :",
      (end-start) * 10**3, "ms")

    fig, ax = plt.subplots(nrows=1, ncols=len(t),sharey=True)
    for i in range(len(t)):
        ppT = tuple(pp[(i*4):(i*4+3)])+tuple(pp[-3:])+(pp[i*4+3],)
        sim = simulate(t[i],v, V0, M0, A0, ppT)
        ax[i].plot(t[i],yE[i], 'k-',label="Exp")
        ax[i].plot(t[i],sim, 'r-', alpha=0.5, label="Fit")
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('DP(μcal/s)')
    plt.legend()
    plt.savefig("result/fitting_global_1mix.png")
    plt.show()

    dof = len(data)-pp.size
    resV = np.sum(mod.fun**2)/dof
    pcov = np.linalg.inv(np.dot(mod.jac.T, mod.jac))*resV
    cv = np.sqrt(np.diag(pcov))/pp
    diag = np.sqrt(np.diag(np.diag(pcov)))
    gaid = np.linalg.inv(diag)
    pcor = gaid @ pcov @ gaid

    plt.imshow(np.log10(np.abs(pcor)))
    plt.xticks(range(len(pp)),labels=['n1', 'Tmix1', 'Titc1', 'Hdil1',\
                                      "n2", "Tmix2", "Titc2", 'Hdil2',\
                                      'n3', 'Tmix3', 'Titc3', 'Hdil3',\
                                      "pkoff", "pKd", "H"])
    plt.yticks(range(len(pp)), labels=['n1', 'Tmix1', 'Titc1', 'Hdil1',\
                                      "n2", "Tmix2", "Titc2", 'Hdil2',\
                                      'n3', 'Tmix3', 'Titc3', 'Hdil3',\
                                      "pkoff", "pKd", "H"])
    plt.colorbar()
    plt.savefig("result/pcor_global_1mix.png")
    plt.close()


    rnames = ['n1', 'Tmix1', 'Titc1', 'Hdil1',\
              "n2", "Tmix2", "Titc2", 'Hdil2',\
              'n3', 'Tmix3', 'Titc3', 'Hdil3',\
              "pkoff", "pKd", "H"]
    
    df = pd.DataFrame({'coef':pp, 'se':np.sqrt(np.diag(pcov))}, index=rnames)
    df = df.join(pd.DataFrame(pcor,columns=rnames,index=rnames))
    df.to_csv("result/parameters_global_1mix.csv")   
        

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
