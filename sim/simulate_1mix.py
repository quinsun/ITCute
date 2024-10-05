import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def simulate(t, v, V0, M0, A0, par):
    """
    t - time poitns
    known: v - inject rate; V0 - cell volume; A0 - syringe conc
    fitting n; koff; Kd; Tmix - Mixing delay
    """
    par = np.array(par)
    n = par[0]
    Tmix, Titc = par[1:3]
    koff, Kd = par[3:5] #log10
    H, Hdil= par[5:7]
              
    def dy_dt(y, t):  # M + A = MA
        M, As, A, MA, Qs, Pm = y
        dM_dt = -v/V0*M
        dAs_dt = (A0-A)*v/V0
        dA_dt = (As-A)/Tmix
        dMA_dt = 10**(koff-Kd)*(M-MA)*(A-MA)-10**koff*MA-v/V0*MA
        dQs_dt = H*V0*(10**(koff-Kd)*(M-MA)*(A-MA)-10**koff*MA)+Hdil*V0*dA_dt
        dPm_dt = (dQs_dt-Pm)/Titc
        return [dM_dt,dAs_dt, dA_dt,dMA_dt,dQs_dt, dPm_dt]
    
##    def jacobian(y,t):
##        P, PA, A = y[0:3]
##        return [[kaA*A,kdA,-kaA*P],\
##                [kaA*A,-kdA,kaA*P],\
##                [-kaA*A,kdA,-tcA-kaA*P]]

        
    # integrate
    y0 = [M0*n,0,0,0,0,0]
    yPA = []
    ds, info = odeint(dy_dt, y0, t, #Dfun=jacobian, \
                          full_output=True)#, tcrit= inj[i:i+2])
    p = ds[:,-1]

    return p


if __name__ == "__main__":
    '''
   fk506->fkbp12: ka 6*10**5 /M*s; kd 1*10**(-3) /s
   CN->fkbp12:fk506: ka 10**5.8 /M*s; kd 10**(-4) /s
   '''
    
    import time
    data = pd.read_csv('data/12Final Figure.csv')
    t = data['time (s)']
    pE = data['DP(μcal/s)']

    v, V0, M0, A0 = 0.1, 206.1, 20, 200
    n = 0.779
    Tmix, Titc = 5, 8
    koff = 2
    Kd = 0.0435 # uM
    H = -10/10**3  #(cal/umol)
    Hdil = -1/10**2

    par =(n, Tmix, Titc, koff, Kd, H, Hdil) # parameter with 5 bulk index

    start = time.time()

    sim = simulate(t,v, V0, M0, A0, par)
    end = time.time()
    print("The time of simulation with Jacobian is :",
      (end-start) * 10**3, "ms")

     
    plt.plot(t,pE, 'k-',label="Exp")
    plt.plot(t,sim, '-', label="Sim")
    plt.xlabel('Time (s)')
    plt.ylabel('DP(μcal/s)')
    plt.legend()
    plt.savefig("result/simulate_12.png")
    plt.show()

