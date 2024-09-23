"""
back calculate the V0 and starting n from ITC injection table
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import optimize as opt

syr = 0.2
dV = 0.1

def xt(x, n, v0):
    """ Xt eq from Malvern equations"""
    return syr*(dV*(x+n))/v0*(1-dV*(x+n)/2/v0)

def estN_V0(x,y):
    popt, pcov = opt.curve_fit(xt, x, y, p0=[0,200])
    return popt


if __name__ == "__main__":
    df = pd.read_csv("injectionTable/Injection Table.csv")
    est = estN_V0(df['Unnamed: 0'], df['Xt (mM)'])
    print(est)

