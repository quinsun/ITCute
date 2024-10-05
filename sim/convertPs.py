"""
Convert Q(t) from Q(m) from ITC injection table
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import os
import findV0_n

infold, outfold = "injectionTable/", "Qt/"
files = [f for f in os.listdir(infold) if f.endswith('Table.csv')]
for f in files:
    df = pd.read_csv(infold+f)
    est = findV0_n.estN_V0(df['Unnamed: 0'], df['Xt (mM)'])
    n = round(est[0])
    out = pd.DataFrame({'time (s)': n+df['Unnamed: 0'],'Q (μcal)':df['ΔQ (μcal)']})
    out.to_csv(outfold+f)
