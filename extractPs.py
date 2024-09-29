"""
Convert Q(t) from Q(m) from ITC injection table
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import os
import findV0_n

infold, outfold = "injectionTable/", "Qt2/"
files = [f for f in os.listdir(infold) if f.endswith('Figure.csv')]
for f in files:
    df = pd.read_csv(infold+f)
    t = round(df['DP_X']*60)-180
    dp = df['DP_Y'][(t>0) & (t<380)]
    t = t[(t>0) & (t<380)]
    out = pd.DataFrame({'time (s)': t,'DP(μcal/s)':dp})
    out.to_csv(outfold+f)
