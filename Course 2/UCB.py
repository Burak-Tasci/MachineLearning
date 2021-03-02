import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')


N = len(veriler)
d = len(veriler.columns)
#Ri
oduller =[0] * d

secilenler = []
toplam = 0

#Ni
clicks = [0] * d


for n in range(N):
    ad = 0
    max_ucb = 0
    for i in range(d):
        if(clicks[i] > 0):
            avg = oduller[i] / clicks[i]
            delta = math.sqrt(3/2*math.log(n) / clicks[i])
            ucb = avg + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    clicks[ad]+=1
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    toplam+=odul
    oduller[ad]+= odul


plt.hist(secilenler)
plt.show()









