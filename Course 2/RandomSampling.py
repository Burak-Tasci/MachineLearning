
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')
toplam = 0
secilenler = list()


for i in range(len(veriler)):
    ad = random.randrange(len(veriler.columns))
    secilenler.append(ad)
    odul = veriler.values[i,ad]
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()






