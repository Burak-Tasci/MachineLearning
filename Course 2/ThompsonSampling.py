import random
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('Ads_CTR_Optimisation.csv')


N = len(data)
d = len(data.columns)


clicks  = [0] * d

chosenOnes = []
summary = 0
Ones  = [0] * d
Zeros = [0] * d

for n in range(1,N):
    ad = 0
    max_th = 0
    for i in  range(d):
        rasbeta = random.betavariate(Ones[i]+1, Zeros[i]+1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    chosenOnes.append(ad)
    reward = data.values[n,ad]
    if reward == 1:
        Ones[ad] += 1
    else:
        Zeros[ad] += 1
    summary += reward

plt.hist(chosenOnes)
plt.show() 










































