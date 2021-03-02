
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

veriler = pd.read_csv('sepet.csv',header = None)
t = []
print(veriler.shape)

for i in range(veriler.shape[0]):
    t.append(list(str(veriler.values[i,j]) for j in range(veriler.shape[1])))
    
from apyori import apriori

rules = apriori(transactions=t,min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)

print(list(rules))









