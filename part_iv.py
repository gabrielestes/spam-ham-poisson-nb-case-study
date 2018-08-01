import scipy.stats as st
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df = pd.get_dummies(df, columns=['v1']).drop(columns=['v1_ham'])

x_bar = df.v1_spam.sum()/5572
rv = poisson(x_bar)
h0 = .125
n = len(df)
s = df.v1_spam.std(ddof=1)
se = s/np.sqrt(n)

z_value = (x_bar - h0)/se
p_value = 1 - st.norm.cdf(z_value) # Part 2 - Reject the null hypothesis

lam = df.v1_spam.sum()/28
rv = poisson(lam)
thirty_emails = 1 - rv.cdf(30) # probability of >= 30 emails/day

true_pos, true_neg, type_1, type_2 = 0, 0, 0, 0

for x in range(10):
    row = df.sample()
    print(row.v2)
    answer = input('Spam or nah?[y/n]')
    print('Answer was: ', row.v1_spam.item())
    if answer == 'y':
        if row.v1_spam.item() == 1:
            true_pos += 1
        else:
            type_1 += 1
    else:
        if row.v1_spam.item() == 1:
            type_2 += 1
        else:
            true_neg += 1

print('true pos: {}\n true neg: {}\n type 1: {}\n type 2: {}'.format(true_pos, true_neg, type_1, type_2))

accuracy = (true_neg + true_pos)/10
print('You answered {} correctly.'.format(accuracy)) # 0.8 Accuracy, Type 1 = 2, Type 2 = 0
