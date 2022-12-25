import pandas as pd
import numpy as np

df1 = pd.read_csv('/home/katya/Downloads/output/1model.csv')

p_A = []
for j in range(1, 9):
    for i in range(1, 109):
        p_A.append(list(df1[f'p_A[{i},{j}]']))

p_B = []
for j in range(1, 11):
    for i in range(1, 109):
        p_B.append(list(df1[f'p_B[{i},{j}]']))

answers_A = pd.read_csv('/home/katya/Downloads/output/data/A.csv')
answers_B = pd.read_csv('/home/katya/Downloads/output/data/B.csv')

x_a = [list(answers_A[f'A{i}']) for i in range(1, 9)]
x_b = [list(answers_B[f'B{i}']) for i in range(1, 11)]

chi_kvadrat_obs = []
chi_kvadrat_replicated = []
for iteration in range(0, 50):
    p_A_s_kryshichkoy = p_A[:][iteration]
    p_B_s_kryshichkoy = p_B[:][iteration]

    chi_kvadrat = 0
    k = 0
    for j in range(0, 8):
        for i in range(0, 108):
            chi_kvadrat += (x_a[j][i] - p_A_s_kryshichkoy[k]) * (x_a[j][i] - p_A_s_kryshichkoy[k]) / p_A_s_kryshichkoy[
                k]
            k += 1
    k = 0
    for j in range(0, 10):
        for i in range(0, 108):
            chi_kvadrat += (x_b[j][i] - p_B_s_kryshichkoy[k]) * (x_b[j][i] - p_B_s_kryshichkoy[k]) / p_B_s_kryshichkoy[
                k]
            k += 1

    chi_kvadrat_obs.append(chi_kvadrat)

    chi_kvadrat_rep = []
    for _ in range(0, 500):
        chi_kvadrat_r = 0

        x_a_rep = [[0 for j in range(0, 108)] for i in range(0, 8)]

        k = 0
        for j in range(0, 8):
            for i in range(0, 108):
                x_a_rep[j][i] = np.random.binomial(1, p_A_s_kryshichkoy[k], size=1)[0]

        k = 0
        x_b_rep = [[0 for j in range(0, 108)] for i in range(0, 10)]
        for j in range(0, 10):
            for i in range(0, 108):
                x_b_rep[j][i] = np.random.binomial(1, p_B_s_kryshichkoy[k], size=1)[0]

        k = 0
        for j in range(0, 8):
            for i in range(0, 108):
                chi_kvadrat_r += (x_a_rep[j][i] - p_A_s_kryshichkoy[k]) * (x_a_rep[j][i] - p_A_s_kryshichkoy[k]) / \
                                 p_A_s_kryshichkoy[k]
                k += 1
        k = 0
        for j in range(0, 10):
            for i in range(0, 108):
                chi_kvadrat_r += (x_b_rep[j][i] - p_B_s_kryshichkoy[k]) * (x_b_rep[j][i] - p_B_s_kryshichkoy[k]) / \
                                 p_B_s_kryshichkoy[k]
                k += 1

        chi_kvadrat_rep.append(chi_kvadrat_r)

    chi_kvadrat_replicated.append(chi_kvadrat_rep)

print(chi_kvadrat_obs)
print(chi_kvadrat_replicated)

df2 = pd.read_csv('/home/katya/Downloads/output/2model.csv')
df3 = pd.read_csv('/home/katya/Downloads/output/3model.csv')
