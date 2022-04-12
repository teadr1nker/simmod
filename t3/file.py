#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#1
print('1)')
data = [float(d.strip()) for d in open('data.txt', 'r').read().split('\n')[:-1]]

k = np.ones(2)
N = 100
sigma = .2
Ca = 10
Cb = 15
t = np.arange(0, len(data))

def A(t, k):            #solving differential equation
    delta = k[0] / k[1]
    a = (1 / (1 + delta)) * (Ca+ Cb)
    a = a + ((1 / (1 + delta)) * (Ca+ Cb) - Cb ) * np.exp(-(k[0]+k[1])*t)
    return a

def prob(k):            #probability function
    s = sum([(data[i] - A(t[i], k))**2 for i in range(len(data))])
    return np.exp((-1/(2*sigma**2)) * s)

#generating probabilities
u  = np.random.normal(0, sigma**2, N)
u1 = np.random.normal(0, sigma**2, N)
u2 = np.random.normal(0, sigma**2, N)

for i in range(N):      #measuting concentrations
    y = k + np.array([u1[i], u2[i]])
    pk = prob(k)
    alpha = 0 if pk == 0 else prob(y) / pk
    k = y if u[i] < alpha else k

print(k)

#2
print('\n2)')
def discreteDist(P):
    u = np.random.uniform()
    cur = 0; prev = 0.
    for p in P:
        if u <= prev + p:
            break
        prev += p
        cur+=1
    return cur

def task2(file, alpha, beta, nTheme):
    df = pd.read_csv(file, delimiter=' ')
    nDoc = max(df['doc'])
    nWord = max(df['word'])

    countW = np.zeros((nTheme, nWord))
    countD = np.zeros((nTheme, nDoc))
    countT = np.zeros(nTheme)
    themeDW = np.zeros((nDoc, nWord))
    #print('data gathered')
    #init
    for doc in range(nDoc):
        for word in range(nWord):
            theme = np.random.randint(0, nTheme)
            countW[theme, word] += 1
            countD[theme, doc]  += 1
            countT[theme] += 1
            themeDW[doc, word] = theme
    #print('initialized')
    phi =   np.zeros((nTheme, nWord))
    theta = np.zeros((nDoc, nTheme))

    #Gibbs
    M = 10  #runs
    for i in range(M):
        for doc in range(nDoc):
            words = np.array(df.query(f'doc == {doc+1}')['word']) - 1
            for word in words:
                theme = int(themeDW[doc, word])
                countW[theme, word] -= 1
                countD[theme, doc]  -= 1
                countT[theme] -= 1

                P = np.array([
                (alpha + countD[theme, doc]) * (beta + countW[theme, word])\
                / (beta * nWord + countT[theme]) for theme in range(nTheme)
                ])
                P /= P.sum()

                theme = discreteDist(P)
                countW[theme, word] += 1
                countD[theme, doc]  += 1
                countT[theme] += 1
                themeDW[doc, word] = theme
        #calc theta
        for doc in range(nDoc):
            words = np.array(df.query(f'doc == {doc+1}')['word']) - 1
            for theme in range(nTheme):
                sum = 0
                for word in words:
                    if theme == themeDW[doc, word]:
                        sum += themeDW[doc, word]
                theta[doc, theme] += (alpha + sum) / (alpha * nTheme + nDoc)

        #calc phi
        for theme in range(nTheme):
            for doc in range(nDoc):
                words = np.array(df.query(f'doc == {doc+1}')['word']) - 1
                for word in range(nWord):
                    sumz = 0; sumy = 0
                    if themeDW[doc, word] == theme:
                        sumy += themeDW[doc, word]
                        if word in words:
                            sumz += themeDW[doc, word]
                    phi[theme, word] += (beta + sumz) / (beta * nWord + sumy)
    print(f'Theta theme:doc\n{theta / M}')
    print(f'Phi theme:word\n{phi.T / M}')
print('test1')
task2('test1.dat', 1, 1, 3)
#print('test2')
#task2('test2.dat', .1, .1, 20)

#3
if input('\n3) run Ising model? y ') != 'y':
    quit(0)

beta1 = 1.
beta2 = 1.5
n = int(1e2)
m = int(1e5)
frame = int(m/120)

def P( x, i, j, n):         #probability function
    sum1 = 0.
    sum2 = 0.

    if 1 <= i:
        sum1 += x[i-1][j]
    if j < n-1:
        sum2 += x[i][j+1]
    if i < n-1:
        sum1 += x[i+1][j]
    if 1 <= j:
        sum2 += x[i][j-1]

    return np.exp( - x[i][j] * beta2 * sum2 - x[i][j] * beta1 * sum1 )

x = 2 * np.random.randint(2, size=(n, n)) - 1

c = 0
while m >= c:
    i = np.random.randint(0, n-1);j = np.random.randint(0, n-1)
    if P(x, i, j, n) > np.random.uniform():   #inversing
        x[i, j] = -x[i, j]

    if c % frame == 0:
        #fig, ax = plt.subplots()
        plt.imshow(x, cmap='binary')
        plt.title(f'Frame: {c // frame}')
        plt.savefig(f'frames/tmp_{c // frame}.png')
        plt.clf()

    c += 1

# *nix moment
os.system('ffmpeg -hide_banner -loglevel error -r 15 -i frames/tmp_%1d.png -c:v\
 libx264 -vf fps=24 -pix_fmt yuv420p video.mp4')
