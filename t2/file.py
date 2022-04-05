#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

muaRPE = np.log(10) * 5 * 120.3
muaB = np.log(10) * 150 * (3995.68 * 0.7 + 3903.64 * (1-0.7)) / 64500

df = pd.DataFrame({
'name' : ['blood', 'retina', 'RPE', 'chorid', 'sclera'],
'mua'  : [muaB, 0.45, muaRPE, 22.69, 0.46],
'mus'  : [70.75, 31, 116.7, 72.37, 103.4],
'g'    : [0.96, 0.97, 0.84, 0.945, 0.9],
'thick': [0, 200, 10, 250, 70]
})

decartDist = np.zeros((800, 1010))
decartWeight = np.zeros((800, 1010))

cylDist = np.zeros((100, 640))
cylWeight = np.zeros((100, 640))

alphaRDist = np.zeros((1000, 1300))


def pCosT(theta, g):
    y = ((1 - g**2)**2) / (4*theta**2)**(1/3)
    y = (-y + 1 + g**2) / (2*g)
    return 1 if pd.isna(y) else y

mi = 1000
ma = -1000


def absorbtion(x, y, z, w):
    xx = int(x*1000 + 300)
    zz = int(z*1000 + 10)

    if xx >= 1 and xx <= 800 and zz >= 1 and zz <= 1010:
        #print('added w')
        decartWeight[xx, zz] += w
        Q = (decartDist[xx, zz] + decartWeight[xx, zz])# ** 1e6
        decartDist[xx, zz] += Q

    ro = int(np.sqrt(x**2 + y**2) * 100)
    phi = int(np.arctan2(y, x) * 100 + 32)

    cylWeight[ro, phi] += w
    Q = (cylDist[ro, phi] + cylWeight[ro, phi])# ** 1e6
    cylDist[ro, phi] += Q

    #total
    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = int(np.arccos(z / r) * 100)
    r = int(r * 100)
    r = 999 if r >= 1000 else r
    alphaRDist[r, alpha] += w

def where(L):
    cur = 0.
    for i, t in enumerate(df['thick']):
        cur += t
        if L < cur:
            return i
    return len(df['thick'])

wLim = 0.02
N = 200
m = 6

for i in range(N):
    mua = df['mua'][0]
    mus = df['mus'][0]
    dist = np.random.exponential(mus + mua)
    w = 1
    x = y = 0
    z = dist
    while True:
        env = where(dist)
        #print(env, end='')
        if env >= len(df['name']):
            print('passed!')
            break
        mua = df['mua'][env]
        mus = df['mus'][env]
        #g = df['g'][env]
        if mua / (mua+mus) > np.random.uniform(): #consumed
            absorbtion(x, y, z, w)
            break

        #dispersion
        phi = np.random.uniform(0, 2 * np.pi)
        u  =  np.random.uniform()
        theta = np.arccos(pCosT(1-2*u, g))
        theta = 1 if pd.isna(theta) else theta
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        #decrease of weight
        w-= w * mua / (mua+mus)
        if w < wLim:
            if np.random.uniform() < 1/m:   #consumption
                absorbtion(x, y, z, w)
                break
            w = w * m

        if z >= 0:
            r = np.sqrt(x**2 + y**2 + z**2)
            alpha = int(np.arccos(z / r) * 100)
            r = int(r * 100)
            alphaRDist[r, alpha] += w

        dist += z

data = {}
#1
X = []
Y = []
Z = []
for i, row in enumerate(decartDist):
    for j, value in enumerate(row):
        if value != 0.:
            X.append((i-300) / 800)
            Y.append((j - 10) / 1010)
            Z.append(value)

data['flat'] = {'x': X, 'y': Y, 'z': Z}



plt.scatter(X, Y)
plt.xlabel('x'); plt.ylabel('y')
plt.savefig('flat1.png')
plt.clf()
plt.scatter(X, Z)
plt.xlabel('x'); plt.ylabel('w')
plt.savefig('flat2.png')
plt.clf()

#2
X = []
Y = []
Z = []

for i, row in enumerate(cylDist):
    for j, value in enumerate(row):
        if value != 0.:
            X.append(i / 100)
            Y.append((j-32)/640)
            Z.append(value)
plt.scatter(X, Y)
plt.xlabel('r'); plt.ylabel('alpha')
plt.savefig('cylinder.png')
plt.clf()
data['cylinder'] = {'x': X, 'y': Y, 'z': Z}

#3
X = []
Y = []
Z = []

for i, row in enumerate(alphaRDist):
    for j, value in enumerate(row):
        if value != 0.:
            X.append(i/1000)
            Y.append(j/1300)
            Z.append(value)

plt.scatter(X, Z)
plt.xlabel('r'); plt.ylabel('alpha')
plt.savefig('weight')
plt.clf()
data['weight'] = {'x': X, 'y': Y, 'z': Z}
json.dump(data, open("data.json", "w"))
