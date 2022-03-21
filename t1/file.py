#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
#import sympy as smp
import sys
sys.path.append('../common/')
from tex import tex

def density(data, n):
    c, b = np.histogram(data, bins=n)
    return (b[:-1], c)

def dens(x):
    return (1/(4*np.pi)) * (2 + np.cos(x))

def dist(x):
    return (np.sin(x) + 2 * x) / (4 * np.pi)

def fisher(x):
  return 0.5 * np.log((1 + x) / (1 - x))


#tex.printhead()
#tex.section("Лабораторная работа №1")
#tex.section("1", 1)
#1.1
n = 10000

X = np.linspace(0, 2*np.pi, n)
Y = [dens(x) for x in X]

M = np.max(Y)

sample = np.zeros(n)
i = 0
while i < n:
    x = 2 * np.pi * np.random.uniform()
    y = M * np.random.uniform()
    if y < dens(x):
        sample[i] = x
        i += 1

b, c = density(sample, 100)
plt.plot(b, c / max(c))
plt.plot(X, Y / max(Y))
plt.title('Сгенерированная выборка')
plt.legend(['Распределение выборки', 'Распределение'])
plt.savefig('dens1.png')
plt.clf()
#tex.addimage('dens1.png')

from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency(sample)
#tex.plaintext(test)
print(f'stat: {stat} pvalue: {p} dof: {dof} expected: {expected}')

#1.2
n = 3
mu = np.array([4, 12, 7])
sigma = np.array([[9, 8, 9],
                  [8, 16, 12],
                  [6, 12, 18]])
mtx = np.linalg.cholesky(sigma).T

arr = []
for i in range(100):
    y = np.random.normal(0, 1, 3)
    z = (y * mtx).T
    for row in z:
        arr.append(row)
arr = np.array(arr)

for x in range(3):
    for y in range(3):
        if x < y:
            r = np.prod(np.corrcoef(arr[x,:], arr[y,:]))
            v = 0.5 * np.log((1+r) / np.sqrt(1-r))
            ro = 0.8
            rv = (v - ro) / np.sqrt(100-3)
            alpha = 0.95
            lp = stats.norm.cdf((1-alpha)/2) - stats.norm.cdf(-(1-alpha)/2)
            print(x, y)
            if (rv < lp):
                print('Равны')
            else:
                print('Не равны')


#2
def root(x, y):
    return pow(x, 1/y)
lmbd = 26
beta = 1
n = 100

def fin(x):
  return lmbd * root(-1 * np.log(1 - x), beta)

a = 0
b = 26

def finc(x):
    y = -np.log(1 - x * (fin(b) - fin(a)) - fin(a))
    return lmbd * root(y, beta)

def f(x):
    return 1 - np.exp(-1 * ((x / lmbd) ** beta))

def dist(x):
    if x < 4:
        return 0
    if x < 15:
        return np.cos(((x * np.pi) / 11) + ((7 * np.pi) / 11) ) + 1
    if x < 25:
        return (7 / 4) + (x / 30) - ((x ** 2) / 900)
    return 0

def task2(first = True):
    Y = []
    X = []
    for i in range(n):
        u = np.random.uniform()
        if first:
            v = fin(u)
        else:
            v = finc(u)
        X.append(v)
        Y.append(dist(v))

    var = np.var(Y)
    mu = np.mean(Y)
    t = 1.95
    r = mu - (np.sqrt(var) * t / np.sqrt(n))
    l = mu + (np.sqrt(var) * t / np.sqrt(n))

    print(l, r)
    b, c = density(Y, 10)
    plt.plot(b, c)
    plt.axhline(l, color = 'r')
    plt.axhline(r, color = 'b')
    str = '1' if first else '2'
    plt.savefig(f'2plot{str}.png')
    plt.clf()


task2()
#task2(first = False)

#3
n = 100
t = 2
h = t / n
v = np.arange(h, t, h)
#print(v)
sigma = []

for x in range(len(v)):
    for y in range(len(v)):
        s = v[max(x, y)] - v[min(x, y)]
        if abs(s) >= 1:
            sigma.append(0)
        else:
            sigma.append((2 + abs(s)) * (1 - abs(s)) ** 2)


mean = 0.0
count = 0
sigma = np.reshape(sigma, (len(v), -1))
mtx = np.linalg.cholesky(sigma)
for i in range(n):
    y = np.random.normal(size = n-1)
    z = np.matmul(mtx, y).T
    zmin = 9999
    tmin = -9999
    flag = False
    for j in range(n-1):
        if z[j] < zmin and z[j] > 2:
            zmin = z[j]
            tmin = z[j]
            flag = True
    if flag:
        mean += tmin
        count += 1

print(f'mean: {mean / count}')
#tex.printend()
