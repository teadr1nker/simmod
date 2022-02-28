#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
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

def fisher(x)
  return 0.5 * np.log((1 + x) / (1 - x))


tex.printhead()
tex.section("Лабораторная работа №1")
tex.section("1", 1)
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
tex.addimage('dens1.png')

test = stats.kstest(sample, dens)
tex.plaintext(test)

#1.2



tex.printend()
