#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
#import sympy as smp


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
print('1.1)')
n = 50


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

b, c = density(sample, 10)
plt.plot(b, c / max(c))
plt.plot(X, Y / max(Y))
plt.title('Сгенерированная выборка')
plt.legend(['Распределение выборки', 'Распределение'])
plt.savefig('dens1.png')
plt.clf()
#tex.addimage('dens1.png')

from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency([sample, Y])
#print(test)
stat_table = 28941
print(f'stat: {stat} stat_table: {stat_table} pvalue: {p} dof: {dof}')
if stat < stat_table:
    print('H0 верна')
else:
    print('H0 не верна')

#1.2
print('\n1.2)')

cov = np.array([[0.99, 0, 0],
                [0, 0.2, 0],
                [0, 0, 0.5]])
print(f'Матрица\n{cov}')
Xs = np.random.multivariate_normal([0, 0, 0], cov, size = n)
covm = np.ma.cov(Xs,rowvar=False)
print(f'Матрица ковариации сгенерированных данных\n{covm}')
dif = cov-covm
print(f'Разница\n{dif}')
#print(f'prod: {np.pord(dif)}')


#2
print('\n2)')
def root(x, y):
    return pow(x, 1/y)
lmbd = 26
beta = 1
n = 100

def fin(x): #weibull inv
  return lmbd * root(-1 * np.log(1 - x), beta)

a = 0
b = 32

def f(x): #weibull
    return 1 - np.exp(-1 * ((x / lmbd) ** beta))

def finc(x): #weibull inv constrained
    y = -np.log(1 - x * (f(b) - f(a)) - f(a))
    return lmbd * root(y, beta)

def dist(x):
    if x < 4:
        return 0
    if x < 15:
        return np.cos(((x * np.pi) / 11) + ((7 * np.pi) / 11) ) + 1
    if x < 25:
        return (7 / 4) + (x / 30) - ((x ** 2) / 900)
    return 0

def task2(first = True):
    Y = np.random.uniform(size=n)
    if first:
        X = np.array([fin(y) for y in Y])
    else:
        X = np.array([finc(y) for y in Y])

    Wind = np.array([dist(x) for x in X])
    WindM = Wind.mean()
    S = Wind.std()
    if first:
        q = fin(0.05)
    else:
        q = fin(0.05)

    Z = q * S / np.sqrt(n)
    a = WindM - Z; b = WindM +Z
    print(f'Доверительный интервал: {a}:{b}')
    B, c = density(Wind, 20)
    plt.plot(B, c)
    plt.axvline(a, color = 'r')
    plt.axvline(b, color = 'g')
    str = '1' if first else '2'
    plt.savefig(f'2plot{str}.png')
    plt.clf()

task2()
task2(first = False)

#3
print('\n3)')
def Y(s):
    if np.abs(s) >= 1:
        return 0
    else:
        return (2 + np.abs(s)) * (1 - np.abs(s))**2

N = 1000
n = 10

t = np.zeros(N)
for i in range(N):
    T = np.random.uniform(size=n)
    mtx = np.zeros((n,n))
    for x in range(n):
        for y in range(n):
            mtx[x, y] = Y(T[x] - T[y])

    mtx = linalg.cholesky(mtx)
    S = np.random.uniform(size=n)
    X = np.dot(mtx, S)
    Xmin = 9999
    pos = 0
    for i, x in enumerate(X):
        if x > 2 and x < Xmin:
            Xmin = x; pos = i

    t[i] = T[np.argmin(pos)]

print(f'mean: {t.mean()}')

#4
print('\n4)')
E0 = 64
d = 3
l = 2
N = 1000
E = np.zeros(N)

for i in range(N):
    deltal = d
    e = E0
    while True:
        x = np.random.uniform(0, l)
        fs = -l * np.log(x)
        if e < 0 or deltal - fs < 0:
            break
        e -= np.random.uniform(0, E0)
    E[i] = e

print(f'loss mean: {E.mean()}')

#5
print('\n5)')
def check(pos, xy):
    mtx = np.array([[-1, 0],
                    [1, 0],
                    [0, -1],
                    [0, 1]])
    ngb = []
    for i in range(4):
        pt = [mtx[i, 0] + pos[0], mtx[i, 1] + pos[1]]
        if pt not in xy:
            ngb.append(pt)

    return ngb if len(ngb) > 0 else None


dist = 0
N = 1024
n = 16
finish = 0
#used = []
for i in range(N):
    pos = [0, 0]
    xy = [pos]
    for x in range(n):
        ngb = check(pos, xy)
        if not ngb:
            break
        shift = np.random.randint(0, len(ngb))
        pos = ngb[shift]
        xy.append(pos)
        #used.append(pos)
        if len(xy) == n:
            dist += np.sqrt(xy[-1][0]**2 + xy[-1][0]**2)
            finish += 1
            mtx = np.asmatrix(xy)
            if finish < 256: # add first 256 successfull paths to plot
                plt.plot(mtx[:,0], mtx[:,1])

plt.plot([0], [0], 'ro')
plt.savefig('wandering.png')
print(f'stuck {N - finish} times! success: {finish}')
print(f'mean: {dist/finish}')
#print(f'Generated {len(used)} unique paths')


#6
print('\n6)')
coords = [1, 2, 3]
R = 1
n = 100
N = 1000

def V(w):
    a = sum([v ** 2 for v in w])
    b = 2 * sum([c * w[i] for i, c in enumerate(coords)])
    c = sum([c ** 2 for c in coords]) - R ** 2
    D = b**2 - 4 * a * c
    return 0 if D >= 0 else 1

def genVectors(n, MC=True):
    vects = np.zeros((n, 3))
    for i in range(n):
        phi = np.random.uniform(0, np.pi / 2)
        theta = np.arccos(1 - 2 * np.random.uniform())
        vect = np.array([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)])
        if MC and np.cos(theta) > 0:
            vect = -vect
        vects[i] = vect
    return vects

def MCc(vecs):
    return 2*np.pi / np.mean([V(vec) * vec[-1] for vec in vecs])

def ISc(vecs):
    return np.pi / np.mean([V(vec) for vec in vecs])

S1 = [MCc(genVectors(n)) for i in range(N)]
S2 = [ISc(genVectors(n, False)) for i in range(N)]


print(f'MC: {np.mean(S1)} {np.std(S1)}')
print(f'IS: {np.mean(S2)} {np.std(S2)}')
