#!/usr/bin/python3
import json
import matplotlib.pyplot as plt

data = json.load(open('data.json', 'r'))
keys = list(data.keys())
choice = 0
while True:
    choice = int(input(f'choose from {keys}: '))
    if choice < 0 or choice >= len(keys):
        break

    a = data[keys[choice]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(a['x'], a['y'], a['z'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('W')
    plt.show()
