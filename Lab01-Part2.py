import numpy as np
import pandas as pd

fread = pd.read_csv('ulysses16.csv')


def calc_distance(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


cities = len(fread)
matrix = np.zeros((cities, cities))

for i in range(cities):
    for j in range(cities):
        if i != j:
            city_i = (fread.iloc[i]['x'], fread.iloc[i]['y'])
            city_j = (fread.iloc[j]['x'], fread.iloc[j]['y'])

            matrix[i][j] = calc_distance(city_i, city_j)

print(matrix)