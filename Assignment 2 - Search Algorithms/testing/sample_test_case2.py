import numpy as np
from Assignment2 import UCS_Traversal

number_to_city = {1:"Oradea", 2:"Zerind", 3:"Arad", 4:"Timisoara", 5:"Lugoj", 6:"Mehadia", 7:"Drobeta", 8:"Sibiu", 
    9:"Rimnicu Vilcea", 10:"Craiova", 11:"Fagaras", 12:"Pitesti", 13:"Bucharest", 14:"Giurgiu", 15:"Neamt", 16:"Iasi",
    17:"Vaslui", 18:"Urziceni", 19:"Hirsova", 20:"Eforie"}

cost = []
for i in range(21):
    cost.append([0]*21)
cost[1][2] = 71
cost[1][8] = 151

cost[2][1] = 71
cost[2][3] = 75

cost[3][2] = 75
cost[3][8] = 140
cost[3][4] = 118

cost[4][3] = 118
cost[4][5] = 111

cost[5][4] = 111
cost[5][6] = 70

cost[6][5] = 70
cost[6][7] = 75

cost[7][6] = 76
cost[7][10] = 120

cost[8][1] = 151
cost[8][3] = 140
cost[8][9] = 80
cost[8][11] = 99

cost[9][8] = 80
cost[9][10] = 146
cost[9][12] = 97

cost[10][7] = 120
cost[10][9] = 146
cost[10][12] = 138

cost[11][8] = 99
cost[11][13] = 211

cost[12][9] = 97
cost[12][10] = 138
cost[12][13] = 101

cost[13][11] = 211
cost[13][12] = 101
cost[13][14] = 90
cost[13][18] = 85

cost[14][13] = 90

cost[15][16] = 87

cost[16][15] = 87
cost[16][17] = 92

cost[17][16] = 92
cost[17][18] = 142

cost[18][13] = 85
cost[18][17] = 142
cost[18][19] = 98

cost[19][18] = 98
cost[19][20] = 86

cost[20][19] = 86


def test(start_point, goals, result):
    l = UCS_Traversal(cost, start_point, goals)
    path = [number_to_city[v] for v in l]
    print("->".join(path))
    return l==result

sp = 8
goals = [13] 
if test(sp, goals, [8,9,12,13]):
    print("Test 1 passed!")
else:
    print("Test 1 fail!")
sp = 1
goals = [7, 11, 10]
if test(sp, goals, [1, 8, 11]):
    print("Test 2 passed!")
else:
    print("Test 2 fail!")

sp = 1
goals = [10, 13, 20]
if test(sp, goals, [1, 8, 9, 10]):
    print("Test 3 passed!")
else:
    print("Test 3 fail!")