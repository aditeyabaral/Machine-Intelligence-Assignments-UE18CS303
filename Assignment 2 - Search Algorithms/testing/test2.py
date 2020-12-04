from Assignment2 import UCS_Traversal
cost=[[0 if i==j else -1 for j in range(9)] for i in range(9)]
cost[1][3]=3
cost[1][2]=7
cost[3][4]=9
cost[2][5]=6
cost[2][3]=2
cost[4][5]=3
cost[4][8]=13
cost[5][6]=2
cost[5][7]=1
for j in range(9):
    for k in range(9):
        if cost[j][k]>0:
            cost[k][j]=cost[j][k]
heuristic=[0 for i in range(9)]
goals = [8]

l=  UCS_Traversal(cost, 1, goals)
print(l)


