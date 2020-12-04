from Assignment2 import tri_traversal
cost=[[-1 for j in range(6)] for i in range(6)]
goals=[5]
cost[1][2]=1
cost[1][3]=4
cost[2][3]=2
cost[2][4]=5
cost[2][5]=12
cost[3][4]=2
cost[4][5]=3
heuristic=[0,7,6,2,1,0]
ele=tri_traversal(cost, heuristic, 1, goals)
'''
if ele[1]==[1,2,3,4,5]:
    print("ucs testcase passed")
else:
    print("ucs testcase FAILED")
'''
if ele[2]==[1,2,4,5]:
    print("A* passed testcase")
else:
    print("A* FAILED testcase",ele[2])