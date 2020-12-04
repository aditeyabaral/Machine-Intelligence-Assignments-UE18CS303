from Assignment2 import *

cost=[[0,0,0,0,0,0],[0,0,6,-1,-1,-1,3],[0,6,0,3,3,-1],[0,-1,3,0,1,7],[0,-1,3,1,0,8],[0,-1,-1,7,8,0]]
heuristic=[0,10,8,7,7,3]
goals=[5]
ele=tri_traversal(cost, heuristic,1, goals)

if ele[2]==[1,2,3,5]:
    print("passed A* and A* was",ele[2])
else:
    print("A* FAILED and ur A* was ",ele[2],"but [1,2,3,5] was expected")
    
