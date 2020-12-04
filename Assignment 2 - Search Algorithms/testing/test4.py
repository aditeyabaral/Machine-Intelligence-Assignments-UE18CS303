from Assignment2 import UCS_Traversal
cost = [[0, 0, 0, 0], 
[0, 0, 5, 10], 
[0, 5, 0, 5],
[0, 10, 5, 0]]

goal = [3]
start = 1

l = UCS_Traversal(cost, start, goal)
print(l)