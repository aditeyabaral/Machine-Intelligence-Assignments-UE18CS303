def search_q(frontier, path):
    for index in range(len(frontier)):
        if frontier[index][1] == path:
            return index


def A_star_Traversal(cost, start_point, goals, heuristic):
    explored = []
    path = [start_point]
    frontier = [[0+heuristic[start_point], path]]
    while len(frontier) > 0:
        curr_cost, curr_path = frontier.pop(0)
        n = curr_path[-1]
        curr_cost -= heuristic[n]
        if n in goals:
            return curr_path
        explored.append(n)
        children = [i for i in range(len(cost[0]))
                    if cost[n][i] not in [0, -1]]
        for i in children:
            new_curr_path = curr_path + [i]
            new_path_cost = curr_cost + cost[n][i] + heuristic[i]
            if i not in explored and new_curr_path not in [i[1] for i in frontier]:
                frontier.append((new_path_cost, new_curr_path))
                frontier = sorted(frontier, key=lambda x: (x[0], x[1]))
            elif new_curr_path in [i[1] for i in frontier]:
                index = search_q(frontier, new_curr_path)
                frontier[index][0] = min(frontier[index][0], new_path_cost)
                frontier = sorted(frontier, key=lambda x: (x[0], x[1]))
    return list()


def UCS_Traversal(cost, start_point, goals):
    explored = []
    path = [start_point]
    frontier = [[0, path]]
    while len(frontier) > 0:
        curr_cost, curr_path = frontier.pop(0)
        n = curr_path[-1]
        if n in goals:
            return curr_path
        explored.append(n)
        children = [i for i in range(len(cost[0]))
                    if cost[n][i] not in [0, -1]]
        for i in children:
            new_curr_path = curr_path + [i]
            new_path_cost = curr_cost + cost[n][i]
            if i not in explored and new_curr_path not in [i[1] for i in frontier]:
                frontier.append((new_path_cost, new_curr_path))
                frontier = sorted(frontier, key=lambda x: (x[0], x[1]))
            elif new_curr_path in [i[1] for i in frontier]:
                index = search_q(frontier, new_curr_path)
                frontier[index][0] = min(frontier[index][0], new_path_cost)
                frontier = sorted(frontier, key=lambda x: (x[0], x[1]))
    return list()


def DFS_Traversal(cost, start_point, t1, visited, goals):
    visited[start_point] = True
    t1.append(start_point)
    if (start_point not in goals):
        temp = cost[start_point]
        for i in range(len(temp)):
            if ((visited[i] == False) and (temp[i] > 0)):
                r = DFS_Traversal(cost, i, t1, visited, goals)
                if r == -1:
                    return -1
                t1.pop()
    else:
        return -1


'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''


def tri_traversal(cost, heuristic, start_point, goals):
    l = []
    visited = [False]*(len(cost[0]))
    t1 = []
    DFS_Traversal(cost, start_point, t1, visited, goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, start_point, goals, heuristic)
    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
