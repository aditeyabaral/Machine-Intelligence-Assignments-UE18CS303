from Assignment2 import *

cost1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
         [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1],
         [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
         [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
         [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
         [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
         [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
         [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
         [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
         [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]

cost2 = [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 3, -1, -1, -1, -1, 2],
         [0, -1, 0, 5, 10, -1, -1, -1],
         [0, -1, -1, 0, 2, -1, 1, -1],
         [0, -1, -1, -1, 0, 4, -1, -1],
         [0, -1, -1, -1, -1, 0, -1, -1],
         [0, -1, -1, -1, -1, 3, 0, -1],
         [0, -1, -1, 1, -1, -1, 4, 0]]  # https://www.geeksforgeeks.org/search-algorithms-in-ai/

cost3 = [[0, 0, 0, 0, 0, 0],
         [0, 0, 6, -1, -1, -1],
         [0, 6, 0, 3, 3, -1],
         [0, -1, 3, 0, 1, 7],
         [0, -1, 3, 1, 0, 8],
         [0, -1, -1, 7, 8, 0],
         ]
heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
heuristic2 = [0, 7, 9, 4, 2, 0, 3, 5]
heuristic3 = [0, 10, 8, 7, 7, 3]

astar = A_star_Traversal
dfs = DFS_Traversal
ucs = UCS_Traversal


def dfstest():
    print("---------------------------------- TESTS FOR DFS SEARCH----------------------------------------")
    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [1])
    print("Test 1: ", t1 == [1])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [2])
    print("Test 2: ", t1 == [1, 2])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [3])
    print("Test 3: ", t1 == [1, 2, 3])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [4])
    print("Test 4: ", t1 == [1, 2, 3, 4])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [5])
    print("Test 5: ", t1 == [1, 2, 3, 4, 8, 5])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [6])
    print("Test 6: ", t1 == [1, 2, 6])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [7])
    print("Test 7: ", t1 == [1, 2, 3, 4, 7])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [8])
    print("Test 8: ", t1 == [1, 2, 3, 4, 8])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [9])
    print("Test 9: ", t1 == [1, 2, 3, 4, 8, 5, 9])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [10])
    print("Test 10: ", t1 == [1, 2, 3, 4, 8, 5, 9, 10])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [6, 7, 10])
    print("Test 11: ", t1 == [1, 2, 3, 4, 7])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [3, 4, 7, 10])
    print("Test 12: ", t1 == [1, 2, 3])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [5, 9, 4])
    print("Test 13: ", t1 == [1, 2, 3, 4])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [4, 8, 10])
    print("Test 14: ", t1 == [1, 2, 3, 4])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [2, 8, 5])
    print("Test 15: ", t1 == [1, 2])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [7, 9, 10])
    print("Test 16: ", t1 == [1, 2, 3, 4, 7])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [10, 6, 8, 4])
    print("Test 17: ", t1 == [1, 2, 3, 4])

    visited = [False]*(len(cost1[0]))
    t1 = []
    dfs(cost1, 1, t1, visited, [9, 7, 5, 10])
    print("Test 18: ", t1 == [1, 2, 3, 4, 7])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [1])
    print("Test 19: ", t1 == [1])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [2])
    print("Test 20: ", t1 == [1, 2])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [3])
    print("Test 21: ", t1 == [1, 2, 3])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [4])
    print("Test 22: ", t1 == [1, 2, 3, 4])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [5])
    print("Test 23: ", t1 == [1, 2, 3, 4, 5])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [6])
    print("Test 24: ", t1 == [1, 2, 3, 6])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [7])
    print("Test 25: ", t1 == [1, 7])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [4, 5, 6])
    print("Test 26: ", t1 == [1, 2, 3, 4])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [3, 6, 7])
    print("Test 27: ", t1 == [1, 2, 3])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [4, 6])
    print("Test 28: ", t1 == [1, 2, 3, 4])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 1, t1, visited, [2, 3, 7])
    print("Test 29: ", t1 == [1, 2])

    visited = [False]*(len(cost2[0]))
    t1 = []
    dfs(cost2, 4, t1, visited, [3])
    print("Test 30 [CAN BE IGNORED - NO POSSIBLE PATH TEST]: ", t1 == [])

    visited = [False]*(len(cost3[0]))
    t1 = []
    dfs(cost3, 1, t1, visited, [5])
    print("Test 31: ", t1 == [1, 2, 3, 4, 5])


def ucstest():
    print("----------------------------------TESTS FOR UCS SEARCH----------------------------------------")
    print("Test 1: ", ucs(cost1, 1, [1]) == [1])
    print("Test 2: ", ucs(cost1, 1, [2]) == [1, 2])
    print("Test 3: ", ucs(cost1, 1, [3]) == [1, 2, 3])
    print("Test 4: ", ucs(cost1, 1, [4]) == [1, 5, 4])
    print("Test 5: ", ucs(cost1, 1, [5]) == [1, 5])
    print("Test 6: ", ucs(cost1, 1, [6]) == [1, 2, 6])
    print("Test 7: ", ucs(cost1, 1, [7]) == [1, 5, 4, 7])
    print("Test 8: ", ucs(cost1, 1, [8]) == [1, 5, 4, 8])
    print("Test 9: ", ucs(cost1, 1, [9]) == [1, 5, 9])
    print("Test 10: ", ucs(cost1, 1, [10]) == [1, 5, 9, 10])
    print("Test 11: ", ucs(cost1, 1, [6, 7, 10]) == [1, 5, 4, 7])
    print("Test 12: ", ucs(cost1, 1, [3, 4, 7, 10]) == [1, 2, 3])
    print("Test 13: ", ucs(cost1, 1, [5, 9, 4]) == [1, 5])
    print("Test 14: ", ucs(cost1, 1, [4, 8, 10]) == [1, 5, 4])
    print("Test 15: ", ucs(cost1, 1, [2, 8, 5]) == [1, 2])
    print("Test 16: ", ucs(cost1, 1, [7, 9, 10]) == [1, 5, 9])
    print("Test 17: ", ucs(cost1, 1, [10, 6, 8, 4]) == [1, 5, 4])
    print("Test 18: ", ucs(cost1, 1, [9, 7, 5, 10]) == [1, 5])


def astartest():
    print("----------------------------------TESTS FOR A* SEARCH----------------------------------------")
    print("Test 1: ", astar(cost1, 1, [1], heuristic1) == [1])
    print("Test 2: ", astar(cost1, 1, [2], heuristic1) == [1, 2])
    print("Test 3: ", astar(cost1, 1, [3], heuristic1) == [1, 2, 3])
    print("Test 4: ", astar(cost1, 1, [4], heuristic1) == [1, 5, 4])
    print("Test 5: ", astar(cost1, 1, [5], heuristic1) == [1, 5])
    print("Test 6: ", astar(cost1, 1, [6], heuristic1) == [1, 2, 6])
    print("Test 7: ", astar(cost1, 1, [7], heuristic1) == [1, 5, 4, 7])
    print("Test 8: ", astar(cost1, 1, [8], heuristic1) == [1, 5, 4, 8])
    print("Test 9: ", astar(cost1, 1, [9], heuristic1) == [1, 5, 9])
    print("Test 10: ", astar(cost1, 1, [10], heuristic1) == [1, 5, 9, 10])
    print("Test 11: ", astar(cost1, 1, [6, 7, 10], heuristic1) == [1, 5, 4, 7])
    print("Test 12: ", astar(cost1, 1, [3, 4, 7, 10], heuristic1) == [1, 2, 3])
    print("Test 13: ", astar(cost1, 1, [5, 9, 4], heuristic1) == [1, 5])
    print("Test 14: ", astar(cost1, 1, [4, 8, 10], heuristic1) == [1, 5, 4])
    print("Test 15: ", astar(cost1, 1, [2, 8, 5], heuristic1) == [1, 2])
    print("Test 16: ", astar(cost1, 1, [7, 9, 10], heuristic1) == [1, 5, 4, 7])
    print("Test 17: ", astar(cost1, 1, [10, 6, 8, 4], heuristic1) == [1, 5, 4])
    print("Test 18: ", astar(cost1, 1, [9, 7, 5, 10], heuristic1) == [1, 5])
    print("Test 19: ", astar(cost2, 1, [1], heuristic2) == [1])
    print("Test 20: ", astar(cost2, 1, [2], heuristic2) == [1, 2])
    print("Test 21: ", astar(cost2, 1, [3], heuristic2) == [1, 7, 3])
    print("Test 22: ", astar(cost2, 1, [4], heuristic2) == [1, 7, 3, 4])
    print("Test 23: ", astar(cost2, 1, [5], heuristic2) == [1, 7, 3, 6, 5])
    print("Test 24: ", astar(cost2, 1, [6], heuristic2) == [1, 7, 3, 6])
    print("Test 25: ", astar(cost2, 1, [7], heuristic2) == [1, 7])
    print("Test 26: ", astar(cost2, 1, [4, 5, 6], heuristic2) == [1, 7, 3, 4])
    print("Test 27: ", astar(cost2, 1, [3, 6, 7], heuristic2) == [1, 7])
    print("Test 28: ", astar(cost2, 1, [4, 6], heuristic2) == [1, 7, 3, 4])
    print("Test 29: ", astar(cost2, 1, [2, 3, 7], heuristic2) == [1, 7])
    print("Test 30 [CAN BE IGNORED - NO POSSIBLE PATH TEST]: ",
          astar(cost2, 4, [3], heuristic2) == [])
    print("Test 31: ", astar(cost3, 1, [5], heuristic3) == [1, 2, 3, 5])


astartest()
ucstest()
dfstest()
