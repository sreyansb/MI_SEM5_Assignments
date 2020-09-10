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

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []
    n=len(cost)
    def dfs(cur_vertex,cur_goal,visited):
        if cur_vertex in visited:
            return []
        if cur_vertex==cur_goal:
            return visited+[cur_goal]
        visited.append(cur_vertex)
        for i in range(1,n):
            if i not in visited and cost[cur_vertex][i]!=-1:
                k=dfs(i,cur_goal,visited)
                if k:
                    return k
                visited.pop()
        return []
                
    t1=[]
    for i in goals:
        k=dfs(start_point,i,[])
        t1.append(k)

    # t1 <= DFS_Traversal
    # t2 <= UCS_Traversal
    # t3 <= A_star_Traversal
    t2=[]
    t3=[]
    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

cost=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
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
heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
print(tri_traversal(cost,heuristic,10,[1]))
