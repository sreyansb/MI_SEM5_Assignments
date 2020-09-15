def DFS_Traversal(cost,start_point,goals):
    l = []
    n=len(cost)
    def dfs(cur_vertex,visited):
        if cur_vertex in visited:
            return []
        if cur_vertex in goals:
            return visited+[cur_vertex]
        visited.append(cur_vertex)
        for i in range(1,n):
            if i not in visited and cost[cur_vertex][i]!=-1:
                k=dfs(i,visited)
                if k:
                    return k
                visited.pop()
        if visited:
            visited.pop()
    l=dfs(start_point,[])
    return l

def UCS_Traversal():
	frontier = []
	explore = []
	e_nodes = [] 
	frontier.append([0,start_point,[1]])
	explored_goals = []
	goal_path = []
	#goal_dist = []
	def ucs(start_point):	
		if (start_point in goals) and (start_point not in explored_goals):
			explored_goals.append(start_point)
			goal_path.append(frontier[0][2])
			#goal_dist.append(frontier[0][0])

		if start_point not in e_nodes:
			for i in range(1,11):
				if (cost[start_point][i] != -1) and (cost[start_point][i] != 0):
					ele_cost = cost[start_point][i] + frontier[0][0]
					dist = []
					for j in frontier[0][2]:
						dist.append(j)
					dist.append(i)
					frontier.append([ele_cost,i,dist])
			explore.append(frontier[0])
			e_nodes.append(frontier[0][1])

		frontier.pop(0)
		frontier.sort()
		if len(frontier) == 0:
			return
		ucs(frontier[0][1])
	ucs(start_point)
	#return goal_path[0]
	#return goal_path
	l = goal_path[0]
	return l

def A_star_Traversal():
    l = []
    return l

def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost,start_point,goals)
    t2 = UCS_Traversal()
    t3 = A_star_Traversal()

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l





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

