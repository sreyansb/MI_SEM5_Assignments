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
        if visited:
            visited.pop()
    l=dfs(start_point,[])
    if not(l):
        l=[]
    return l

def UCS_Traversal(cost, start_point, goals):
	frontier = []
	explore = []
	e_nodes = [] 
	frontier.append([0,start_point,[1]])
	explored_goals = []
	goal_path = []
	def ucs(start_point):	
		if (start_point in goals) and (start_point not in explored_goals):
			explored_goals.append(start_point)
			goal_path.append(frontier[0][2])

		if start_point not in e_nodes:
			for i in range(1,len(cost)):
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
	if len(goal_path)>0:
		return goal_path[0]
	else:
		return goal_path


def constructPath(came_from,current):
    path = []
    while current in came_from:
        path.insert(0,current)
        #print(path)
        current = came_from[current] 
    return [current] + path

def getNeighbours(cost,q):
    neighbours = []
    for i in range(1,len(cost[q])):
        if cost[q][i]>-1 and i!=q:
            neighbours.append(i)
    return neighbours

    
def getCost(path,cost):
    total_cost=0
    for i in range(0,len(path)-1):
        c = cost[path[i]][path[i+1]]
        total_cost += c
    #print('cost of ',path,total_cost)    
    return total_cost

def getMin(open_list,f_score):
    sorted_list = sorted(open_list)
    minimum_node = open_list[0]
    minimum_f = f_score[minimum_node]
    for i in range(1,len(sorted_list)):
        if(f_score[sorted_list[i]]<minimum_f):
            minimum_node = open_list[i]
            minimum_f = f_score[sorted_list[i]]
    return minimum_node

def A_star_Traversal_For_One_Goal(cost, heuristic, start_point, goal):
    if start_point==goal:
        return [start_point]
    l = []
    #print('In astar')
    g_score={i:float('inf') for i in range(1,len(cost))}
    g_score[start_point]=0
    
    f_score={i:float('inf') for i in range(1,len(cost))}
    f_score[start_point]=heuristic[start_point]
    
    open_set = [start_point]
    
    current = start_point 
    came_from = {}

    
    while len(open_set)>0:
        #print(f_score)
        q = getMin(open_set,f_score)
        if q == goal:
            l = constructPath(came_from,q)
            #print(l + [goal])
            return l
    #print('open_set',open_set)
        #print('came from',came_from)
        #print('q:'+str(q))
        open_set.remove(q)
        q_neighbours = getNeighbours(cost,q)
        #print('Neighbours of ' + str(q) + ':',q_neighbours)
        for neighbour in q_neighbours:
            neighbour_g = g_score[q] + cost[q][neighbour]
            neighbour_f = neighbour_g + heuristic[neighbour]
            if neighbour_f < f_score[neighbour]:
                came_from[neighbour] = q
                g_score[neighbour] = neighbour_g
                f_score[neighbour] = neighbour_g + heuristic[neighbour]
                if neighbour not in open_set:
                    open_set.append(neighbour) 
   
    return []


def A_star_Traversal(cost, heuristic, start_point, goals):
    min_cost = float('inf')
    min_path = []
    for i in range(len(goals)):
        path = A_star_Traversal_For_One_Goal(cost,heuristic,start_point,goals[i])
        #print(path)
        if path==[]:
            continue
        #print(min_path)
        path_cost = getCost(path,cost)
        if path_cost<min_cost:
            min_cost=path_cost
            min_path=path
    return min_path



def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost,start_point,goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals)

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

