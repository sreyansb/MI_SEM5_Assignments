"""
Rakshith's solution on UCS: according to definition
def UCS_Traversal(cost, start_point, goals):
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
	#return goal_path[0]
	#return goal_path
	return goal_path[0]
"""

"""
#Krishna's code for Astar
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


def A_star_Traversal_For_One_Goal(cost, heuristic, start_point, goal):
    
    l = []
    #print('In astar')
    g_score={i:float('inf') for i in range(1,len(cost))}
    g_score[start_point]=0
    
    f_score={i:float('inf') for i in range(1,len(cost))}
    f_score[start_point]=heuristic[start_point]
    
    open_set = {start_point}
    
    current = start_point 
    came_from = {}

    
    while len(open_set)>0:
        q = min(open_set,key = lambda x:f_score[x])
        #print('open_set',open_set)
        #print('came from',came_from)
        #print('q:'+str(q))
        open_set.remove(q)
        q_neighbours = getNeighbours(cost,q)
        #print('Neighbours of ' + str(q) + ':',q_neighbours)
        for neighbour in q_neighbours:
            if neighbour == goal:
                l = constructPath(came_from,q)
                #print(l + [goal])
                return l + [neighbour]
            neighbour_g = g_score[q] + cost[q][neighbour]
            if neighbour_g < g_score[neighbour]:
                came_from[neighbour] = q
                g_score[neighbour] = neighbour_g
                f_score[neighbour] = neighbour_g + heuristic[neighbour]
                if neighbour not in open_set:
                    open_set.add(neighbour) 
   
    return []


def A_star_Traversal(cost, heuristic, start_point, goals):
    min_cost = float('inf')
    for i in range(len(goals)):
        path = A_star_Traversal_For_One_Goal(cost,heuristic,start_point,goals[i])
        if path==[]:
            continue
        #print(min_path)
        path_cost = getCost(path,cost)
        if path_cost<min_cost:
            min_cost=path_cost
            min_path=path
    return min_path
"""

from heapq import heappush,heappop

def tri_traversal(cost, heuristic, start_point, goals):
    n=len(cost)
    
    def dfs(cur_vertex,visited):# returns the goal state which comes first
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
                #visited.pop()
        if visited:
            visited.pop()
                       
    def ucs(child):
        costs=[float('inf') for i in range(n)]
        costs[start_point]=0
        vertheap=[]
        heappush(vertheap,(0,start_point))
        parents=[-1 for _ in range(n)]
        visited=set()
        while(vertheap):
            k=heappop(vertheap)
            if k[1] in visited:
                continue
            visited.add(k[1])
            for i in range(1,n):
                if cost[k[1]][i]!=-1 and i not in visited and cost[k[1]][i]+k[0]<costs[i]:
                    costs[i]=cost[k[1]][i]+k[0]
                    heappush(vertheap,(costs[i],i))
                    print(vertheap)
                    parents[i]=k[1]
        return (parents,costs[child])

    def astar(child):#expansion uses heuristic and heuristic is not used anywhere else
        costs=[float('inf') for i in range(n)]
        costs[start_point]=0
        vertheap=[]
        heappush(vertheap,(heuristic[start_point],start_point))
        parents=[-1 for _ in range(n)]
        visited=set()
        while(vertheap):
            k=heappop(vertheap)
            if k[1] in visited:
                continue
            visited.add(k[1])
            for i in range(1,n):
                if cost[k[1]][i]!=-1 and cost[k[1]][i]+heuristic[i]+k[0]<costs[i]:
                    costs[i]=cost[k[1]][i]+k[0]
                    heappush(vertheap,(costs[i],i))
                    parents[i]=k[1]
        return (parents,costs[child])
                
    #dfs call
    l = []     
    t1=[]
    t1=dfs(start_point,[])
    if not(t1):
        t1=[]
      
    #ucs call
    t2=[]
    mincost=float('inf')
    mini=[]
    for i in goals:
        k=ucs(i)
        if mincost>k[1]:
            ans=[]
            j=i
            while(j!=start_point):
                ans.append(j)
                j=k[0][j]
            ans.append(start_point)
            ans.reverse()
            mini=ans.copy()
            mincost=k[1]
    t2.append(mini)
    print("t2",t2)
    
    #astar call
    t3=[]
    mini=[]
    mincost=float('inf')
    for i in goals:
        k=astar(i)
        ans=[]
        j=i
        if mincost>k[1]:
            while(j!=start_point):
                ans.append(j)
                j=k[0][j]
            ans.append(start_point)
            ans.reverse()
            mini=ans
            mincost=k[1]
    t3.append(mini)
    print("t3",t3)
    
    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
'''
cost=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, 1],
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
cost=[[0,0,0,0,0,0,0],[0,0,7,-1,-1,-1,5],[0,-1,0,3,-1,-1,-1],[0,-1,-1,0,2,-1,-1],\
      [0,-1,-1,-1,0,1,-1],[0,-1,-1,-1,-1,-1,-1],[0,-1,-1,-1,-1,100,0]]
print(tri_traversal(cost,heuristic,1,[6,7,10]))
print(A_star_Traversal(cost, heuristic, 1, [6,7,10]))
'''
