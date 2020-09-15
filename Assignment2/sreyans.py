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
                k=dfs(i,cur_goal,visited)
                if k:
                    return k
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
                    parents[i]=k[1]
        return parents

    def astar(child):
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
        return parents
                
    #dfs call
    l = []     
    t1=[]
    t1=dfs(cost,start_point,goals)
    
        
    #ucs call
    t2=[]
    for i in goals:
        k=ucs(i)
        ans=[]
        j=i
        while(j!=start_point):
            ans.append(j)
            j=k[j]
        ans.append(start_point)
        ans.reverse()
        t2.append(ans)
    print(t2)
    
            

    # t1 <= DFS_Traversal
    # t2 <= UCS_Traversal
    # t3 <= A_star_Traversal
    t3=[]
    for i in goals:
        k=astar(i)
        ans=[]
        j=i
        while(j!=start_point):
            ans.append(j)
            j=k[j]
        ans.append(start_point)
        ans.reverse()
        t3.append(ans)
    print("t3",t3)
    
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

print(tri_traversal(cost,heuristic,1,[6,7,10]))
