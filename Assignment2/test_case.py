from Assignment2 import *


def test_case():
    cost = [[0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 3, -1, -1],
            [0, -1, 0, -1, 3, -1], 
            [0, -1, -1, 0, -1, 4],
            [0, -1, -1, -1, 0, -1],
            [0, -1, -1, -1, -1, 0]]
    heuristic = [0, 5, 1, 3, 0, 0]

    try:
        # start node is 1, goal states to reach are 6,7 & 10
        if (tri_traversal(cost,heuristic, 1, [4, 5]))[0] == [1, 2, 4]:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1 FOR THE DFS_TRAVERSAL FAILED")
    try:
        if (tri_traversal(cost,heuristic, 1, [4, 5]))[1] == [1, 3, 5]:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2 FOR THE UCS_TRAVERSAL FAILED")

    try:
        if (tri_traversal(cost,heuristic, 1, [4, 5]))[2] == [1, 3, 5]:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 3 FOR THE A_star_TRAVERSAL FAILED")


test_case()
