# # BFS

from collections import deque
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

def BFS(start):
  queue = []
  visited = set()
  queue.append(start)
  visited.add(start)

  while queue:
    node = queue.pop(0)
    print(node, end="->")
    for neighbor in graph[node]:
      if neighbor not in visited:
        queue.append(neighbor)
        visited.add(neighbor)

BFS('A')


# # DLS

graph = {
    'A': {'B': 10, 'C': 15},
    'B': {'D': 12},
    'C': {'E': 10},
    'D': {'F': 5},
    'E': {'G': 7},
    'F': {},
    'G': {}
}

def DLS(graph, start, target, depth, depth_limit):
    print(start, end=" ")
    if start == target:
        return True
    if depth >= depth_limit:
        return False
    for neighbour, _ in graph[start].items():
        if DLS(graph, neighbour, target, depth+1, depth_limit):
            return True
    return False

DLS(graph, "A", "G", 0, 3)


# # Best First Search

import heapq

graph = {
    'A': {'B': 10, 'C': 15},
    'B': {'D': 12},
    'C': {'E': 10},
    'D': {'F': 5},
    'E': {'G': 7},
    'F': {},
    'G': {}
}
heuristics = {'A': 78, 'B': 40, 'C': 30, 'D': 25, 'E': 20, 'F': 10, 'G': 0}

def Best_First_Search(graph, start, goal, heuristic):
  visited = set()
  heap = [(0, start)]
  path = []
  while heap:
    cost, node = heapq.heappop(heap)
    path.append(node)
    if node == goal:
      return path
    visited.add(node)

    for neighbour, neighbour_cost in graph[node].items():
      if neighbour not in visited:
        heapq.heappush(heap, (heuristic[neighbour], neighbour))

path = Best_First_Search(graph, "A", "G", heuristics)
print(path)


# # A Star

import heapq

graph = {
    'A': {'B': 10, 'C': 15},
    'B': {'D': 12},
    'C': {'E': 10},
    'D': {'F': 5},
    'E': {'G': 7},
    'F': {},
    'G': {}
}
heuristics = {'A': 78, 'B': 40, 'C': 30, 'D': 25, 'E': 20, 'F': 10, 'G': 0}

def Astar(start, target):
  visited = set()
  open_list = [(0, start, [start])]

  while open_list:
    cost, node, path = heapq.heappop(open_list)

    if node == target:
      return cost, path

    visited.add(node)
    for neighbour, neighbour_cost in graph[node].items():
      if neighbour not in visited:
        heuristic_cost = heuristics[neighbour]
        new_cost = cost + neighbour_cost
        total_new_cost = new_cost + heuristic_cost
        new_path = path + [neighbour]
        heapq.heappush(open_list, (total_new_cost, neighbour, new_path))

print(Astar("A", "G"))

#8 puzzle

from functools import cmp_to_key


def key(a, b):
    if a[0] > b[0]:
        return -1

    return 1


key1 = cmp_to_key(key)

last_move = []
cl = {}


def take_input(prompt):
    print(f"\nEnter the 3x3 {prompt}")
    print("\ndenote - by 0")
    state = []
    row = []
    for i in range(3):
        for j in range(3):
            row.append(int(input(f"enter state - {i}{j} ")))
        state.append(row)
        row = []

    return state


def print_in_format(state):
    print("\n\n")
    for row in state:
        for j in row:
            if j == 0:
                print('-', sep=' ', end=' ')
            else:
                print(j, sep=' ', end=' ')
        print()


def get_dash_loc(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return (i, j)


def get_movable_places(curr_dash):
    movable = []

    # change the i pos
    i = curr_dash[0]
    i -= 1
    if i > -1:
        movable.append((i, curr_dash[1]))
    i = curr_dash[0]
    i += 1
    if i < 3:
        movable.append((i, curr_dash[1]))

    # change the j pos
    j = curr_dash[1]
    j -= 1
    if j > -1:
        movable.append((curr_dash[0], j))

    j = curr_dash[1]
    j += 1
    if j < 3:
        movable.append((curr_dash[0], j))

    return movable


def cal_fn(state, place, gn):
    miss_placed = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != goal_state[i][j]:
                miss_placed += 1
    cl[miss_placed + gn] = place


def generate_states(places_to_move, gn, dash_loc):
    # state_possibility = []
    curr_state
    dash_loc_x = dash_loc[0]
    dash_loc_y = dash_loc[1]

    for place in places_to_move:
        if place not in last_move:
            move_x = place[0]
            move_y = place[1]
            curr_state[dash_loc_x][dash_loc_y], curr_state[move_x][move_y] = curr_state[move_x][move_y], curr_state[dash_loc_x][dash_loc_y]
            cal_fn(curr_state, place, gn)
            curr_state[dash_loc_x][dash_loc_y], curr_state[move_x][move_y] = curr_state[move_x][move_y], curr_state[dash_loc_x][dash_loc_y]
        # last_move.append(place)
    last_move.clear()


def puzzle_solver(curr_dash_loc):
    key = -2
    gn = 1
    updated_dash_loc = curr_dash_loc
    i = 1
    while (key - gn) != 0:
        places_to_move = get_movable_places(updated_dash_loc)
        generate_states(places_to_move, gn, updated_dash_loc)
        key = sorted(list(cl.keys()))[0]
        fn = cl[key]
        last_move.append(get_dash_loc(curr_state))
        curr_state[updated_dash_loc[0]][updated_dash_loc[1]], curr_state[fn[0]][fn[1]
                                                                                ] = curr_state[fn[0]][fn[1]], curr_state[updated_dash_loc[0]][updated_dash_loc[1]]
        updated_dash_loc = fn
        print_in_format(curr_state)
        print(f"\n{key}")
        cl.clear()
        i += 1
    print(f"\n i = {i}")


# curr_state = take_input("Initial state")
# goal_state = take_input("goal state")
curr_state = [[1, 2, 3], [0, 5, 6], [4, 7, 8]]
goal_state =[[1, 2, 3], [4, 5, 6], [7, 8, 0]]


print_in_format(curr_state)
curr_dash_loc = get_dash_loc(curr_state)
last_move.append(curr_dash_loc)
print("\n")

puzzle_solver(curr_dash_loc)


#n-queen

def is_safe(row, col, board, n):
  r, c = row, col
  while r >= 0:
    if board[r][c] == 1:
      return False
    r = r - 1

  r, c = row, col
  while r >= 0 and c >= 0:
    if board[r][c] == 1:
      return False
    r = r - 1
    c = c - 1

  r, c = row, col
  while r >= 0 and c < n:
    if board[r][c] == 1:
      return False
    r = r - 1
    c = c + 1

  return True

def chess_board(r, n, board):
  if r >= n:
    print_solution(board)
    return False

  for c in range(n):
    if is_safe(r, c, board, n):
      board[r][c] = 1
      chess_board(r+1, n, board)
      board[r][c] = 0

  return False

def print_solution(board):
  global solutions
  solutions += 1
  print(f"Solution {solutions}: ")
  for i in range(len(board)):
    for j in range(len(board)):
      if board[i][j] == 1:
        print("Q", end=" ")
      else:
        print(".", end=" ")
    print()
  print()

if __name__ == "__main__":
  n = int(input("Enter length of chess board or number of queens :- "))
  board = [[0 for _ in range(n)] for _ in range(n)]
  solutions = 0

  if not chess_board(0, n, board):
    print("No solution is found")


# crypt arithematic

import itertools

def get_value(word, substitution):
    s = 0
    factor = 1
    for letter in reversed(word):
        s += factor * substitution[letter]
        factor *= 10
    return s


def solve2(equation):
    # split equation in left and right
    left, right = equation.lower().replace(' ', '').split('=')
    # split words in left part
    left = left.split('+')
    # create list of used letters
    letters = set(right)
    for word in left:
        for letter in word:
            letters.add(letter)
    letters = list(letters)

    digits = range(10)
    for perm in itertools.permutations(digits, len(letters)):
        sol = dict(zip(letters, perm))

        if sum(get_value(word, sol) for word in left) == get_value(right, sol):
            print(' + '.join(str(get_value(word, sol)) for word in left) + " = {} (mapping: {})".format(get_value(right, sol), sol))

if __name__ == '__main__':
    solve2('TWO+TWO=FOUR')


# graph coloring

# Adjacent Matrix
G = [[ 0, 1, 1, 0, 1, 0],
     [ 1, 0, 1, 1, 0, 1],
     [ 1, 1, 0, 1, 1, 0],
     [ 0, 1, 1, 0, 0, 1],
     [ 1, 0, 1, 0, 0, 1],
     [ 0, 1, 0, 1, 1, 0]]



# inisiate the name of node.
node = "abcdef"
t_={}
for i in range(len(G)):
  t_[node[i]] = i

# count degree of all node.
degree =[]
for i in range(len(G)):
  degree.append(sum(G[i]))

# inisiate the posible color
colorDict = {}
for i in range(len(G)):
  colorDict[node[i]]=["Blue","Red","Yellow","Green"]


# sort the node depends on the degree
sortedNode=[]
indeks = []

# use selection sort
for i in range(len(degree)):
  _max = 0
  j = 0
  for j in range(len(degree)):
    if j not in indeks:
      if degree[j] > _max:
        _max = degree[j]
        idx = j
  indeks.append(idx)
  sortedNode.append(node[idx])

# The main process
theSolution={}
for n in sortedNode:
  setTheColor = colorDict[n]
  theSolution[n] = setTheColor[0]
  adjacentNode = G[t_[n]]
  for j in range(len(adjacentNode)):
    if adjacentNode[j]==1 and (setTheColor[0] in colorDict[node[j]]):
      colorDict[node[j]].remove(setTheColor[0])


# Print the solution
for t,w in sorted(theSolution.items()):
  print("Node",t," = ",w)




#hill climbing

import copy

# TOP SECOND THIRD BOTTOM
curr = [['A_B_C_D']]
goal = [['D_C_B_A']]
goal_rev = goal[0][0][::-1]

open_list = []
close_list = []


def calculate_hueristic(state, ret_flag=False):
    hueristic = 0

    for tower in state:
        wrong = -1
        right = 1
        blocks = tower[0]
        if len(blocks) > 1:
            blocks_rev = blocks[::-1]
            for i in range(2, len(blocks_rev)):
                if blocks_rev[i] != goal_rev[i] and blocks_rev[i] != '_':
                    hueristic += wrong
                    wrong += -1
                elif blocks_rev[i] == goal_rev[i] and blocks_rev[i] != '_':
                    if blocks_rev[i - 2] == goal_rev[i - 2]:
                        hueristic += right
                        right += 1
                    else:
                        hueristic += wrong
                        wrong += -1

    # push in to open list with hueristic and done
    if not ret_flag:
        open_list.append((hueristic, state))
    else:
        return hueristic


def generate_combinations():
    for idx in range(len(curr)):
        temp = copy.deepcopy(curr)
        if len(temp[idx][0]) > 1:
            # just pop and push as a list
            stack = temp[idx].pop(0)
            top = stack[0]
            temp[idx].append(stack.replace(f'{top}_', ''))
            temp.append([top])
            temp.sort(key=len)
            calculate_hueristic(temp)

        elif len(temp[idx][0]) == 1:
            # go backward from idx - 1 to -1
            for back in range(idx - 1, -1, -1):
                tower_new_state = temp[back][0]
                temp[back][0] = f'{temp[idx][0]}_' + tower_new_state
                temp.pop(idx)
                temp.sort(key=len)
                calculate_hueristic(temp)
                temp = copy.deepcopy(curr)

            for front in range(idx + 1, len(curr)):
                # go forward from idx + 1 to last

                if len(temp[front][0]) == 1:
                    single_block = temp[front][0]
                    temp[front][0] = temp[idx][0] + f'_{single_block}'

                else:
                    tower = temp[front][0]
                    temp[front][0] = temp[idx][0] + f'_{tower}'

                temp.pop(idx)
                temp.sort(key=len)
                calculate_hueristic(temp)
                temp = copy.deepcopy(curr)


hueristic = calculate_hueristic(curr, True)
goal_hueristic = calculate_hueristic(goal, True)

while hueristic < goal_hueristic:
    generate_combinations()
    open_list.sort()

    next_state = open_list.pop(-1)
    hueristic = next_state[0]
    curr = next_state[1]

    close_list.append(next_state[1])
    open_list.clear()

    print(next_state)




#minimax

MAX, MIN = -1000, 1000


def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]

    if maximizingPlayer:
        best = MAX
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MIN
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


if __name__ == "__main__":
    values = [3, 5, 6, 9, 1, 2, 0, -1]
    print("The optimal value is :", minimax(0, 0, True, values, MAX, MIN))


# canibal

import copy

# caninbals # missionary # boat at the coast
num_c = 2
num_m = 2
coast1 = [num_c, num_m, 1]
coast2 = [0, 0, 0]

open_list = []
closed_list = []
last_moves = []

are_cani_reached = False
are_mis_reached = False


def revert(idx, amt):
    if coast1[2] == 1:
        coast1[idx] -= amt
        coast2[idx] += amt
        coast1[2] = 0
        coast2[2] = 1
    else:
        coast1[idx] += amt
        coast2[idx] -= amt
        coast1[2] = 1
        coast2[2] = 0


def isvalid():
    if (coast1[0] > coast1[1] and coast1[1] != 0) or (coast2[0] > coast2[1] and coast2[1] != 0):
        return False
    return True


def move_person(idx, amt, from_coast):
    if from_coast == 1:
        coast1[idx] -= amt
        coast2[idx] += amt

        coast1[2] = 0
        coast2[2] = 1

    elif from_coast == 2:
        coast2[idx] -= amt
        coast1[idx] += amt

        coast1[2] = 1
        coast2[2] = 0


def update_open_list(cani_moved, mis_moved):
    if isvalid():
        c1 = copy.deepcopy(coast1)
        c2 = copy.deepcopy(coast2)

        open_list.append((cani_moved, mis_moved, c1, c2))


def select_move():
    state = (0, 0, [], [])
    if not are_cani_reached:
        for i in open_list:
            if state[0] < i[0] and i[2:] not in last_moves:
                state = i
        return state

    if not are_mis_reached:
        # only select min cani and mis from coast 2 and select max mis from coast 1
        if coast1[2] == 1:
            for i in open_list:
                if state[1] < i[1] and i[2:] not in last_moves:
                    state = i
            return state
        else:
            for i in open_list:
                if i[0] == 1 and i[1] <= 1 and i[2:] not in last_moves:
                    state = i
            return state
    else:
        if coast2[2] == 1:
            for i in open_list:
                if i[0] == 1 and i[1] == 0 and i[2:] not in last_moves:
                    state = i
            return state
        else:
            for i in open_list:
                if state[0] < i[0] and i[2:] not in last_moves:
                    state = i
        return state


def generate_states():
    # check where boat is
    global coast1, coast2, are_cani_reached, are_mis_reached
    if coast1[2] == 1:
        # coast1
        if coast1[0] - 1 > - 1:
            # move 1 canibals
            move_person(0, 1, 1)
            update_open_list(1, 0)

            move_person(0, 1, 2)

        if coast1[0] - 2 > -1:
            # move 2 canibals
            move_person(0, 2, 1)
            update_open_list(2, 0)
            move_person(0, 2, 2)

        if coast1[0] - 1 > -1 and coast1[1] - 1 > -1:
            # move 1 canibal and 1 mis
            move_person(0, 1, 1)
            move_person(1, 1, 1)

            update_open_list(1, 1)

            move_person(0, 1, 2)
            move_person(1, 1, 2)

        if coast1[1] - 2 > -1:
            # move 2 mis
            move_person(1, 2, 1)
            update_open_list(0, 2)

            move_person(1, 2, 2)

        if coast1[1] - 1 > -1:
            # move 1 mis
            move_person(1, 1, 1)
            update_open_list(0, 1)

            move_person(1, 1, 2)

    else:
        # coast2
        if coast2[0] - 1 > - 1:
            # move 1 canibals
            move_person(0, 1, 2)
            update_open_list(1, 0)

            move_person(0, 1, 1)

        if coast2[0] - 2 > -1:
            # move 2 canibals
            move_person(0, 2, 2)
            update_open_list(2, 0)
            move_person(0, 2, 1)

        if coast2[0] - 1 > -1 and coast2[1] - 1 > -1:
            # move 1 canibal and 1 mis
            move_person(0, 1, 2)
            move_person(1, 1, 2)

            update_open_list(1, 1)

            move_person(0, 1, 1)
            move_person(1, 1, 1)

        if coast2[1] - 2 > -1:
            # move 2 mis
            move_person(1, 2, 2)
            update_open_list(0, 2)

            move_person(1, 2, 1)

        if coast2[1] - 1 > -1:
            # move 1 mis
            move_person(1, 1, 2)
            update_open_list(0, 1)
            move_person(1, 1, 1)

    state = select_move()

    c1 = copy.deepcopy(coast1)
    c2 = copy.deepcopy(coast2)

    last_moves.append((c1, c2))

    coast1 = state[2]
    coast2 = state[3]

    if not are_cani_reached and coast2[0] == num_c:
        are_cani_reached = True

    if not are_mis_reached and coast2[1] == num_m:
        are_mis_reached = True

    print(coast1, coast2)
    open_list.clear()


while coast2 != [num_c, num_m, 1]:
    # generate state
    # select which satisfy
    # and continue until found
    generate_states()


#goal stack

import copy

# STACK SEQ TOP SECOND THIRD BOTTOM. , denotes another tower
# curr = 'BCAD'
curr = 'DBA,E,CF'
goal = 'CBE,DFA'


# docs:
# bloak A is on groud if 'A', OR ,'A'
# is A clear if 'AB' or ,'AB'


# TODO: WHILE MAKING CLEAR CHECK IF THIS BLOCK CAN BE PUT ON SOME THING ELSE TO SAVE STEPS BUT MAKE SURE YOU DONT GET Sussman Anomaly JUST CHECK IF THE TOWER STRUCTURE IS CORRECT OR NOT
def make_clear(curr_tower: str, idx_in_curr):
    dummy = copy.deepcopy(curr_tower)
    i = 0

    for back in range(idx_in_curr - 1, -1, -1):
        if dummy[back] == ',':
            break

        block = dummy[back]
        dummy = dummy.replace(block, '')
        dummy += f',{block}'
        i += 1

    return i, dummy


def is_clear(curr_tower, idx_in_curr):
    if idx_in_curr - 1 == -1 or curr_tower[idx_in_curr - 1] == ',':
        return True
    return False


def is_not_on_ground(curr, block_init_idx):
    return block_init_idx + 1 < len(curr) and curr[block_init_idx + 1] != ','


def move_to_ground(curr, moved_by, block_init_idx):
    dummy = curr[block_init_idx - moved_by]
    curr = curr.replace(dummy, '')
    return curr + f',{dummy}'


goal_towers = goal.split(',')
moved_by = 0

print(curr)


for tower in goal_towers:
    # find start finding bloaks in init from bottom
    reversed_tower = tower[::-1]

    for block_idx_in_goal_tower in range(len(tower)):

        block_in_revd = reversed_tower[block_idx_in_goal_tower]
        block_idx_in_curr = curr.find(block_in_revd)

        if block_idx_in_goal_tower == 0:
            # check if this bloak in init is on groud or not
            if is_not_on_ground(curr, block_idx_in_curr):
                # not at bottom then make at bottom
                # to do this this bloak needs to be clear
                # check if this bloak is clear or not
                if not is_clear(curr, block_idx_in_curr):
                    # if not then make it clear
                    moved_by, cleared_state = make_clear(
                        curr, block_idx_in_curr)
                    curr = cleared_state

                # clear move to ground
                curr = move_to_ground(curr, moved_by, block_idx_in_curr)

        else:
            if is_not_on_ground(curr, block_idx_in_curr):
                # not on ground
                # check if the block below it it correct
                if curr[block_idx_in_curr + 1] != reversed_tower[block_idx_in_goal_tower - 1]:
                    if not is_clear(curr, block_idx_in_curr):
                        moved_by, cleared_state = make_clear(
                            curr, block_idx_in_curr)
                        curr = cleared_state

                    to_make_clear_idx_in_curr = curr.find(
                        reversed_tower[block_idx_in_goal_tower - 1])

                    to_make_clear_block_in_curr = curr[to_make_clear_idx_in_curr]

                    if not is_clear(curr, to_make_clear_idx_in_curr):
                        moved_by, cleared_state = make_clear(
                            curr, to_make_clear_idx_in_curr)
                        curr = cleared_state

                        to_make_clear_idx_in_curr -= moved_by
                    # move this
                    curr = curr.replace(block_in_revd, '')
                    curr = curr.replace(to_make_clear_block_in_curr,
                                        block_in_revd + to_make_clear_block_in_curr)

            else:
                # on the ground
                # can we move it ?
                if not is_clear(curr, block_idx_in_curr):
                    # if not clear make it
                    moved_by, cleared_state = make_clear(
                        curr, block_idx_in_curr)
                    curr = cleared_state
                    block_idx_in_curr -= moved_by

                # check where to move is clear or not
                to_make_clear_idx_in_curr = curr.find(
                    reversed_tower[block_idx_in_goal_tower - 1])

                to_make_clear_block_in_curr = curr[to_make_clear_idx_in_curr]

                if not is_clear(curr, to_make_clear_idx_in_curr):
                    moved_by, cleared_state = make_clear(
                        curr, to_make_clear_idx_in_curr)
                    curr = cleared_state

                    to_make_clear_idx_in_curr -= moved_by

                # move this
                if block_idx_in_curr != 0:
                    to_remove = ',' + block_in_revd
                else:
                    to_remove = block_in_revd + ','
                curr = curr.replace(to_remove, '')
                curr = curr.replace(to_make_clear_block_in_curr,
                                    block_in_revd + to_make_clear_block_in_curr)


print(curr)
print(goal)


#network

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Define the structure of the Bayesian Network
model = BayesianNetwork([('e', 'm'), ('i', 'm'), ('i', 's'), ('m', 'a')])

'''
                    Student
                      |
                      v
exam ---> marks <--- IQ
           |
           v
      admission
'''

# Define Conditional Probability Distributions (CPDs)

cpd_e = TabularCPD(variable='e', variable_card=2, values=[[0.6], [0.4]])
cpd_i = TabularCPD(variable='i', variable_card=2, values=[[0.7], [0.3]])
cpd_m = TabularCPD(variable='m', variable_card=2,
                   evidence=['e', 'i'],
                   values=[[0.9, 0.6, 0.7, 0.1],
                           [0.1, 0.4, 0.3, 0.9]],
                   evidence_card=[2, 2])
cpd_s = TabularCPD(variable='s', variable_card=2,
                   evidence=['i'],
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence_card=[2])
cpd_a = TabularCPD(variable='a', variable_card=2,
                   evidence=['m'],
                   values=[[0.8, 0.1],
                           [0.2, 0.9]],
                   evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_e, cpd_i, cpd_m, cpd_s, cpd_a)

# Check if the model is consistent
print("Model is consistent:", model.check_model())

# Print CPDs
for cpd in model.get_cpds():
    print("CPD for {}: \n{}".format(cpd.variable, cpd))

# Doing exact inference using Variable Elimination
from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

# Calculate the probability of admission given marks=1
print("\nProbability of admission given marks=1:")
print(infer.query(variables=['a'], evidence={'m': 1}).values[0])

# Calculate the probability of marks given exam=0 and IQ=1
print("\nProbability of marks given exam=0 and IQ=1:")
print(infer.query(variables=['m'], evidence={'e': 0, 'i': 1}).values)