import numpy as np

import matplotlib.pyplot as plt

from random import choice
"""
In GridWorld, we design our grid world- ACTION_LIST contains all the possible
actions that the agent can take i.e. left, right, up or down
and it is used to create the grid
"""

class GridWorld:

#in this class we design our grid world- ACTION_LIST contains all the possible

#actions that the agent can take i.e. left, right, up or down.

    ACTION_LIST = ( 'L' , 'R' , 'U' ,  'D')


    def __init__(self, width, height, startstate, actions, wall , trapstate, goalstate):

     #we define the height and width of our grid, state from which it starts, possible actions

     #trap states, states acting as the wall, trap state and the goal state.

        self.width = width #width of the grid

        self.height = height #height of the grid

        self.x = startstate[0]  # row from which the agent starts - start state coordinates

        self.y = startstate[1]  # y = column from which the agent starts - start state coordinates

        self.action = actions # Actions that it can take

        self.walls = wall # states acting as the wall in the grid

        self.trapstate = trapstate # trap state

        self.goalstate = goalstate # goal state

        self.startstate = startstate# grid start state

        self.gamma = 1 # discount factor [** but changed in policy iterations and value iterations

        self.transition_reward = -0.02 #we will need this in case of Sarsa

    #set the start state of the agent as [0,1]

    def set_state(self, s):

        self.x = s[0]

        self.y = s[1]

     #tells the current state the agent is in

    def current_state(self):

        return self.x, self.y

    #this function covers the condition when agent is the topmost row or the last row

    #and performs action up or down respectively

    #or is in the leftmost or the rightmost column

    #and performs action left or right respectively.

    def out_of_grid(self):

        # If row coordinate goes out of the grid

        if self.x < 0:

            self.x = 0

        if self.x == self.width:

            self.x = self.width - 1

        # If column coordinate goes out of the grid

        if self.y < 0:

            self.y = 0

        if self.y == self.height:

            self.y = self.height - 1


     #this function tells us if the agent is in a terminal state-

    #Terminal State is the goal state

    def is_terminalstate(self, s):

        is_terminalstate = False

        if s in self.trapstate:

            is_terminalstate = True

        if s == self.goalstate:

            is_terminalstate = True

        return is_terminalstate



    # agent performs an action - and moves to the next state

    # this function returns the following: current state, action performed, next state, reward

    def move(self, state, action):

    #initial coordinates

        self.x = state[0]

        self.y = state[1]


        if action == 'L':

            self.x -= 1

       #already at leftmost position - will get a negative reward on performing left action

        elif action == 'R':

            self.x += 1

       #going right means moving forward

        elif action == 'U':

            self.y -= 1

       #already at topmost position - will get a negative reward on performing up action

        elif action == 'D':

            self.y += 1

        #going right means moving forward

        # If agent goes out of grid on performing action it stays on the current state

        self.out_of_grid()


        #If agent encounters wall- it will bounce back to the state it started from

       #in that case - we undid the action that it performed

        if (self.x, self.y) in self.walls:

            self.undo_action(action)


        next_state = (self.x, self.y)

        reward = self.get_reward(self.x, self.y)


        return state, action, next_state, reward

     #for all the other state transitions: reward = -0.02

     #for trap state: reward = -1, for goal state: reward = +1

    def get_reward(self, x, y):

        reward = -0.02

        if (x, y) in self.trapstate:

            reward = -1

        elif (x, y) == self.goalstate:

            reward = +1

        return reward

    #for the cases when the agent bounces back to the current state we perfrom undo

   #this function performs the opposite of what these moves originally do

    def undo_action(self, action):
        if action == 'L':
            self.x += 1
        elif action == 'R':
            self.x -= 1
        elif action == 'U':
            self.y += 1
        elif action == 'D':
            self.y -= 1



        # If agent goes out of the grid then it undo the action and stays on the current state

        self.out_of_grid()


    @staticmethod

    def create_grid():

        # X : x-axis oordinate , Y : y-axis oordinate

        # State positions are indicated as (X, Y)

        # S - Start state

        # G - Goal state:  reward = +1

        # T - Trap state: reward = -1

        # W- Walls


        #     0   1   2  3  4  5  6   7

        # 7 | T  W  -  -  T  W  W  G

        # 6 | -    -   -  -   -   -   -    -

        # 5 | -   -   W W W -  -     -

        # 4 | -   -   T  -   W  -  W  -

        # 3 | -  W  W -   -    -  W  -

        # 2 | -   -    W -  -   W  T  -

        # 1 | -  -   -   -   -  W  W  -

        # 0 | S -  -  W  -   -     -    -

        width = 8

        height = 8

        startstate = (0, 0)

        goalstate = (7, 7)

        walls = [(1, 7), (5, 7), (6, 7), (2, 5), (3, 5), (4, 5), (4, 4), (6, 4), (1, 4), (2, 2), (6, 3), (2, 3), (5,        2),(5, 1), (6, 1), (3, 0)]

        trapstate = [(0,7), (4, 0), (2, 4), (6, 2)]

        my_grid = GridWorld(width, height, startstate, GridWorld.ACTION_LIST, walls, trapstate, goalstate)


        return my_grid

     #set the state to start state

    def gameover(self):

        self.set_state(self.startstate)

   #if action is up or down - sideways acion will be left or right and vice versa

    def get_sideways_actions(self, action):

        sideways = []

        if action == 'U' or action == 'D':

            sideways = ['L', 'R']

        elif action == 'L' or action == 'R':

            sideways = ['U', 'D']

        return sideways


    @staticmethod

    def turn_right(action):

        return GridMDP.ACTION_LIST[(GridWorld.ACTION_LIST.index(action) + 1) % len(GridWorld.ACTION_LIST)]


    @staticmethod

    def turn_left(action):

        return GridWorld.ACTION_LIST[GridWorld.ACTION_LIST.index(action) - 1]


    # building transition matrix

    def transition_matrix(self, state, action, uniform = False):


        if(uniform):

            uniform_matrix = [(0.25, self.move(state, m)) for m in GridWorld.ACTION_LIST if m != action]

            uniform_matrix.append((0.25, self.move(state, action)))

            return uniform_matrix


        t1 = [0.7, self.move(state, action)]

        t2 = [0.15, self.move(state, GridWorld.turn_right(action))]

        t3 = [0.15, self.move(state, GridWorld.turn_left(action))]


        return [t1, t2, t3]



def get_all_states():

    """


    :return: grid = Grid Object, l = list of accessible states

    """

    grid = GridWorld.create_grid()

    initial_grid = np.zeros(shape=(grid.width, grid.height))

    l = [(i, j) for i, y in enumerate(initial_grid) for j, x in enumerate(y) if (i, j) not in grid.walls]


    # Start state initialization

    l[l.index(grid.startstate)], l[0] = l[0], l[l.index(grid.startstate)]


    return grid, l



def reward_matrix():

    pass



g, l = get_all_states()

s = choice(l)

a = choice(GridWorld.ACTION_LIST)

l1 = g.transition_matrix(s, a, True)

# print(l1)
