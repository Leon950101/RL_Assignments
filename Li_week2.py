# import numpy as np

class MyKarel:
    
    def __init__(self):
        self.actions = ["m", "l", "r", "p", "f"] # move, turnLeft, turnRight, pickMarker, finish
        self.walls = [[3, 2], [1, 3], [0, 5], [4, 3]] # self._set_walls(), set random walls in the grid
        self.diamonds = [[5, 0], [2, 1]] # self._set_diamonds(), set random diamonds in the grid
        self.action_index = -1 # for automatical run
        self.x, self.y = 6, 6 # can be changed in the future
        self.states = self._set_states()
       
    def _set_walls(self): # TODO
        walls = 0
        return walls
    def _set_diamonds(self): # TODO
        diamonds = 0
        return diamonds
       
    def _set_states(self):
        states = [[0] * 4] * (4 * self.x * self.y) # np.zeros((self.x * self.y * 4, 4), dtype=int)
        index = 0
        for x in range(self.x):
            for y in range(self.y):
                for d in range(4):
                    if [x, y] in self.walls:
                        states[index] = d, x, y, 1
                    elif [x, y] in self.diamonds:
                        states[index] = d, x, y, 2
                    else:
                        states[index] = d, x, y, 0
                    index += 1
        return states
    
    def reset(self):
        self.walls = [[3, 2], [1, 3], [0, 5], [4, 3]]
        self.diamonds = [[5, 0], [2, 1]]
        self.action_index = -1
        # S_0
        self.agent = (0, 3, 1, 0) # d: for direction, 0: west, 1: south, 2: east, 3: north | x: 0-5 | y: 0-5 | g: 0: empty, 1: diamond, 2: wall

    def transition(self, a):
        d, x, y, g = self.agent
        if a == "m" : t = self._move(d, x, y, g)
        if a == "l" : t = self._turnLeft(d, x, y, g)
        if a == "r" : t = self._turnRight(d, x, y, g)
        if a == "p" : t = self._pickMarker(d, x, y, g)
        if a == "f" : t = self._finish(d, x, y, g)
        return t

    def _move(self, d, x, y, g):
        if d == 0: # to west
            if (y-1) < 0 or [x, y-1] in self.walls:
                t = True
            else:
                self.agent = (d, x, y-1, g)
                t = False
        elif d == 1: # to south
            if (x+1) > 5 or [x+1, y] in self.walls:
                t = True
            else:
                self.agent = (d, x+1, y, g)
                t = False
        elif d == 2: # to east
            if (y+1) > 5 or [x, y+1] in self.walls:
                t = True
            else:
                self.agent = (d, x, y+1, g)
                t = False

        elif d == 3: # to north
            if (x-1) < 0 or [x-1, y] in self.walls:
                t = True
            else:
                self.agent = (d, x-1, y, g)
                t = False
        return t

    def _turnLeft(self, d, x, y, g):
        d = (d+1) % 4
        self.agent = (d, x, y, g)
        t = False
        return t 

    def _turnRight(self, d, x, y, g):
        d = (d+3) % 4
        self.agent = (d, x, y, g)
        t = False
        return t

    def _pickMarker(self, d, x, y, g): # TODO
        if [self.agent[1], self.agent[2]] not in self.diamonds:
            t = True
        else:
            self.agent = (d, x, y, 0)
            self.diamonds.remove([x, y])
            t = False
        return t

    def _finish(self, d, x, y, g):
        t = True
        return t

    def show(self):
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == self.agent[1] and y == self.agent[2]:
                    if self.agent[0] == 0:
                        print('w', end="")
                    elif self.agent[0] == 1:
                        print('s', end="")
                    elif self.agent[0] == 2:
                        print('e', end="")
                    elif self.agent[0] == 3:
                        print('n', end="")
                elif [x, y] in self.walls:
                    print('W', end="")
                elif [x, y] in self.diamonds:
                    print('D', end="")
                else:
                    print("-", end="")
        print()
        print()

    def render(self):
        termination = True
        while True:
            if termination:
                print()
                print("***** New Episode *****")
                self.reset()
                self.show()

            print("Action? (five actions or q to quit) ", end="")
            a = input() # self.run(): run the optimal actions automatically
            print()
            if a == "q":
                break
            if a in self.actions:
                termination = self.transition(a)
                self.show()
    
    def run(self):
        actions = ["m", "l", "m", "m", "p", "l", "m", "m", "f", "q"]
        self.action_index += 1
        return actions[self.action_index]

    def reward_function(self, d, x, y, g, a): # TODO no need in this task
        return
    
if __name__ == '__main__':
   
    play = MyKarel()
    play.render()