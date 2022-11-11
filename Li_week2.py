class MyKarel:
    
    def __init__(self):
        self.actions = ["m", "l", "r", "p", "f"] # move, turnLeft, turnRight, pickMarker, finish
        self.walls = [[3, 2], [1, 3], [0, 5], [4, 3]] # self._set_walls(), set random walls in the grid
        self.markers = [[5, 0], [2, 1]] # self._set_markers(), set random markers in the grid
        self.x, self.y = 6, 6 # can be changed in the future
        self.gamma = 0.9 # for reward discount
        self.s_0 = (0, 3, 1, 0)  # d: 0/1/2/3: west/south/east/north | x/y: 0-5 | h: 0/1: empty/marker
        # reward function setting for now.
        self.r_minus = -1
        self.r_small_minus = -0.1
        self.r_small_positive = 0.1
        self.r_positive = 1
       
    def _set_walls(self): # TODO
        walls = 0
        return walls
    
    def _set_markers(self): # TODO
        markers = 0
        return markers
    
    def reset(self):
        self.walls = [[3, 2], [1, 3], [0, 5], [4, 3]]
        self.markers = [[5, 0], [2, 1]]
        self.agent = self.s_0

    def transition(self, a):
        d, x, y, h = self.agent
        if a == "m" : 
            if d == 0: # to west
                if (y-1) < 0 or [x, y-1] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    self.agent = (d, x, y-1, h)
                    t = False
                    r = self.r_small_minus
            elif d == 1: # to south
                if (x+1) > 5 or [x+1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    self.agent = (d, x+1, y, h)
                    t = False
                    r = self.r_small_minus
            elif d == 2: # to east
                if (y+1) > 5 or [x, y+1] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    self.agent = (d, x, y+1, h)
                    t = False
                    r = self.r_small_minus
            elif d == 3: # to north
                if (x-1) < 0 or [x-1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    self.agent = (d, x-1, y, h)
                    t = False
                    r = self.r_small_minus
        
        if a == "l" :
            d = (d+1) % 4
            self.agent = (d, x, y, h)
            t = False
            r = self.r_small_minus

        if a == "r" :
            d = (d+3) % 4
            self.agent = (d, x, y, h)
            t = False
            r = self.r_small_minus
        
        if a == "p" :
            if [self.agent[1], self.agent[2]] not in self.markers:
                t = True
                r = self.r_minus
            else:
                self.agent = (d, x, y, 1)
                self.markers.remove([x, y])
                t = False
                r = self.r_small_positive
        
        if a == "f" :
            t = True
            if x == 5 and y == 2 and h == 1: r = self.r_positive
            else: r = self.r_minus
        
        return t, r

    def show(self):
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == self.agent[1] and y == self.agent[2]:
                    if self.agent[0] == 0:
                        print('w ', end="")
                    elif self.agent[0] == 1:
                        print('s ', end="")
                    elif self.agent[0] == 2:
                        print('e ', end="")
                    elif self.agent[0] == 3:
                        print('n ', end="")
                elif [x, y] in self.walls:
                    print('W ', end="")
                elif [x, y] in self.markers:
                    print('D ', end="")
                else:
                    print("- ", end="")
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
                termination, r = self.transition(a)
                print("reward:", r)
                self.show() 

if __name__ == '__main__':
   
    play = MyKarel()
    play.render()