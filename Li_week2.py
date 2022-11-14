
class MyKarel:
    
    def __init__(self):
        self.actions = ["m", "l", "r", "p", "f"] # Action space: move, turnLeft, turnRight, pickMarker, finish
        self.walls = [[3, 2], [1, 3], [0, 5], [4, 3]] # TODO: self._set_walls(): set walls randomly in the grid
        self.markers = [[5, 0], [2, 1]] # TODO: self._set_markers(), set markers randomly in the grid
        self.x, self.y = 6, 6 # 6*6 grid world
        self.gamma = 0.9 # For reward discount
        # State: d: 0/1/2/3: west/south/east/north | x/y: 0-5 | h: 0/1/2: empty/hold 1/2 marker | g: empty(0)/marker(1)/wall(2)
        self.s_0 = (0, 3, 1, 0, 0) # Init state
        # Reward function setting for now
        self.r_minus = -1
        self.r_small_minus = -0.1
        self.r_small_positive = 0.1
        self.r_positive = 1
        # Max episode step setting and step counter
        self.max_steps = 100  
       
    def _set_walls(self): # TODO
        walls = []
        return walls
    
    def _set_markers(self): # TODO
        markers = []
        return markers
    
    # Reset environment
    def reset(self):
        self.markers = [[5, 0], [2, 1]] # In case any marker has been picked in the last episode
        self.agent = self.s_0 # Set init state to the agent
        self.steps = 0 # Step counter for the current episode
        self.return_all = 0 # Accumulative reward in current episode

    # Transition probability function
    def transition(self, a):
        d, x, y, h, g = self.agent
        if a == "m" : # Action move
            if d == 0: # To west
                if (y-1) < 0: # Cross the edge: termination and return minus reward
                    t = True
                    r = self.r_minus
                elif [x, y-1] in self.walls: # Hit the wall: termination and return minus reward
                    self.agent = (d, x, y-1, h, 2)
                    t = True
                    r = self.r_minus
                else: # Move to the destination grid and return small minus reward
                    if [x, y-1] in self.markers:
                        self.agent = (d, x, y-1, h, 1)
                    else: 
                        self.agent = (d, x, y-1, h, 0)
                    t = False
                    r = self.r_small_minus
            elif d == 1: # To south
                if (x+1) > 5:
                    t = True
                    r = self.r_minus
                elif [x+1, y] in self.walls:
                    self.agent = (d, x+1, y, h, 2)
                    t = True
                    r = self.r_minus
                else:
                    if [x+1, y] in self.markers:
                        self.agent = (d, x+1, y, h, 1)
                    else: 
                        self.agent = (d, x+1, y, h, 0)
                    t = False
                    r = self.r_small_minus
            elif d == 2: # To east
                if (y+1) > 5:
                    t = True
                    r = self.r_minus
                elif [x, y+1] in self.walls:
                    self.agent = (d, x, y+1, h, 2)
                    t = True
                    r = self.r_minus
                else:
                    if [x, y+1] in self.markers:
                        self.agent = (d, x, y+1, h, 1)
                    else: 
                        self.agent = (d, x, y+1, h, 0)
                    t = False
                    r = self.r_small_minus
            elif d == 3: # To north
                if (x-1) < 0:
                    t = True
                    r = self.r_minus
                elif [x-1, y] in self.walls:
                    self.agent = (d, x-1, y, h, 2)
                    t = True
                    r = self.r_minus
                else:
                    if [x-1, y] in self.markers:
                        self.agent = (d, x-1, y, h, 1)
                    else: 
                        self.agent = (d, x-1, y, h, 0)
                    t = False
                    r = self.r_small_minus
        
        if a == "l" : # Action turnLeft
            d = (d+1) % 4
            self.agent = (d, x, y, h, g)
            t = False
            r = self.r_small_minus # Return small minus reward

        if a == "r" : # Action turnRight
            d = (d+3) % 4
            self.agent = (d, x, y, h, g)
            t = False
            r = self.r_small_minus # Return small minus reward
        
        if a == "p" : # Action pickMarker
            if [self.agent[1], self.agent[2]] not in self.markers: # no MARKER to pick: termination and return minus reward
                t = True
                r = self.r_minus
            else: # pick the MARKER and change h to h+1 (hold a marker) and change the grid status to 0 (empty)
                self.agent = (d, x, y, h+1, 0)
                self.markers.remove([x, y]) # Remove this MARKER from the environment
                t = False
                r = self.r_small_positive # Return small positive reward
        
        if a == "f" : # Action finish
            t = True
            # If the agent at target grid with the specific MARKER picked: return positive reward; else: return minus reward
            if x == 5 and y == 2 and [5, 0] not in self.markers: r = self.r_positive
            else: r = self.r_minus
        
        return t, r

    # To show the grid world with: "-" for empty grid, "D" for MARKER, "W" for wall, 
    # "w/s/e/n" for the agent's current location with direction "west/south/east/north"
    def _show(self):
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

    # To run the game
    def run(self):
        termination = True
        while True:
            if termination:
                # print()
                print("** New Episode **")
                self.reset()
                self._show()

            print("Enter your action from m/l/r/p/f or q to quit: ", end="")
            a = input()
            print()
            if a == "q":
                break
            if a in self.actions:
                termination, r = self.transition(a)
                self.return_all = r + self.gamma * self.return_all
                self.steps += 1
                print("reward:", r)
                print("accumulative reward", round(self.return_all, 3))
                if self.steps >= self.max_steps: termination = True # Episode steps more than 100: Start a new episode
                self._show() 

if __name__ == '__main__':
   
    play = MyKarel()
    play.run()