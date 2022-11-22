
class MyKarel_1:
    
    def __init__(self):
        # TODO Terminal state ?
        self.actions = ["m", "l", "r", "p", "f"] # Action space: move, turnLeft, turnRight, pickMarker, finish
        self.walls = [[3, 2], [1, 3], [0, 5], [4, 3]] # Set walls
        self.markers = [[5, 0], [2, 1]] # Set markers
        self.x, self.y = 6, 6 # 6*6 grid world
        self.gamma = 0.99 # For reward discount, not needed for episodic task
        # State: d: 0/1/2/3: west/south/east/north | x/y: 0-5 |
        # 1st MARKER status: 0/1: Present or not | 2nd MARKER status: 0/1: Present or not
        # Agent will always have MARKERs in hand and can pick up infinity MARKERs
        self.s_0 = (0, 3, 1, 1, 1) # Init state
        self.s_f = (2, 5, 2, 0, 1) # Target state
        # Reward function setting for now
        self.r_minus = -1
        # self.r_small_minus = -0.1
        # self.r_small_positive = 0.1
        self.r_positive = 100
        # Max episode step setting and step counter
        # self.max_steps = 100  
    
    # Reset environment
    def reset(self):
        self.markers = [[5, 0], [2, 1]] # In case any marker has been picked in the last episode
        self.agent = self.s_0 # Set init state to the agent
        # self.steps = 0 # Step counter for the current episode
        self.return_all = 0 # Accumulative reward in current episode
        return self.agent

    # Transition probability function
    def transition(self, s, a):
        d, x, y, m1, m2 = s
        s_1 = (d, x, y, m1, m2)
        if a == "m" : # Action move
            if d == 0: # To west
                if (y-1) < 0: # Cross the edge: termination and return minus reward
                    t = True
                    r = self.r_minus
                elif [x, y-1] in self.walls: # Hit the wall: termination and return minus reward
                    s_1 = (d, x, y-1, m1, m2)
                    t = True
                    r = self.r_minus
                else: # Move to the destination grid
                    if [x, y-1] in self.markers:
                        s_1 = (d, x, y-1, m1, m2)
                    else: 
                        s_1 = (d, x, y-1, m1, m2)
                    t = False
                    r = self.r_minus
            elif d == 1: # To south
                if (x+1) > 5:
                    t = True
                    r = self.r_minus
                elif [x+1, y] in self.walls:
                    s_1 = (d, x+1, y, m1, m2)
                    t = True
                    r = self.r_minus
                else:
                    if [x+1, y] in self.markers:
                        s_1 = (d, x+1, y, m1, m2)
                    else: 
                        s_1 = (d, x+1, y, m1, m2)
                    t = False
                    r = self.r_minus
            elif d == 2: # To east
                if (y+1) > 5:
                    t = True
                    r = self.r_minus
                elif [x, y+1] in self.walls:
                    s_1 = (d, x, y+1, m1, m2)
                    t = True
                    r = self.r_minus
                else:
                    if [x, y+1] in self.markers:
                        s_1 = (d, x, y+1, m1, m2)
                    else: 
                        s_1 = (d, x, y+1, m1, m2)
                    t = False
                    r = self.r_minus
            elif d == 3: # To north
                if (x-1) < 0:
                    t = True
                    r = self.r_minus
                elif [x-1, y] in self.walls:
                    s_1 = (d, x-1, y, m1, m2)
                    t = True
                    r = self.r_minus
                else:
                    if [x-1, y] in self.markers:
                        s_1 = (d, x-1, y, m1, m2)
                    else: 
                        s_1 = (d, x-1, y, m1, m2)
                    t = False
                    r = self.r_minus
        
        if a == "l" : # Action turnLeft
            d = (d+1) % 4
            s_1 = (d, x, y, m1, m2)
            t = False
            r = self.r_minus

        if a == "r" : # Action turnRight
            d = (d+3) % 4
            s_1 = (d, x, y, m1, m2)
            t = False
            r = self.r_minus
        
        if a == "p" : # Action pickMarker
            if [x, y] not in self.markers: # no MARKER to pick: termination and return minus reward
                t = True
                r = self.r_minus
            else:
                if x == 5 and y == 0:
                    s_1 = (d, x, y, 0, m2) # Pick 1st MARKER
                else: 
                    s_1 = (d, x, y, m1, 0) # Pick 2nd MARKER
                self.markers.remove([x, y]) # Remove this MARKER from the environment
                t = False
                r = self.r_minus
        
        if a == "f" : # Action finish
            t = True
            # If the agent at target grid with the specific MARKER picked: return positive reward; else: return minus reward
            if s_1 == self.s_f: r = self.r_positive
            else: r = self.r_minus
        
        return t, r, s_1

    # To show the grid world with: "-" for empty grid, "D" for MARKER, "W" for wall, 
    # "w/s/e/n" for the agent's current location with direction "west/south/east/north"
    def show(self, s):
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == s[1] and y == s[2]:
                    if s[0] == 0:
                        print('w ', end="")
                    elif s[0] == 1:
                        print('s ', end="")
                    elif s[0] == 2:
                        print('e ', end="")
                    elif s[0] == 3:
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
                s = self.reset()
                self.show(s)

            print("Enter your action from m/l/r/p/f or q to quit: ", end="")
            a = input()
            print()
            if a == "q":
                break
            if a in self.actions:
                termination, r, s_1 = self.transition(s, a)
                s = s_1
                self.return_all = r + self.gamma * self.return_all
                # self.steps += 1
                print("reward:", r)
                print("accumulative reward", round(self.return_all, 3))
                # if self.steps >= self.max_steps: termination = True # Episode steps more than 100: Start a new episode
                self.show(s) 

if __name__ == '__main__':
   
    play = MyKarel_1()
    play.run()