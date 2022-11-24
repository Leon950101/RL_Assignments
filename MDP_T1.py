
class MyKarel_1:
    
    def __init__(self):
        self.actions = ["m", "l", "r", "p", "f"] # Action space: move, turnLeft, turnRight, pickMarker, finish
        self.walls = [[3, 2], [1, 3], [0, 5], [4, 3]] # Set walls
        self.markers = [[5, 0], [2, 1]] # Set markers
        self.x, self.y = 6, 6 # 6*6 grid world
        # State: d: 0/1/2/3: west/south/east/north | x/y: 0-5
        # 1st MARKER status: 0/1: Present or not | 2nd MARKER status: 0/1: Present or not
        self.s_0 = (0, 3, 1, 1, 1) # Init state 
        self.s_f = (2, 5, 2, 0, 1) # Target state 
        # Reward function
        self.r_minus = -1
        self.r_pick_m1 = 10
        self.r_positive = 100
    
    # Reset environment
    def reset(self):
        return self.s_0 # Set init state to the agent

    # Transition probability function
    def transition(self, s, a):
        d, x, y, m1, m2 = s
        s_1 = (d, x, y, m1, m2) # Next state
        t = True # Termination status
        r = 0 # Reward
        if a == "m" : # Action "move"
            if d == 0: # To west
                if (y-1) < 0 or [x, y-1] in self.walls: # Cross the edge or Hit the wall: termination and return minus reward (-1)
                    t = True
                    r = self.r_minus
                else: # Move to the destination grid and return minus reward (-1)
                    s_1 = (d, x, y-1, m1, m2)
                    t = False
                    r = self.r_minus
            elif d == 1: # To south
                if (x+1) > (self.x - 1) or [x+1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    s_1 = (d, x+1, y, m1, m2)
                    t = False
                    r = self.r_minus
            elif d == 2: # To east
                if (y+1) > (self.y - 1) or [x, y+1] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    s_1 = (d, x, y+1, m1, m2)
                    t = False
                    r = self.r_minus
            elif d == 3: # To north
                if (x-1) < 0 or [x-1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    s_1 = (d, x-1, y, m1, m2)
                    t = False
                    r = self.r_minus
        
        if a == "l" : # Action "turnLeft"
            d = (d+1) % 4
            s_1 = (d, x, y, m1, m2)
            t = False
            r = self.r_minus

        if a == "r" : # Action "turnRight"
            d = (d+3) % 4
            s_1 = (d, x, y, m1, m2)
            t = False
            r = self.r_minus
        
        if a == "p" : # Action "pickMarker"
            if [x, y] not in self.markers: # not MARKER grid-cell: termination and return minus reward (-1)
                t = True
                r = self.r_minus
            else:
                t = False
                if x == 5 and y == 0 and m1 == 1: # MARKER 1 is still there
                    s_1 = (d, x, y, 0, m2) # Pick 1st MARKER (the target one), ruturn minus reward (-1) or extra reward (10)
                    r = self.r_pick_m1
                elif x == 2 and y == 1 and m2 == 1: # MARKER 2 is still there
                    s_1 = (d, x, y, m1, 0) # Pick 2nd MARKER, return minus reward (-1)
                    r = self.r_minus
                else: 
                    t = True
                    r = self.r_minus
                
        if a == "f" : # Action "finish"
            t = True
            # If the agent at target grid with the specific MARKER picked: return positive reward (100);
            if s_1 == self.s_f: r = self.r_positive
            # Else: return minus reward (-1)
            else: r = self.r_minus
        
        return t, r, s_1

    # Show the grid world with: "-" for empty grid, "o" for MARKER, "#" for WALL, 
    # "w/s/e/n" for the agent's current location with direction "west/south/east/north",
    # "W/S/E/N" for the agent's current location with a MARKER on
    def show(self, s):
        for x in range(self.x):
            print()
            for y in range(self.y):
                if x == s[1] and y == s[2]: # Agent current location
                    if s[0] == 0:
                        if x == 5 and y == 0 and s[3] == 1: print('W ', end="")
                        elif x == 2 and y == 1 and s[4] == 1: print('W ', end="")
                        else: print('w ', end="")
                    elif s[0] == 1:
                        if x == 5 and y == 0 and s[3] == 1: print('S ', end="")
                        elif x == 2 and y == 1 and s[4] == 1: print('S ', end="")
                        else: print('s ', end="")
                    elif s[0] == 2:
                        if x == 5 and y == 0 and s[3] == 1: print('E ', end="")
                        elif x == 2 and y == 1 and s[4] == 1: print('E ', end="")
                        else: print('e ', end="")
                    elif s[0] == 3:
                        if x == 5 and y == 0 and s[3] == 1: print('N', end="")
                        elif x == 2 and y == 1 and s[4] == 1: print('N ', end="")
                        else: print('n ', end="")
                elif [x, y] in self.walls:
                    print('# ', end="")
                elif [x, y] in self.markers:
                    if x == 5 and y == 0 and s[3] == 0: print("- ", end="")
                    elif  x == 2 and y == 1 and s[4] == 0: print("- ", end="")
                    else: print('o ', end="")
                else:
                    print("- ", end="")
        print()
        print()

    # Run the game
    def run(self):
        termination = True
        while True:
            if termination:
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
                print("reward:", r)
                self.show(s) 

if __name__ == '__main__':
   
    play = MyKarel_1()
    play.run()