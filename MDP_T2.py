
class MyKarel_2:
    
    def __init__(self):
        self.actions = ["m", "l", "r", "f"] # Action space: move, turnLeft, turnRight, finish
        self.walls = [[1, 2], [2, 3]] # Set walls
        self.markers = [] # Set markers
        self.x, self.y = 4, 4 # 4*4 grid world
        # State: d: 0/1/2/3: west/south/east/north | x/y: 0-3
        self.s_0 = (0, 1, 1) # Init state
        self.s_f = (2, 3, 2) # Target state
        # Reward function
        self.r_minus = 0
        self.r_positive = 1
    
    # Reset environment
    def reset(self):
        return self.s_0 # Set init state to the agent

    # Transition probability function
    def transition(self, s, a):
        d, x, y = s
        s_1 = (d, x, y) # Next state
        t = True # Termination status
        r = 0 # Reward
        if a == "m" : # Action "move"
            if d == 0: # To west
                if (y-1) < 0 or [x, y-1] in self.walls: # Cross the edge or Hit the wall: termination and return minus reward (0)
                    t = True
                    r = self.r_minus
                else: # Move to the destination grid and return minus reward (0)
                    s_1 = (d, x, y-1)
                    t = False
                    r = self.r_minus
            elif d == 1: # To south
                if (x+1) > (self.x - 1) or [x+1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    s_1 = (d, x+1, y)
                    t = False
                    r = self.r_minus
            elif d == 2: # To east
                if (y+1) > (self.y - 1) or [x, y+1] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    s_1 = (d, x, y+1)
                    t = False
                    r = self.r_minus
            elif d == 3: # To north
                if (x-1) < 0 or [x-1, y] in self.walls:
                    t = True
                    r = self.r_minus
                else:
                    s_1 = (d, x-1, y)
                    t = False
                    r = self.r_minus
        
        if a == "l" : # Action "turnLeft" 
            d = (d+1) % 4 # Change the direction and return minus reward (0)
            s_1 = (d, x, y)
            t = False
            r = self.r_minus

        if a == "r" : # Action "turnRight"
            d = (d+3) % 4
            s_1 = (d, x, y)
            t = False
            r = self.r_minus
        
        if a == "f" : # Action "finish"
            t = True
            # If the agent at target grid with specific direction: return positive reward (1)
            if s_1 == self.s_f: r = self.r_positive
            # Else: return minus reward (0)
            else: r = self.r_minus
        
        return t, r, s_1

    # Show the grid world with: "-" for empty grid, "#" for wall, 
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
                    print('# ', end="")
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

            print("Enter your action from m/l/r/f or q to quit: ", end="")
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
   
    play = MyKarel_2()
    play.run()