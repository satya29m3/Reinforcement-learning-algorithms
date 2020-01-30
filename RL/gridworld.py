import  numpy as np

class grid():
    def __init__(self,wt,ht,start):
        self.wt = wt
        self.ht = ht
        self.start = start
    
    def set(self,rewards ,actions):
        self.rewards = rewards
        self.actions = actions
    
    def set_state(self,s):
        self.i = s[0]
        self.j = s[1]
    
    def curren_state(self):
        return (self.i,self.j)

    def  is_terminal(self,s):
        return s not in self.actions
    
    def move(self,action):
        if action in self.actions[(self.i,self.j)]:
            if action == 'U':
                self.i -= 1
            if action == 'D':
                self.i +=1
            if action == 'R':
                self.j -= 1
            if action == 'L':
                self.j += 1
        return self.rewards.get((self.i,self.j),0)

    def undo_move(self,action):
        if action == 'U':
            self.i += 1
        if action == 'D':
            self -= 1
        if action == 'R':
            self.j += 1
        if action == 'L':
            self.j -= 1
        assert(self.curren_state() in self.all_states())
    
    def game_over(self):
        return (self.i,self.j) not in self.actions
    
    def all_states(self):
        return set(self.actions.keys() + self.rewards.keys())
    
def standard_grid():
    g = grid(3,4,(2,0))
    rewards = {(0,3):1 , (1,3):-1}
    actions = {
        (0,0):('D','R'),
        (0,1):('L','R'),
        (0,2):('L','D','R'),
        (1,0):('U','D'),
        (1,2):('U','D','R'),
        (2,0):('U','R'),
        (2,1):('R','L'),
        (2,2):('U','R','L'),
        (2,3):('U','L')
    }
    g.set(rewards,actions)
    return g

def neg_grid(step_cost =-0.1):
    g = standard_grid()
    g.rewards.update({
      (0,0):step_cost,
      (0,1):step_cost,
      (0,2):step_cost,
      (1,0):step_cost,
      (1,2):step_cost,
      (2,0):step_cost,
      (2,1):step_cost,
      (2,2):step_cost,
      (2,3):step_cost,
    })
    return g

def play_game(g):
    pass