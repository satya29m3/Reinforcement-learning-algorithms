import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pickle
import time
from matplotlib import style

style.use('ggplot')
# episode_rewards = []

size = 10

episodes  = 20
move_penalty = 1
enemy_penalty = 300
food_reward = 25
epsilon = 0
eps_dec = 0.9998
show_every = 1

start_q_table = "qtable-1564598122.pickle"

lr = 0.1
discount = 0.95


playerN = 1
foodN = 2
enemyN = 3

d = {
    1:(255,175,0),
    2:(0,255,0),
    3:(0,0,255)
}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0,size)
        self.y = np.random.randint(0,size)
    

    def __str__(self):
        return f"{self.x},{self.y}"
    
    def __sub__(self,other):
        return (self.x - other.x , self.y - other.y)
    
    def action(self,choice):

        if choice == 0:
            self.move(x=1,y=1)
        elif choice == 1:
            self.move(x=-1,y=-1)
        elif choice == 2:
            self.move(x=-1,y=1)
        elif choice == 3:
            self.move(x =1 ,y=-1)


    def move(self,x=False,y= False):
        if not x:
            self.x += np.random.randint(-1, 2)
            # self.y += np.random.randint(-1 ,2)
        else:
            self.x += x
        
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > size-1:
            self.x = size-1
        
        
        if self.y < 0:
            self.y = 0
        elif self.y > size-1:
            self.y = size-1

# making qtable


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-size+1, size):
        for ii in range(-size+1, size):
            for iii in range(-size+1, size):
                    for iiii in range(-size+1, size):
                        q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards =[]

for episode in range(episodes):
    player  = Blob()
    food = Blob()
    enemy = Blob()


    if episode % show_every == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{show_every} ep mean: {np.mean(episode_rewards[-show_every:])}")
        show = True
        
    else :
        show = False
    
    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        # print(obs)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
        player.action(action)
        food.move()
        enemy.move()

        if player.x == enemy.x  and player.y == enemy.y:
            reward  = -enemy_penalty
        elif player.x == food.x and player.y == food.y:
            reward = food_reward
        else : 
            reward = -move_penalty

        new_obs = (player-food,player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == food_reward:
            new_q = food_reward
        else:
            new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)
        
        q_table[obs][action] = new_q
        # visualize the environment
        if show == True:
            env = np.zeros((size,size,3), dtype = np.uint8)
            env[food.x][food.y] = d[foodN]
            env[player.x][player.y] = d[playerN]
            env[enemy.x][enemy.y] = d[enemyN]
            img = Image.fromarray(env , 'RGB')
            img = img.resize((300,300))
            cv2.imshow("image",np.array(img))
            if reward == food_reward or reward == -enemy_penalty:
                if cv2.waitKey(500) & 0xFF == 'q':
                    break
            else:
                if cv2.waitKey(1) & 0xFF == 'q':
                    break

        episode_reward += reward
        if reward == food_reward or reward == -enemy_penalty:
            break
        
    episode_rewards.append(episode_reward)
    epsilon *= eps_dec


moving_avg = np.convolve(episode_rewards, np.ones((show_every,))/show_every, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {show_every}ma")
plt.xlabel("episode #")
plt.show()
if start_q_table is None:
    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)