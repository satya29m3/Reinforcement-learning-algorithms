import matplotlib.pyplot as plt
import numpy as np
import gym
env = gym.make('MountainCar-v0')
# env.reset()

# q learning settings
LR = 0.1
discount = 0.95
episodes = 25000
show_every = 3000
stats_every = 100


epsilon = 1
start = 1
end = episodes//2
ep_dec_val = episodes/(end-start)
ep_rewards = []
ag_ep_reward = {'ep':[] , 'avg':[] , 'max':[] , 'min':[]}




discrete_os_size = [40]* len(env.observation_space.high)
discrete_win_os_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size
q_table = np.random.uniform(low = -2, high = 0 , size = (discrete_os_size + [env.action_space.n]))

def get_discrete_State(state):
    discrete_state = (state - env.observation_space.low)/discrete_win_os_size
    return tuple(discrete_state.astype(np.int))






for episode in range(episodes):

    eps_reward =0
    discrete_state = get_discrete_State(env.reset())
    done = False
    if episode % show_every == 0:
        render =True
        print(episode)
        
    else :
        render = False
    


    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        
        
        new_state,reward , done ,_ = env.step(action)
        eps_reward += reward
        new_discrete_state = get_discrete_State(new_state)
        # env.render()
        # new_q = (1-LR)*old_q + (LR * (reward + discount*max_future_q))
        if episode % show_every == 0:
            env.render()

        
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q =(1 -LR)*current_q + LR*(reward + discount*max_future_q)

            q_table[discrete_state + (action,)] = new_q
        
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
        
        discrete_state = new_discrete_state
        
    ep_rewards.append(eps_reward)
    if not episode % stats_every:
        average_reward = sum(ep_rewards[-stats_every:])/stats_every
        ag_ep_reward['ep'].append(episode)
        ag_ep_reward['avg'].append(average_reward)
        ag_ep_reward['max'].append(max(ep_rewards[-stats_every:]))
        ag_ep_reward['min'].append(min(ep_rewards[-stats_every:]))

    if end >= episode >= start:
        epsilon -= ep_dec_val


env.close()
plt.plot(ag_ep_reward['ep'],ag_ep_reward['avg'],label = 'average_reward')
plt.plot(ag_ep_reward['ep'],ag_ep_reward['max'], label ='mac reward')
plt.plot(ag_ep_reward['ep'],ag_ep_reward['min'],label = ' min reward')
plt.legend(loc = 4)

plt.grid(True)
plt.show()