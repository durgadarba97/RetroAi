"""
How to get the atari games
https://anaconda.org/conda-forge/gymnasium-atari

run : python3 ./stable_retro.py 
"""

import gymnasium as gym
import pandas as df
import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime


class Game:
    def __init__(self):
        # self.env = gym.make('Tennis-ram-v0', render_mode='human')
        # self.env = gym.make('Pong-ramDeterministic-v0', render_mode='human')
        
        self.env = gym.make('Boxing-ram-v0')
        # self.env = gym.make('ALE/Galaxian-ram-v5', render_mode='human')
        self.state = self.env.reset()
        self.done = False

        

    def getAllGames(self):
        return gym.envs.registry.keys()
    
    def randomAction(self):
        return self.env.action_space.sample()
    
    
    def reset(self):
        self.state = self.env.reset()
        self.done = False
        return self.state
    
    def _testRandom(self):
        for i in range(100):
            action = self.randomAction()
            # print(action)
            observation, reward, self.done, _,_ = self.env.step(action)

            self.env.render()
            if self.done:
                self.reset()

        # print(f"obersevation: {observation}\ntype:{type(observation)}")

    def run(self, learner, num_episodes=10):
        # measurements = df.DataFrame(columns=['observation', 'reward', 'done',])
        measurements = []
        total_reward = 0.0

        print(f"Starting game... Total episodes: {num_episodes}")
        for i in range(num_episodes):
            episode_reward = 0.0
            start = time.time()
            while not self.done:
                # make the make the action
                type(self.state)
                action = learner.getAction(self.state)
                observation, reward, self.done, truncated, info = self.env.step(action)
                # print(f"observation: {observation}, reward: {reward}, done: {self.done}, truncated: {truncated}, info: {info}")

                # learn and update the state
                # print(type(observation))
                learner.learn(reward, observation, self.env)
                self.state = observation
                
                # take measurements of how the learner is performing
                total_reward += reward
                episode_reward += reward
                measurements.append([i, observation, reward, self.done, total_reward, episode_reward, learner.rar, learner.radr, learner.Q.shape[0], learner.Q.shape[1]])
                

            print(f"Episode {i} finished, Reward: {total_reward}, episode reward: {episode_reward}, time: {(time.time() - start)/60.0} minutes")

            self.reset()

        self.env.close()
        return measurements
            


# self.Q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.Q[s_prime]) - self.Q[self.s, self.a])
class Qlearner:


    def __init__(self, num_states=256, num_actions=5, alpha=0.01, gamma=0.9, rar=0.99, radr=0.9999999999999, dyna=1):  		  	   		 	   			  		 			 	 	 		 		 	
        """  		  	   		 	   			  		 			 	 	 		 		 	
        Constructor method  		  	   		 	   			  		 			 	 	 		 		 	
        """   	   		 	   			  		 			 	 	 		 		 		   		 	   			  		 			 	 	 		 		 		   		 	   			  		 			 	 	 		 		 	
        self.s = 0  		  	   		 	   			  		 			 	 	 		 		 	
        self.a = 0  

        self.rar = rar
        self.radr = radr		
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.num_states = num_states
        self.num_actions = num_actions

        # Initialize Q table
        # self.Q = self.load("boxer.h5")
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.experience = []
        self.start = time.time()

    
    # s_prime = np.digitize(s_prime[0], np.linspace(0, 255, 6)) - 1
    def discretize(self, s_prime):
        # Dirtizing mean converting the continous state space to a discrete state space
        # for all the obsrvations in the state space, 
        # each thing you observe will be placed in a "bin"
        # th bin is the index of the observation in the linspace

        # print(f"state: {s_prime}, type: {type(s_prime)}\n\n s_prime[0]: {s_prime[0]}, type: {type(s_prime[0])}")
        return np.digitize(s_prime[0], np.linspace(0, 255, self.num_actions) - 1) 

    def getAction(self, s_prime):

        # decaay the randonm action rate by the random action decay rate. 
        #  This needs to run for 1000s of generations before it can start applying 
        # s_prime = self.discretize(s_prime)
        self.rar = max(self.rar * self.radr, 0.30)

        if(np.random.rand() < self.rar):
            # Exploration: choose a random action
            # print("Exploration")
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: choose the best action based on current knowledge
            # print("Exloitation")
            try:
                print(type(s_prime))
                return np.clip(np.argmax(self.Q[s_prime]), 0, self.num_actions - 1)
            except:
                print(f"Q: {self.Q.shape}, s_prime: {s_prime}")
                print(f"sp rime shape: {s_prime.shape}")
            # return np.argmax(self.Q[s_prime])
        



    def learn(self, reward, s_prime, next_obs):
        s_prime = self.discretize(s_prime)

        # Update Q table
        self.Q[self.s, self.a] = self.Q[self.s, self.a] + self.alpha * (reward + self.gamma * np.max(self.Q[s_prime]) - self.Q[self.s, self.a])

        # Dyna
        if self.dyna > 0:
            self.experience.append((self.s, self.a, reward, s_prime))
            for _ in range(self.dyna):
                s, a, r, s_prime = self.experience[np.random.choice(len(self.experience))]
                self.Q[s, a] = self.Q[s, a] + self.alpha * (r + self.gamma * np.max(self.Q[s_prime]) - self.Q[s, a])

        # Update state and action
        self.s = s_prime
        print(type(s_prime))
        self.a = self.getAction(s_prime)
            

    
    def save(self):
        # save the q table
        filename = "tashi2.h5"
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("Q", data=self.Q)

    def load(self, infile="test.h5"):
        f = h5py.File(infile, 'w')
        if(f.get('Q') is None):
            return np.zeros((self.num_states, self.num_actions))
        return f.get('Q')
    

class Test:
    def setup():   
        g = Game()
        l = Qlearner()
        return g, l
    
    def t_run():
        g, l = Test.setup()
        start_time = datetime.now()
        measurements = g.run(l, num_episodes=100)
        l.save()
        measurements = df.DataFrame(measurements, columns=['episode', 'observation', 'reward', 'done', "total_reward", "episode_reward", "rar", "radr", "Q_rows", "Q_cols"])
        # get time now

        time_delta = datetime.now() - start_time
        # total reward at each episode
        print(f"final stats {str(time_delta)}\ntotal reward {measurements['total_reward'].iloc[-1]}, episode reward: {measurements['episode_reward'].iloc[-1]} rar: {measurements['rar'].iloc[-1]}, radr: {measurements['radr'].iloc[-1]}, Q Size: {measurements["Q_rows"]}x{measurements["Q_cols"]},")
        # plot measurements )
        plt.figure("Rewards")
        plt.plot(measurements['episode'], measurements['total_reward'], label='Total Reward')   
        plt.legend()
        plt.show()

        # moving average of the rewards
        plt.figure("Episode Reward")
        reward_moving_avg = measurements['episode_reward'].rolling(window=10).mean()
        plt.plot(measurements['episode'], reward_moving_avg, label=' Avg Episode Reward')
        # plt.plot(measurements['episode'], measurements['rar'], label='rar')
        # plt.plot(measurements['episode'], measurements['radr'], label='radr')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    Test.t_run()