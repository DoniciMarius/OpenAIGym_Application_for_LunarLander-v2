#!/usr/bin/python

import gym
import random
import os
import csv

class LunarLander():
	
	def __init__(self, alpha, gamma, epsilon, number_of_episodes, number_of_trials, rounding_factor, state_length):
		self.env = gym.make("LunarLander-v2")
		self.Q_matrix = {}
		self.Q_0 = 0
		self.alpha = alpha 
		self.gamma = gamma
		self.epsilon = epsilon
		self.number_of_episodes = number_of_episodes
		self.number_of_trials = number_of_trials
		# Derive action space
		self.action_space = []
		for noa in range(int(repr(self.env.action_space)[::-1][1:2])):
			self.action_space.append(noa)
		self.filename = os.path.join("/Users/matthiaswettstein/CloudStation/Hack/Python/Udacity/MLE/06P_Capstone/01_Material", "q_matrix.csv") # Init file save
		# Naive state reduction
		self.rounding_factor = rounding_factor 
		self.state_length = state_length 
		self.best_episode_cumulative_reward = -1000 #XXX
			
			
	def best_action_select(self, state=None, epsilon=None):
		# Initialize state tuple key in dictionary
		if state not in self.Q_matrix:
			action_reward = {}			
			for action in self.action_space:
				action_reward[action] = self.Q_0
				self.Q_matrix[state] = action_reward
			best_action = random.choice(self.action_space)
		else:
			if random.uniform(0,1) <= epsilon:
				best_action = random.choice(self.action_space)
			else: 
				best_actions = [w[0] for w in self.Q_matrix[state].iteritems() if w[1] == max([v[1] for v in self.Q_matrix[state].iteritems()])]
				if len(best_actions) == 1:
					best_action = best_actions[0]
				else:
					best_action = random.choice(best_actions)         
		return best_action 
		
		
	def create_state(self, state=None, rounding_factor=None, state_length=None):
		modified_state = tuple([round(s, rounding_factor) for s in state])[0:state_length]
		return modified_state
	
	
	def train(self):
		self.env.reset() # Reset environment
		
		for e in range(self.number_of_episodes):
			episode_cumulative_reward = 0 # Reset enviroment reward
			self.state = self.create_state(self.env.reset(), self.rounding_factor, self.state_length)
			
			for t in range(self.number_of_trials):
				self.env.render()
				# Memorize state for the later update of the Q matrix
				state_t0 = self.state
				action = self.best_action_select(state=self.state, epsilon=self.epsilon) # Take action
				# Learn policy based on state, action, reward
				''' Q-learning
				- Evaluate Bellman equations from data: Estimate Q from transitions
					- (1) Get the Q value (Q(s_t, a_t)) from the action at time t
					- (2) Update the state after the first action
					- (3) Perform the action at time t+1
					- (4) Get the Q value (Q_s_t+1, a_t+1)) from the second action
					- (5) Update the Q matrix for the first state and action, in hindsight '''
				Q_t0 = self.Q_matrix[self.state][action] ## (1)
				state, reward, done, info = self.env.step(action) ## (2)
				self.state = self.create_state(state, self.rounding_factor, self.state_length)
				action_t1 = self.best_action_select(state=self.state, epsilon=self.epsilon) ## (3)
				Q_t1 = self.Q_matrix[self.state][action_t1] ## (4)
				self.Q_matrix[state_t0][action] = Q_t0 + self.alpha * (reward + self.gamma * Q_t1 - Q_t0) ## (5)
				episode_cumulative_reward += reward # Sum up rewards
					
				if done:
					# Check for best reward ever
					if episode_cumulative_reward >= self.best_episode_cumulative_reward:
						self.best_episode_cumulative_reward = episode_cumulative_reward
					# Control state space explosion
					count_equal_states = 0 
					for k in self.Q_matrix.iterkeys(): 
						checksum = 0
						for w, u in self.Q_matrix[k].iteritems():
							if not u == 0:
								checksum += 1
						if checksum > 1:
							count_equal_states += 1
					
					# Show performance metrics
					print " \nEpisode {} \
							\nEpisode Reward: {} \
							\nBest Episode Reward: {} \
							\nEpisode finished after {} timesteps \
							\n{} out of {} Q-matrix states are repeatedly used ({percent:.2%})".format(e, \
						episode_cumulative_reward, \
						self.best_episode_cumulative_reward, \
						t+1, \
						count_equal_states, \
						len(self.Q_matrix), \
						percent=float(count_equal_states)/len(self.Q_matrix))
					break
					
				# write out Q matrix at the end
				if e == self.number_of_episodes-1:
					with open(self.filename,"wb") as f:
						w = csv.writer(f)
						w.writerows(self.Q_matrix.items())
	
def main():
	l = LunarLander(0.1, 0.8, 0.01, 2, 1000, 1, 8) # 100, 1000, maximum digits are 15, maximum length is 8
	l.train()
	
if __name__ == '__main__':
	main()