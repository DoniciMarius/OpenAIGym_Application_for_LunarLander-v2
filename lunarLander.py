#!/usr/bin/python

import gym
import random
import os
import csv

class LunarLander():
	
	def __init__(self, alpha, gamma, epsilon, number_of_episodes, number_of_trials):
		self.env = gym.make("LunarLander-v2")
		self.Q_matrix = {}
		self.Q_0 = 0
		self.alpha = alpha 
		self.gamma = gamma
		self.epsilon = epsilon
		self.number_of_episodes = number_of_episodes
		self.number_of_trials = number_of_trials
		# derive action space
		self.action_space = []
		for noa in range(int(repr(self.env.action_space)[::-1][1:2])):
			self.action_space.append(noa)
		self.count_equal_states = 0 # XXX
		self.cumulative_reward = 0 # XXX
		self.filename = os.path.join("/Users/matthiaswettstein/CloudStation/Hack/Python/Udacity/MLE/06P_Capstone/01_Material", "q_matrix.csv") # XXX
			
	def best_action_select(self, state=None, epsilon=None):
		# initialize state tuple key in dictionary
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
		
	def train(self):
		self.env.reset()
		for e in range(self.number_of_episodes):
			print "\nEpisode {}".format(e)
			self.state = self.env.reset() # random first obs
			self.state = tuple(self.state) # convert ndarray to tuple
			for t in range(self.number_of_trials):
				self.env.render()
				# Memorize state for the later update of the Q matrix
				state_t0 = self.state
				#action = env.action_space.sample() # take a random action
				# take action
				action = self.best_action_select(state=self.state, epsilon=self.epsilon)
				# Learn policy based on state, action, reward
				''' Q-learning
				- Evaluate Bellman equations from data: Estimate Q from transitions
					- (1) Get the Q value (Q(s_t, a_t)) from the action at time t
					- (2) Update the state after the first action
					- (3) Perform the action at time t+1
					- (4) Get the Q value (Q_s_t+1, a_t+1)) from the second action
					- (5) Update the Q matrix for the first state and action, in hindsight '''
				Q_t0 = self.Q_matrix[self.state][action] ## (1)
				self.state, reward, done, info = self.env.step(action) ## (2)
				self.state = tuple(self.state) # convert ndarray to tuple
				action_t1 = self.best_action_select(state=self.state, epsilon=self.epsilon) ## (3)
				Q_t1 = self.Q_matrix[self.state][action_t1] ## (4)
				self.Q_matrix[state_t0][action] = Q_t0 + self.alpha * (reward + self.gamma * Q_t1 - Q_t0) ## (6)
				# XXX
				self.cumulative_reward += reward
				for k, v in self.Q_matrix.iteritems():
					for w, u in self.Q_matrix[k].iteritems():
						checksum = 0
						if not u == 0:
							checksum += 1
						if checksum > 1:
							self.count_equal_states += 1
				if done:
					print len(self.Q_matrix)
					print self.count_equal_states
					print self.cumulative_reward
					print("Episode finished after {} timesteps".format(t+1))
					break
				# write out Q matrix at the end
				if e == self.number_of_episodes-1:
					#print self.Q_matrix
					with open(self.filename,"wb") as f:
						w = csv.writer(f)
						w.writerows(self.Q_matrix.items())
	
def main():
	l = LunarLander(0.1, 0.8, 0.01, 1, 1000)
	l.train()
	
if __name__ == '__main__':
	main()