#!/usr/bin/python

import gym

class Environment():
	# Constructor
	def __init__(self):
		self.env = gym.make('LunarLander-v2')
	#env.reset()
	