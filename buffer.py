import numpy as np
import random
from collections import deque

class DataStorage(object):

	def __init__(self, T, num_new_data, state_dim, action_dim, constraint_dim, window, q):
		self.T = T
		self.window=window
		self.q=q
		self.num_new_data = num_new_data
		self.count = 0
		self.n_entries = 0
		self.constraint_dim = constraint_dim

		self.state_memory = np.zeros((2 * self.T, state_dim))
		self.action_memory = np.zeros((2 * self.T, action_dim))
		self.action_max_memory = np.zeros((2 * self.T, action_dim))
		self.cost_memory = np.zeros((2 * self.T, 1+constraint_dim))
		self.next_state_memory = np.zeros((2 * self.T, state_dim))

		self.state_memory_tmp = np.zeros((self.num_new_data, state_dim))
		self.action_memory_tmp = np.zeros((self.num_new_data, action_dim))
		self.action_max_memory_tmp = np.zeros((self.num_new_data, action_dim))
		self.cost_memory_tmp = np.zeros((self.num_new_data, 1+constraint_dim))
		self.next_state_memory_tmp = np.zeros((self.num_new_data, state_dim))

		self.aver_reward_memory = np.zeros((window, 1))
		self.aver_cost_memory = np.zeros((window, constraint_dim))
		self.aver_reward_memory_tmp = np.zeros((self.num_new_data, 1))
		self.aver_cost_memory_tmp = np.zeros((self.num_new_data, constraint_dim))

	def store_experiences(self, state, action, action_max, costs, next_state, reward):
		a=1
		if self.count < 2 * self.T:
			self.state_memory[self.count] = state
			self.action_memory[self.count] = action
			self.action_max_memory[self.count] = action_max
			self.cost_memory[self.count] = costs
			self.next_state_memory[self.count] = next_state
		else:
			ind = self.count % self.num_new_data
			self.state_memory_tmp[ind] = state
			self.action_memory_tmp[ind] = action
			self.action_max_memory_tmp[ind] = action_max
			self.cost_memory_tmp[ind] = costs
			self.next_state_memory_tmp[ind] = next_state
			if ind == self.num_new_data-1:
				self.state_memory[0: 2 * self.T - self.num_new_data] = self.state_memory[self.num_new_data:]
				self.state_memory[2 * self.T - self.num_new_data:] = self.state_memory_tmp
				self.action_memory[0: 2 * self.T - self.num_new_data] = self.action_memory[self.num_new_data:]
				self.action_memory[2 * self.T - self.num_new_data:] = self.action_memory_tmp
				self.action_max_memory[0: 2 * self.T - self.num_new_data] = self.action_max_memory[self.num_new_data:]
				self.action_max_memory[2 * self.T - self.num_new_data:] = self.action_max_memory_tmp
				self.cost_memory[0: 2 * self.T - self.num_new_data] = self.cost_memory[self.num_new_data:]
				self.cost_memory[2 * self.T - self.num_new_data:] = self.cost_memory_tmp
				self.next_state_memory[0: 2 * self.T - self.num_new_data] = self.next_state_memory[self.num_new_data:]
				self.next_state_memory[2 * self.T - self.num_new_data:] = self.next_state_memory_tmp

		if self.count < self.window:
			self.aver_reward_memory[self.count] = reward
			self.aver_cost_memory[self.count,:] = costs[1:(self.constraint_dim+1)]
		else:
			ind = self.count % self.num_new_data
			self.aver_reward_memory_tmp[ind] = reward
			self.aver_cost_memory_tmp[ind, :] = costs[1:(self.constraint_dim+1)]
			if ind == self.num_new_data-1:
				self.aver_reward_memory[0: self.window - self.num_new_data] = self.aver_reward_memory[self.num_new_data:]
				self.aver_reward_memory[self.window - self.num_new_data:] = self.aver_reward_memory_tmp
				self.aver_cost_memory[0: self.window - self.num_new_data,:] = self.aver_cost_memory[self.num_new_data:,:]
				self.aver_cost_memory[self.window - self.num_new_data:,:] = self.aver_cost_memory_tmp
		self.count += 1

	def take_experiences(self):
		if self.count < self.window:
			return self.state_memory, self.action_memory, self.action_max_memory, self.cost_memory, self.next_state_memory, self.aver_reward_memory[0:self.count], self.aver_cost_memory[0:self.count]
		else:
			return self.state_memory, self.action_memory, self.action_max_memory, self.cost_memory, self.next_state_memory, self.aver_reward_memory, self.aver_cost_memory


class DataStorage_soft(object):

	def __init__(self, T, num_new_data, state_dim, action_dim, constraint_dim, window, q):
		self.T = T
		self.window=window
		self.q=q
		self.num_new_data = num_new_data
		self.count = 0
		self.n_entries = 0
		self.constraint_dim = constraint_dim

		self.state_memory = np.zeros((2 * self.T, state_dim))
		self.action_memory = np.zeros((2 * self.T, action_dim))
		self.action_max_memory = np.zeros((2 * self.T, action_dim))
		self.cost_memory = np.zeros((2 * self.T, 1+constraint_dim))
		self.next_state_memory = np.zeros((2 * self.T, state_dim))

		self.state_memory_tmp = np.zeros((self.num_new_data, state_dim))
		self.action_memory_tmp = np.zeros((self.num_new_data, action_dim))
		self.action_max_memory_tmp = np.zeros((self.num_new_data, action_dim))
		self.cost_memory_tmp = np.zeros((self.num_new_data, 1+constraint_dim))
		self.next_state_memory_tmp = np.zeros((self.num_new_data, state_dim))

		self.aver_reward_memory = np.zeros((window, 1))
		self.aver_cost_memory = np.zeros((window, constraint_dim))
		self.aver_reward_memory_tmp = np.zeros((self.num_new_data, 1))
		self.aver_cost_memory_tmp = np.zeros((self.num_new_data, constraint_dim))

	def store_experiences(self, state, action, action_max, costs, next_state, reward):
		a=1
		if self.count < 2 * self.T:
			self.state_memory[self.count] = state
			self.action_memory[self.count] = action
			self.action_max_memory[self.count] = action_max
			self.cost_memory[self.count] = costs
			self.next_state_memory[self.count] = next_state
		else:
			ind = self.count % self.num_new_data
			self.state_memory_tmp[ind] = state
			self.action_memory_tmp[ind] = action
			self.action_max_memory_tmp[ind] = action_max
			self.cost_memory_tmp[ind] = costs
			self.next_state_memory_tmp[ind] = next_state
			if ind == self.num_new_data-1:
				self.state_memory[0: 2 * self.T - self.num_new_data] = self.state_memory[self.num_new_data:]
				self.state_memory[2 * self.T - self.num_new_data:] = self.state_memory_tmp
				self.action_memory[0: 2 * self.T - self.num_new_data] = self.action_memory[self.num_new_data:]
				self.action_memory[2 * self.T - self.num_new_data:] = self.action_memory_tmp
				self.action_max_memory[0: 2 * self.T - self.num_new_data] = self.action_max_memory[self.num_new_data:]
				self.action_max_memory[2 * self.T - self.num_new_data:] = self.action_max_memory_tmp
				self.cost_memory[0: 2 * self.T - self.num_new_data] = self.cost_memory[self.num_new_data:]
				self.cost_memory[2 * self.T - self.num_new_data:] = self.cost_memory_tmp
				self.next_state_memory[0: 2 * self.T - self.num_new_data] = self.next_state_memory[self.num_new_data:]
				self.next_state_memory[2 * self.T - self.num_new_data:] = self.next_state_memory_tmp

		if self.count < self.window:
			self.aver_reward_memory[self.count] = reward
			self.aver_cost_memory[self.count,:] = costs[1:(self.constraint_dim+1)]

		else:
			ind = self.count % self.num_new_data
			self.aver_reward_memory_tmp[ind] = reward
			self.aver_cost_memory_tmp[ind,:] = costs[1:(self.constraint_dim+1)]
			if ind == self.num_new_data-1:
				self.aver_reward_memory[0: self.window - self.num_new_data] = self.aver_reward_memory[self.num_new_data:]
				self.aver_reward_memory[self.window - self.num_new_data:] = self.aver_reward_memory_tmp
				self.aver_cost_memory[0: self.window - self.num_new_data,:] = self.aver_cost_memory[self.num_new_data:,:]
				self.aver_cost_memory[self.window - self.num_new_data:,:] = self.aver_cost_memory_tmp
		self.count += 1
	def take_experiences(self):
		if self.count < self.window:
			return self.state_memory, self.action_memory, self.action_max_memory, self.cost_memory, self.next_state_memory, self.aver_reward_memory[0:self.count], self.aver_cost_memory[0:self.count]
		else:
			return self.state_memory, self.action_memory, self.action_max_memory, self.cost_memory, self.next_state_memory, self.aver_reward_memory, self.aver_cost_memory
