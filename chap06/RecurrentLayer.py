#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-01 15:20:01
import numpy as np
from ElementWiseOP import elements_wise_op as ewop

import inspect

def printFrame():
	callerFrameRecord = inspect.stack()[1] # 0 represents this line
											# 1 represents line at caller
	frame = callerFrameRecord[ 0 ]
	info = inspect.getframeinfo( frame )
	print "__FILE__", info.filename
	print "__FUNC__", info.function
	print "__LINE__", info.lineno

class RecurrentLayer(object):
	def __init__(self, input_width, state_width, activator, learning_rate):
		self.input_width = input_width
		self.state_width = state_width
		self.activator = activator
		self.learning_rate = learning_rate
		self.times = 0
		self.state_list = []
		self.state_list.append((np.zeros((state_width, 1))))
		#print self.state_list[ -1 ]
		self.U = np.random.uniform(-0.0001, 0.0001,
									(state_width, input_width))
		#print self.U
		self.W = np.random.uniform(-0.0001, 0.0001,
									(state_width, state_width))
		#print self.W

	def forward(self, input_array):
		self.times += 1
		#print input_array
		#print np.dot( self.U, input_array )
		#print np.dot( self.W, self.state_list[ -1 ] )
		#print "state"
		state = (np.dot(self.U, input_array) +
				 np.dot(self.W, self.state_list[-1]))
		#print state
		ewop.element_wise_op(state, self.activator.forward)
		self.state_list.append(state)

	def backward(self, sensitivity_map, activator):
		self.calc_delta(sensitivity_map, activator)
		self.calc_gradient()

	def calc_delta(self, sensitivity_map, activator):
		self.delta_list = []
		for i in range(self.times):
			self.delta_list.append(np.zeros( (self.state_width, 1 )))
		self.delta_list.append(sensitivity_map)

		for k in range(self.times - 1, 0, -1):
			self.calc_delta_k(k, activator)

	def calc_delta_k(self, k, activator):
		state = self.state_list[k + 1].copy()

		printFrame()
		print state

		# ewop.element_wise_op( self.state_list[k + 1], activator.backward )
		ewop.element_wise_op( state, activator.backward )
		print self.state_list[ k + 1 ]
		print np.diag( state[ :, 0 ] )
		self.delta_list[k] = np.dot(np.dot(self.delta_list[k + 1].T,
										   self.W),
									np.diag(state[:, 0])).T

	def calc_gradient(self):
		self.gradient_list = []
		for t in range(self.times + 1):
			self.gradient_list.append(
				np.zeros((self.state_width, self.state_width)))
		for t in range(self.times, 0, -1):
			self.calc_gradient_t(t)
		self.gradient = reduce(lambda a, b: a + b, self.gradient_list,
							   self.gradient_list[0])

	def calc_gradient_t(self, t):
		gradient = np.dot(self.delta_list[t], self.state_list[t - 1].T)
		self.gradient_list[t] = gradient

	def update(self):
		self.W -= self.learning_rate * self.gradient
	def reset_state(self):
		self.times = 0       # 当前时刻初始化为t0
		self.state_list = [] # 保存各个时刻的state
		self.state_list.append(np.zeros(
			(self.state_width, 1)))      # 初始化s0
