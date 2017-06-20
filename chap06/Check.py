#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-01 14:55:23
from RecurrentLayer import RecurrentLayer
from IdentityActivator import IdentityActivator
import numpy as np

def gradient_check():
	error_function = lambda o: o.sum()
	rl = RecurrentLayer(3, 2, IdentityActivator(), 0.001)
	# x, d = data_set()
	x = list()
	x.append( [
		[1],
		[2],
		[3]] )
	x.append( [
		[4],
		[5],
		[6] ] )
	x.append( [
		[7],
		[8],
		[9] ] )
	rl.forward(x[0])
	rl.forward(x[1])
	# rl.forward(x[2])

	sensitity_array = np.ones(rl.state_list[-1].shape,
							  dtype=np.float64)
	print sensitity_array
	rl.backward(sensitity_array, IdentityActivator())

	# check gradient
	epsilon = 10e-4
	for i in range( rl.W.shape[ 0 ] ):
		for j in range( rl.W.shape[ 1 ] ):
			rl.W[i, j] += epsilon
			rl.reset_state()
			rl.forward( x[ 0 ] )
			rl.forward( x[ 1 ] )
			# rl.forward( x[ 2 ] )
			err1 = error_function( rl.state_list[ -1 ] )
			rl.W[ i, j ] -= 2*epsilon
			rl.reset_state()
			rl.forward( x[ 0 ] )
			rl.forward( x[ 1 ] )
			# rl.forward( x[ 2 ] )
			err2 = error_function( rl.state_list[ -1 ] )
			expect_grad = ( err1 - err2 ) / ( 2*epsilon )
			rl.W[ i, j ] += epsilon
			print 'weights(%d,%d): expected - actural %f - %f' % (
					i, j, expect_grad, rl.gradient[ i, j] )
