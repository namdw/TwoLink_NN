#!/usr/bin/python

"""
	from TwoLink_sim import *

	place the TwoLink_sim.py in the working folder and 
	import the simulator as above to use


	list of current function

	TwoLink() : create the object
	move_link1(x) : move link 1 in x radians
	move_link2(x) : move link 2 in x radians
	getAngles(x) : return array of current joint angles in radians

	examples are below

	2016/08/20
	dnjsxodp@gmail.com
"""

from TwoLink_sim import * # add this linke to use the simulator
import time
import numpy as np
import random
import math
import pickle
import os.path
import base 

# Variables
TRAINING = True
numTrain = 5
cntrl_freq = 100

goal = [150,100]

num_epoch = 3

if(TRAINING):
	epsilon = 0.3
else:
	epsilon = 0.0

max_speed = 2

filename1 = "test_net_2states.p"

if os.path.isfile(filename1):
	f = open(filename1,'rb')
	q_net = pickle.load(f)
	f.close()
else:
	q_net = base.NN(4,2,[64,128,64], func='lrelu', dropout=0.8, weight='xavier')


# Function
# def getAction(net, state):
# 	for i in range(len(actionList)):
# 		value = net.forward(state+actionList[i])
# 		valList[i] = value
# 	# print(valList)
# 	indices = [i for i, x in enumerate(valList) if x == max(valList)]
# 	min_action = math.inf
# 	min_indices = []
# 	for idx in indices:
# 		if sum(actionList[idx]) <= min_action:
# 			min_action = sum(actionList[idx])
# 	for idx in indices:
# 		if sum(actionList[idx]) == min_action:
# 			min_indices += [idx]
# 	maxIndex = random.choice(min_indices)
# 	# print(valList)
# 	# print(valList[maxIndex])
# 	# time.sleep(1)
# 	print(valList[maxIndex])
# 	return actionList[maxIndex]


def egreedyExplore(net, state, epsilon):
	value = net.forward(state)
	rand = 0
	if random.random() < epsilon:
		# print('random choice')
		action = (value+np.random.normal(0,max_speed/2.0,value.size))[0]
		# action = (2*max_speed*np.random.random(value.size))-max_speed
		rand = 1
	else:
		action = value[0]
	for i, a in enumerate(action):
		action[i] = min(a,max_speed)
		rand = 1
	return action, rand

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# Create the simulator object
sim = TwoLink()

# Start and display the Simulator graphics
sim.show()

time.sleep(1)

max_reward = -10000
LR = 0

for numTry in range(numTrain):
	# print("Trial #",numTry)
	sim.reset()
	counter = 0
	# sim.randGoal()
	sim.makeGoal(goal)
	state = sim.getState()
	state = [st/180.0*math.pi for st in state]
	state2 = [sim.getVert(), sim.getHorz()]
	action = [0,0]
	stateVal = sim.getStateVal()

	state_list = []
	action_list = []
	rand = 0
	while(abs(stateVal)>5 and counter<300):
		counter = counter + 1
		# Trainig phase
		if(TRAINING):
			reward = sim.getStateVal()-stateVal
			if(reward > 0 and rand):
				if(reward>max_reward):
					max_reward = reward
				LR = max(0,0.002/(1+exp(-reward/max_reward))+0.001)
				# LR = max(0,0.1*reward/max_reward)
			elif(reward <= 0 and not rand):
				for i in range(len(action)):
					action[i] = 0.0 # rather do nothing
				LR = 0.001
			# for k in range(num_epoch):
			q_net.train(state+state2, action, LR)
			state_list.append(state+state2)
			action_list.append(action)
		elif(not TRAINING):
			print(action)
		stateVal = sim.getStateVal()

		# Update
		state = sim.getState()
		state = [st/180.0*math.pi for st in state]
		state2 = [sim.getVert(), sim.getHorz()]
		action, rand = egreedyExplore(q_net, state+state2, epsilon)

		# Apply the actions
		sim.move_link1(action[0]/cntrl_freq)
		sim.move_link2(action[1]/cntrl_freq)
		time.sleep(1/cntrl_freq)
	if(counter<300):
		# success train once more
		epsilon = epsilon * 0.9
		for i in range(len(state_list)):
			for _ in range(num_epoch):
				q_net.train(state_list[i], action_list[i], 0.001)
	else:
		epsilon = min(0.3,epsilon*1.1)
	print("Trial #",numTry," Score:",300-counter)


print("done exploring")

if(TRAINING):
	f = open(filename1,'wb')
	pickle.dump(q_net,f)
	f.close()

	print("done with pickle")

# if __name__ == '__main__':
#     sys.exit(main())