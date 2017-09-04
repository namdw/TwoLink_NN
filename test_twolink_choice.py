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
TRAINING = False
numTrain = 10
RUNTIME = 500
cntrl_freq = 100

goal = [100,150]

num_epoch = 5

max_speed = 2

filename1 = "test_net_build.p"

if os.path.isfile(filename1):
	f = open(filename1,'rb')
	q_net = pickle.load(f)
	f.close()
else:
	input_layer = base.Layer('input',3)
	hidden_layer1 = base.Layer('hidden',128,func='lrelu',dropout=0.8,weight_scale=5.0,optimizer='ADAM')
	hidden_layer2 = base.Layer('hidden',256,func='lrelu',dropout=0.8,weight_scale=5.0,optimizer='ADAM')
	hidden_layer3 = base.Layer('hidden',128,func='lrelu',dropout=0.8,weight_scale=5.0,optimizer='ADAM')
	output_layer = base.Layer('output',2,dropout=0.8,weight_scale=10.0,optimizer='ADAM')
	q_net = base.NNb()
	q_net.addLayer(input_layer)
	q_net.addLayer(hidden_layer1)
	q_net.addLayer(hidden_layer2)
	q_net.addLayer(hidden_layer3)
	q_net.addLayer(output_layer)

# print(q_net.W)
# Create the simulator object
sim = TwoLink()

# Start and display the Simulator graphics
sim.show()

time.sleep(1)

LR = 0.01

state_list = []
action_list = []
for numTry in range(numTrain):
	# print("Trial #",numTry)
	# sim.reset()
	counter = 0

	sim.randGoal()
	# sim.makeGoal(goal)
	state = sim.getState()
	# state = [st/180.0*math.pi for st in state]
	state = [state[0]]
	state2 = [sim.getVert(), sim.getHorz()]
	state = state+state2
	action = [0,0]
	stateVal = sim.getStateVal()
	init_dist = sim.getStateVal()

	stay_count = 0
	goal_flag = 0
	while(not(goal_flag==1 and stay_count>100)):
		if(abs(stateVal)<5):
			goal_flag = 1
		if(goal_flag==1):
			stay_count+=1

		# Update to currenrt state
		stateVal = sim.getStateVal()
		state = sim.getState()
		# state = [st/180.0*math.pi for st in state]
		state = [state[0]]
		state2 = [sim.getVert(), sim.getHorz()]
		state = state+state2

		# Current action
		action = q_net.forward(state)[0]
		
		# Apply the actions
		counter+=1
		sim.move_link1(action[0]/cntrl_freq)
		sim.move_link2(action[1]/cntrl_freq)
		time.sleep(1/cntrl_freq)
		
		# Trainig phase
		if(TRAINING):
			# reward of current action
			reward = sim.getStateVal()-stateVal
			# Move back
			counter+=1
			sim.move_link1(-action[0]/cntrl_freq)
			sim.move_link2(-action[1]/cntrl_freq)
			time.sleep(1/cntrl_freq)
			
			# random action trial
			rand_action = (2*max_speed*np.random.random(action.size))-max_speed
			# Move to random action
			counter+=1
			sim.move_link1(rand_action[0]/cntrl_freq)
			sim.move_link2(rand_action[1]/cntrl_freq)
			time.sleep(1/cntrl_freq)

			rand_reward = sim.getStateVal()-stateVal
			
			if(rand_reward > reward):
				state_list.append(state)
				action_list.append(rand_action)
				# for k in range(num_epoch):
				# 	q_net.train(state, rand_action, LR)
			else:
				# Move back
				counter+=1
				sim.move_link1(-rand_action[0]/cntrl_freq)
				sim.move_link2(-rand_action[1]/cntrl_freq)
				time.sleep(1/cntrl_freq)

				# Move back
				counter+=1
				sim.move_link1(action[0]/cntrl_freq)
				sim.move_link2(action[1]/cntrl_freq)
				time.sleep(1/cntrl_freq)
		else:
			print(action)

	if(counter!=0):
		print("Trial #",numTry," Score:",-init_dist/counter)

if(len(state_list)!=0):
	rand_order = np.random.permutation(len(state_list))
	for idx in rand_order:
		for _ in range(num_epoch):
			q_net.train(state_list[idx], action_list[idx], LR)

print("done exploring")

if(TRAINING):
	f = open(filename1,'wb')
	pickle.dump(q_net,f)
	f.close()

	print("done with pickle")

# if __name__ == '__main__':
#     sys.exit(main())