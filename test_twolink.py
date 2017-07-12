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
# tryNumber = range(1,1,100) # how many try
numTrain = 10
cntrl_freq = 100
goal = [150,100]

num_epoch = 3

epsilon = 0.1
actionList = []
actionX = range(-2,3)
actionY = range(-2,3)
actionList = []
for x in actionX:
	for y in actionY:
		actionList.append([x,y])
valList = len(actionList)*[0]
numState = 3
filename1 = "test_net_xavier.p"

if os.path.isfile(filename1):
	f = open(filename1,'rb')
	q_net = pickle.load(f)
	f.close()
else:
	q_net = base.NN(6,1,[128,256,128], func='lrelu', dropout=0.8, weight='xavier')



# Function
def getAction(net, state):
	for i in range(len(actionList)):
		value = net.forward(state+actionList[i])
		valList[i] = value
	# print(valList)
	indices = [i for i, x in enumerate(valList) if x == max(valList)]
	maxIndex = random.choice(indices)
	# print(valList)
	# print(valList[maxIndex])
	# time.sleep(1)
	return actionList[maxIndex]


def egreedyExplore(net, state, epsilon):
	if random.random() < epsilon:
		# print('random choice')
		return random.choice(actionList)

	else:
		return getAction(net, state)

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# Create the simulator object
sim = TwoLink()

# Start and display the Simulator graphics
sim.show()

time.sleep(1)


for numTry in range(numTrain):
	print(numTry)
	sim.reset()
	# sim.makeGoal(goal)
	sim.randGoal()
	state = sim.getState()
	state = [st/180.0*math.pi for st in state]
	state2 = [sim.getVert(), sim.getHorz()]
	action = [0,0]
	stateVal = sim.getStateVal()
	for i in range(200):
		reward = sim.getStateVal()-stateVal
		# print('{0:.4f}'.format(reward/100.0), '{0:.4f}'.format(q_net.forward(state+state2+action)[0][0]))
		# print('{0:.4f}'.format(reward/100.0 - q_net.forward(state+state2+action)[0][0]))
		stateVal = sim.getStateVal()
		for k in range(num_epoch):
			q_net.train(state+state2+action, reward, 0.01)

		# Command the first link to move delta_angle
		state = sim.getState()
		state = [st/180.0*math.pi for st in state]
		state2 = [sim.getVert(), sim.getHorz()]
		action = egreedyExplore(q_net, state+state2, epsilon)
		# print(state, action, q_net.forward(state+action), stateVal)

		sim.move_link1(action[0]/cntrl_freq)
		sim.move_link2(action[1]/cntrl_freq)
		time.sleep(1/cntrl_freq)


print("done exploring")

f = open(filename1,'wb')
pickle.dump(q_net,f)
f.close()

print("done with pickle")

# if __name__ == '__main__':
#     sys.exit(main())