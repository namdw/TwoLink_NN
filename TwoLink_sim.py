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
	

	2016/08/20
	dnjsxodp@gmail.com
"""

from tkinter import *
from tkinter import ttk
import time
import threading
from math import *
import random


# def _create_circle(self, x, y, r, **kwargs):
# 	return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)

degperpi = 57.296

ground_pos = (300,300)
joint_pos = (0,0)
gripper_pos = [0,0]
link1_len = 100
link2_len = 100
w = ''
link1 = ''
link2 = ''
joint = ''
gripper = ''
goal_point = ''
angle1 = 0
angle2 = 0
thread1 = ''
thread2 = ''
max_speed = 10
cntrl_freq = 200
#link1_limit = [-90,90]
#link2_limit = [-150,30]

initAngle1 = 30/degperpi
initAngle2 = -90/degperpi

posStateDivider = 5
angleStateDivider = 1

def sign(num):
	if num > 0:
		return 1
	elif num < 0:
		return -1
	else:
		return 0	

class myThread(threading.Thread):
	def __init__(self, func, name):
		threading.Thread.__init__(self)
		self.func = func
		self.name = name

	def run(self):
		self.func()
		print(self.name, 'thread done')

class TwoLink(object):
	def __init__(self):
		self.__run_control = False
		self.__in_control = False
		self.__root = ''

		self.angle1 = initAngle1
		self.angle2 = initAngle2
		self.target_angle1 = initAngle1
		self.target_angle2 = initAngle2
		self.goal = self.getEndpoint()
	
	def reset(self):
		self.angle1 = initAngle1
		self.angle2 = initAngle2
		self.target_angle1 = initAngle1
		self.target_angle2 = initAngle2

		temp = self.getEndpoint()
		x = temp[0]
		y = temp[1]
		delta_x = x - self.goal[0]
		delta_y = self.goal[1] - y
		w.move(goal_point, delta_x, delta_y)
		self.goal = [x,y]

	def show(self):
		global thread1, thread2
		thread1 = myThread(self.show_sub, 'display')
		thread1.start()
		time.sleep(0.1)
		self.__run_control = True
		thread2 = myThread(self.controlLoop, 'control')
		thread2.daemon = True
		thread2.start()

	def getAngles(self):
		return [self.angle1, self.angle2]
		
	def getState(self):
		angleState1 = ((self.angle1 - atan2(self.goal[1],self.goal[0])) *degperpi) % 180
		angleState2 = (self.angle2*degperpi)
		# posStatex = gripper_pos[0]//posStateDivider
		# posStatey = gripper_pos[1]//posStateDivider
		return [angleState1, angleState2]  

	def getGoalDist(self):
		# return (sqrt(self.goal[0]**2 + self.goal[1]**2))//posStateDivider
		return (sqrt(self.goal[0]**2 + self.goal[1]**2))

	def getVert(self):
		endpoints = self.getEndpoint()
		return self.getGoalDist()-sqrt(endpoints[0]**2 + endpoints[1]**2)

	def getHorz(self):
		# return sign(self.angle1 - atan2(self.goal[1],self.goal[0]))
		endpoints = self.getEndpoint()
		return (atan2(endpoints[1],endpoints[0]) - atan2(self.goal[1],self.goal[0])) % 3.1415

	def getStateVal(self):
		endPosx = link1_len*cos(self.angle1) + link2_len*cos(self.angle2+self.angle1)
		endPosy = link1_len*sin(self.angle1) + link2_len*sin(self.angle2+self.angle1)
		return -1*sqrt((self.goal[0]-endPosx)**2 + (self.goal[1]-endPosy)**2)

	def getEndpoint(self):
		endPosx = link1_len*cos(self.angle1) + link2_len*cos(self.angle2+self.angle1)
		endPosy = link1_len*sin(self.angle1) + link2_len*sin(self.angle2+self.angle1)
		return [endPosx, endPosy]

	def move_link1(self, delta_a):
		self.target_angle1 = self.target_angle1 + delta_a

	def move_link2(self, delta_a):
		self.target_angle2 = self.target_angle2 + delta_a

	def randGoal(self):
		global goal_point
		# x = random.random()*150
		# y = random.random()*300-150
		rand_angle = random.random()*3.14159-3.14159/2
		rand_len = random.random()*(link1_len+link2_len-10)
		x = rand_len*cos(rand_angle)
		y = rand_len*sin(rand_angle)
		delta_x = x - self.goal[0]
		delta_y = self.goal[1] - y
		w.move(goal_point, delta_x, delta_y)
		self.goal = [x, y]

	def makeGoal(self, newgoal):
		global goal_point
		x = newgoal[0]
		y = newgoal[1]
		delta_x = x - self.goal[0]
		delta_y = self.goal[1] - y
		w.move(goal_point, delta_x, delta_y)
		self.goal = [x, y]


	def controlLoop(self):
		while(self.__run_control):
			if(self.angle1 != self.target_angle1):
				self.angle1 = self.angle1 + (2*((self.target_angle1-self.angle1) > 0)-1) * max_speed/cntrl_freq
				if(abs(self.target_angle1-self.angle1) < max_speed/cntrl_freq):
					self.angle1 = self.target_angle1
			if(self.angle2 != self.target_angle2):
				self.angle2 = self.angle2 + (2*((self.target_angle2-self.angle2) > 0)-1) * max_speed/cntrl_freq
				if(abs(self.target_angle2-self.angle2) < max_speed/cntrl_freq):
					self.angle2 = self.target_angle2
			# update stored position
			# self.redPosx = link1_len*cos(self.angle1) + link2_len*cos(self.angle2)
			# self.redPosy = link1_len*sin(self.angle1) + link2_len*sin(self.angle2)
			self.redraw()
			time.sleep(1/cntrl_freq)
		print('End of while loop')
		self.__root.destroy()

	def redraw(self):
		global w, link1, link2, joint_pos, gripper_pos
		new_pos = (ground_pos[0] + link1_len * cos(self.angle1), ground_pos[1] - link1_len * sin(self.angle1))
		new_pos2 = (new_pos[0] + link2_len * cos(self.angle1+self.angle2), new_pos[1] - link2_len * sin(self.angle1+self.angle2))

		delta_x1 = new_pos[0]-joint_pos[0]
		delta_y1 = new_pos[1]-joint_pos[1]
		delta_x2 = new_pos2[0]-gripper_pos[0]
		delta_y2 = new_pos2[1]-gripper_pos[1]
		
		w.coords(link1, ground_pos[0], ground_pos[1], new_pos[0], new_pos[1])
		w.coords(link2, new_pos[0], new_pos[1], new_pos2[0], new_pos2[1])

		joint_pos = new_pos
		gripper_pos = new_pos2
		w.move(joint, delta_x1, delta_y1)
		w.move(gripper, delta_x2, delta_y2)

	def show_sub(self):
		global joint_pos, gripper_pos, w, link1, link2, joint, gripper, root, goal_point
		if (self.__root==''):
			self.__root = Tk()
			self.__root.title = "Two Link Arm Simulator"
			self.__root.geometry('620x620')
			self.__root.resizable(FALSE, FALSE)

			mainframe = ttk.Frame(self.__root, padding="10 10 10 10")
			mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

			w = Canvas(mainframe, width=600, height=600)
			w.config(bg='gray')
			w.grid(row=0,column=0)

			node_r = 10
			goal_r = 7

			ground = w.create_oval(ground_pos[0]-node_r, ground_pos[1]-node_r, ground_pos[0]+node_r, ground_pos[1]+node_r, fill='red', outline='red')
			joint_pos = (ground_pos[0] + link1_len * cos(angle1), ground_pos[1] + link1_len * sin(angle2))
			joint = w.create_oval(joint_pos[0]-node_r, joint_pos[1]-node_r, joint_pos[0]+node_r, joint_pos[1]+node_r, fill='red', outline='red')
			gripper_pos = (joint_pos[0] + link2_len * cos(angle1+angle2), joint_pos[1] + link2_len * sin(angle1+angle2))
			gripper = w.create_oval(gripper_pos[0]-node_r, gripper_pos[1]-node_r, gripper_pos[0]+node_r, gripper_pos[1]+node_r, outline='black', width=3)
			
			goal_point = w.create_oval(self.goal[0]+300-goal_r, 300-self.goal[1]-goal_r, self.goal[0]+300+goal_r, 300-self.goal[1]+goal_r, fill='green', outline='green')

			link1 = w.create_line(ground_pos[0], ground_pos[1], joint_pos[0], joint_pos[1], tags='link',  width=5)
			link2 = w.create_line(joint_pos[0], joint_pos[1], gripper_pos[0], gripper_pos[1], tags='link', width=5)
			self.__root.protocol("WM_DELETE_WINDOW", self.on_closing)
			self.__root.mainloop()

	def on_closing(self):
		self.__run_control = False
		print('run_control set to false')
		# self.__root.destroy()
		# self.__root==''