
import numpy as np 
import math

class NN():
	'''
	Create a Custom Neural Network
	input:
	numX, numW, numY
	numX(int) : number of inputs nodes of the network. Default 2
	numW([int]) : list of number of weigth for each hidden layer. [numW1, numW2,...,numWH]. Default [5]
	numY(int) : number of outputs to the network. Default 2
	'''
	def __init__(self, *args, **kwargs):
		## Default values
		self.numX = 2
		self.numW = [5]
		self.numY = 2
		self.func = 'relu'

		## Initialize input variables
		if (len(args)>0):
			self.numX = args[0]
			if(len(args)>1):
				self.numY = args[1]	
				if(len(args)>2):
					if(type(args[2])==type(1)):
						self.numW = [args[2]]
					else:
						self.numW = args[2]
		self.numH = len(self.numW) # number of hidden layers

		
		for name, value in kwargs.items():
			# if (name=='func' and (value=='sigmoid' or value=='relu')):
			if(name=='func'):
				self.func = value


		## Initialize input, weigths, and outputs of the network
		self.X = np.zeros([1, self.numX])
		self.W = [0] * (len(self.numW)+1)
		W_scale = 10.0
		W_offset = W_scale/2.0
		for i in range(len(self.W)):
			if (i==0):
				self.W[i] = W_scale*np.random.rand(self.numX, self.numW[i])-W_offset
			elif (i==len(self.W)-1):
				self.W[i] = W_scale*np.random.rand(self.numW[i-1], self.numY)-W_offset
			else:
				self.W[i] = W_scale*np.random.rand(self.numW[i-1], self.numW[i])-W_offset
		self.B = [0] * (len(self.numW)+1)
		for i in range(len(self.B)):
			if (i==len(self.B)-1):
				self.B[i] = W_scale*np.random.rand(1,self.numY)-W_offset
			else:
				self.B[i] = W_offset*np.random.rand(1,self.numW[i])-W_offset
		self.Y = np.zeros([self.numY,1])

	'''
	Sigmoid function for an array
	X(float[]) : array of values to calculate sigmoid
	'''	
	def sigmoid(self, X):
		result = np.zeros(X.shape)
		for i in range(len(X[0])):
			try:
				result[0][i] = 1 / (1+math.exp(-1*X[0][i])) # sigmoid function 
			except OverflowError:
				print("Over flow", X[0][i])
		return result

	def relu(self, X):
		result = np.zeros(X.shape)
		for i in range(len(X[0])):
			result[0][i] = max([0,X[0][i]])
		return result

	def relu2(self, X):
		result = X
		return result

	def lrelu(self, X):
		result = abs(X)
		return result

	'''
	NN forward propagation
	X(float[]) : array of inputs to calculate the forward propagation
	'''
	def forward(self, X):
		a = np.array([X])
		for i in range(len(self.W)):
			z = np.dot(a, self.W[i]) / np.prod(a.shape) + self.B[i] # z = a*w + b
			if (i==len(self.W)-1):
				return z
			if(self.func=='sigmoid'):
				a = self.sigmoid(z) # a = sigma(a*w)
			elif(self.func=='relu'):
				a = self.relu(z)
			elif(self.func=='reul2'):
				a = self.relu2(z)
			elif(self.func=='lrelu'):
				a = self.lrelu(z)
			else:
				a = self.relu(z)
		return a

	def train(self, X, Y, *arg):
		X = np.array([X])
		Y = np.array([Y])
		if(len(arg)==1):
			n = arg
		else:
			n = 1.0
		A = [0]*(len(self.W)+1)
		Z = [0]*len(self.W)
		A[0] = X
		for i in range(len(self.W)):
			Z[i] = np.dot(A[i], self.W[i]) / np.prod(A[i].shape) + self.B[i] # z = a*w + b
			if (i==len(self.W)-1):
				A[i+1] = Z[i]
			elif(self.func=='sigmoid'):
				A[i+1] = self.sigmoid(Z[i]) # a = sigma(a*w)
			elif(self.func=='relu'):
				A[i+1] = self.relu(Z[i])
			elif(self.func=='relu2'):
				A[i+1] = self.relu2(Z[i])
			elif(self.func=='lrelu'):
				A[i+1] = self.lrelu(Z[i])
			else:
				A[i+1] = self.relu(Z[i])
		D = [0]*(len(self.W)+1)
		# D[0] = (A[-1]-Y) * A[-1]*(1-A[-1])
		D[0] = (A[-1]-Y)
		# D[0] = 0.5 * (A[-1]-Y)**2 / np.prod(A[-1].shape) * A[-1]*(1-A[-1])

		for i in range(len(self.W)):
			if (self.func=='sigmoid'):
				D[i+1] = np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))) * A[-2-i]*(1-A[-2-i])
			elif (self.func=='relu'):
				D[i+1] = np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))) * (np.sign(A[-2-i])+1)/2
			elif (self.func=='relu2'):
				D[i+1] = np.transpose(np.dot(self.W[-i-1], np.transpose(D[i])))
			elif (self.func=='lrelu'):
				D[i+1] = np.transpose(np.dot(self.W[-i-1], np.transpose(D[i]))) * 1 if A[-2-i]>=0 else 0.01
			self.W[-1-i] = self.W[-1-i] - n * np.dot(np.transpose(A[-2-i]), D[i])
			self.B[-1-i] = self.B[-1-i] - n * D[i]