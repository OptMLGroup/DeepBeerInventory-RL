import os 
import time
from time import gmtime, strftime
import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

class DQN:
	def __init__(self,agentNum,config):
		if agentNum==0:
			graph_dqn1 = tf.Graph()
			graph_dqn = graph_dqn1
		elif agentNum==1:
			graph_dqn2 = tf.Graph()
			graph_dqn = graph_dqn2
		elif agentNum==2:
			graph_dqn3 = tf.Graph()
			graph_dqn = graph_dqn3
		elif agentNum==3:
			graph_dqn4 = tf.Graph()
			graph_dqn = graph_dqn4

		with graph_dqn.as_default():		

			tf.set_random_seed(1)
			self.agentNum = agentNum
			self.global_step = tf.Variable(0, trainable=False)
			# Hyper Parameters Link:
			self.config = config	
			modelNumber = 'model'+str(agentNum+1)
			#self.addressName = 'model'+str(agentNum+1)+'/savetrained' + str(self.config.address) + '/network-'
			self.address = os.path.join(self.config.model_dir, modelNumber) # 'model'+str(agentNum+1)+'/savetrained'+ str(self.config.address)
			self.addressName = self.address + '/network-'
			if self.config.maxEpisodesTrain != 0:
				self.epsilon = config.epsilonBeg
			else:
				self.epsilon = 0
			self.epsilonRed	 = self.epsilonBuild()
			self.inputSize = self.config.stateDim * self.config.multPerdInpt 
			self.timeStep = 0		
			self.learning_rate = 0		# this is used when we have decaying
			self.iflrReseted = False	# this is used to manage the scale of lr
			
			# init replay memory
			self.replayMemory = deque()
			self.replaySize = 0
			
			# create input placeholders 
			self.createInputs()

			we = []
			be = []
			# create a network same as the saved network, to use some of its weight values. It is used 
			# when the number of output in the loaded network is different than the current model.
			if self.config.ifUsePreviousModel and self.config.ifTransferFromSmallerActionSpace:
				# with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.config.gpu_memory_fraction))) as sess:
				with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.config.number_cpu_active, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
					weights, biases = self.createQNetworkForTL()
					sess.run(tf.global_variables_initializer())				
					if self.config.baseDemandDistribution == 0:
						directory=os.path.join(self.config.pre_model_dir,'uniform/'+str(int(self.config.demandLow))+'-'+str(int(self.config.demandUp)))
					elif self.config.baseDemandDistribution == 1:
						directory=os.path.join(self.config.pre_model_dir,'normal/'+str(int(self.config.demandMu))+'-'+str(int(self.config.demandSigma)))
					elif self.config.baseDemandDistribution == 2:
						directory=os.path.join(self.config.pre_model_dir,'classic')
					elif self.config.baseDemandDistribution == 3:
						directory=os.path.join(self.config.pre_model_dir,'basket'+str(self.config.data_id))
					elif self.config.baseDemandDistribution == 4:
						directory=os.path.join(self.config.pre_model_dir,'forecast'+str(self.config.data_id))

					if self.config.gameConfig == 1:
						# the Sterman case.
						base_brain = 7 + self.config.tlBaseBrain
					elif self.config.gameConfig == 2:
						# the BS case.
						base_brain = 3 + self.config.tlBaseBrain
					else:
						base_brain = self.config.tlBaseBrain
					checkpoint = tf.train.get_checkpoint_state(os.path.join(directory, 'brain'+str(base_brain)))
					# checkpoint = tf.train.get_checkpoint_state(os.path.join(self.config.pre_model_dir, 'brain'+str(self.config.tlBaseBrain)))
					if checkpoint and checkpoint.model_checkpoint_path:
						saver = tf.train.Saver()
						saver.restore(sess, checkpoint.model_checkpoint_path)
						we = sess.run(weights)
						np.save('weights',we)
						be=sess.run(biases)
						if self.config.INFO_print: 
							print("Successfully loaded:", checkpoint.model_checkpoint_path)
						ifLoadedModel = True
					else:
						ifLoadedModel = False
						if self.config.INFO_print: 
							print("Could not find old network weights")
				

			# init Q network
			self.QValue,self.W_fc,self.b_fc = self.createQNetwork('Q', we, be)
			# init Target Q Network
			self.QValueT,self.W_fcT,self.b_fcT = self.createQNetwork('TQ')
			
			# copy the network to target network
			self.copyTargetQNetworkOperation = self.copyTargetQNetworkFunc() 

			# create the placeholders and training model
			self.createTrainingMethod()
			self.currentState = []
			
			# saving and loading networks
			self.saver = tf.train.Saver()
			config_tf = tf.ConfigProto()
			# config_tf.log_device_placement=True
			config_tf.gpu_options.per_process_gpu_memory_fraction = self.config.gpu_memory_fraction
			config_tf.gpu_options.allow_growth = True
			config_tf.intra_op_parallelism_threads = self.config.number_cpu_active
			
			# create the session 
			# self.session = tf.InteractiveSession(config=config_tf)
			self.session = tf.Session(config=config_tf)

			# call tensor board 
			self.merged = []
			if self.config.TB:
				self.merged = tf.summary.merge_all()
			
			# create summary writer 
			self.train_writer = tf.summary.FileWriter(self.config.model_dir + '/tb', self.session.graph)

			# initialize the variables 
			self.session.run(tf.global_variables_initializer())

			if self.config.ifUsePreviousModel:
				if not self.config.ifTransferFromSmallerActionSpace:
					# check if all agents are dnn, use the save network by each of them.

					if self.config.ifSinglePathExist:
						directory=self.config.pre_model_dir
					elif self.config.baseDemandDistribution == 0:
						directory=os.path.join(self.config.pre_model_dir,'uniform/'+str(int(self.config.demandLow))+'-'+str(int(self.config.demandUp)))
					elif self.config.baseDemandDistribution == 1:
						directory=os.path.join(self.config.pre_model_dir,'normal/'+str(int(self.config.demandMu))+'-'+str(int(self.config.demandSigma)))
					elif self.config.baseDemandDistribution == 2:
						directory=os.path.join(self.config.pre_model_dir,'classic')
					elif self.config.baseDemandDistribution == 3:
						directory=os.path.join(self.config.pre_model_dir,'basket'+str(self.config.data_id))
					elif self.config.baseDemandDistribution == 4:
						directory=os.path.join(self.config.pre_model_dir,'forecast'+str(self.config.data_id))

					if self.config.ifSinglePathExist:
						base_brain = self.config.tlBaseBrain + 1
					else:
						if self.config.gameConfig == 1:
							# the Sterman case.
							base_brain = 7 + self.config.tlBaseBrain
						elif self.config.gameConfig == 2:
							base_brain = 3 + self.config.tlBaseBrain
						else:
							# the BS case.
							base_brain = self.config.tlBaseBrain
					# checkpoint = tf.train.get_checkpoint_state(os.path.join(self.config.pre_model_dir, 'brain'+str(self.config.gameConfig)))
					if self.config.ifSinglePathExist:
						model_address = os.path.join(directory, 'model'+str(base_brain))
					else:
						model_address = os.path.join(directory, 'brain'+str(base_brain))
					
					checkpoint = tf.train.get_checkpoint_state(model_address)
					if checkpoint and checkpoint.model_checkpoint_path:
						self.saver.restore(self.session, checkpoint.model_checkpoint_path)
						if self.config.INFO_print: 
							print("Successfully loaded:", checkpoint.model_checkpoint_path)

						# copy the network to target network
						self.session.run(self.copyTargetQNetworkOperation)
					else:
						if self.config.INFO_print: 
							print("Could not find old network weights in ", model_address)
				else:				
					if ifLoadedModel:
						# copy the network to target network
						self.session.run(self.copyTargetQNetworkOperation)
					else:
						if self.config.INFO_print: 
							print("Could not find old network weights")
			else:
				if self.config.INFO_print: 
					print("Previous models will not be used")


	# returns the operator which copies the Q network to the target network	
	def copyTargetQNetworkFunc(self):
		operation = []
		for i in range(self.config.NoHiLayer+1):
			operation += [	self.W_fcT[i].assign(self.W_fc[i]),self.b_fcT[i].assign(self.b_fc[i])]
		return operation

	def copyBaseNetworkFunc(self, weights, biases):
		operation = []
		for i in range(self.config.NoHiLayer): # we ignored the last layer (Q-value) that its dimension is different
			operation += [	self.W_fc[i].assign(weights[i]),self.b_fc[i].assign(biases[i])]
		return operation

	def createInputs(self):
		# input layer
		with tf.name_scope('input'):
			self.stateInput = tf.placeholder("float",[None,self.config.multPerdInpt,self.config.stateDim])
		with tf.name_scope('input_reshape'):
			self.stateInputFlat = tf.reshape(self.stateInput,[-1,self.inputSize])

	def createQNetworkForTL(self, graph_name='Q'):
		# input layer
		W = []
		b = []
		layer = []

		for j in range(self.config.NoHiLayer+1):
			# var = np.sqrt(1.0/(self.config.nodes[j] + 0.0))
			if j == 0:
				# hidden layers
				name=graph_name+'-layer'+str(j+1)
				hidden, weights, biases = self.fc_layer(self.stateInputFlat, self.config.nodes[j], 
				self.config.nodes[j+1], name, j) # act=tf.sigmoid
			elif j == self.config.NoHiLayer:
				# output value
				name=graph_name+'-layer'+str(j+1)
				QValue, weights, biases = self.fc_layer(layer[j-1], self.config.nodes[j],
				 self.config.baseActionSize, name,j ,act=tf.identity)
			else:
				# hidden layers
				name=graph_name+'-layer'+str(j+1)
				hidden, weights, biases = self.fc_layer(layer[j-1],
				 self.config.nodes[j], self.config.nodes[j+1], name, j)

			layer += [hidden]
			W += [weights]
			b += [biases]
			
		return W, b

	def createQNetwork(self, graph_name, initial_w=[], initial_b=[]):
		# initiate the weight variables
		W = []
		b = []
		layer = []

		for j in range(self.config.NoHiLayer+1):
			# var = np.sqrt(1.0/(self.config.nodes[j] + 0.0))
			if list(initial_w):
				w_init = initial_w[j]
				b_init = initial_b[j]
			else:
				w_init = []
				b_init = []

			if j == 0:
				# hidden layers
				name=graph_name+'-layer'+str(j+1)
				hidden, weights, biases = self.fc_layer(self.stateInputFlat, self.config.nodes[j], 
				self.config.nodes[j+1], name, j, w_init, b_init) # act=tf.sigmoid
			elif j == self.config.NoHiLayer:
				# output value
				name=graph_name+'-layer'+str(j+1)
				QValue, weights, biases = self.fc_layer(layer[j-1], self.config.nodes[j],
				 self.config.nodes[j+1], name,j, init_w=[], init_b=[] ,act=tf.identity)
			else:
				# hidden layers
				name=graph_name+'-layer'+str(j+1)
				hidden, weights, biases = self.fc_layer(layer[j-1],
				 self.config.nodes[j], self.config.nodes[j+1], name, j, w_init, b_init)

			layer += [hidden]
			W += [weights]
			b += [biases]

		return QValue,W,b


	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.config.actionListLen])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1) # dim: batchSize *1
		with tf.name_scope('cost'):
			self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		tf.summary.scalar('cost', self.cost)		
		#self.trainStep = tf.train.RMSPropOptimizer(self.config.lr0,self.config.decay,self.config.momentum,1e-6).minimize(self.cost)
		if self.config.ifDecayAdam: 
			with tf.name_scope('train'):
				self.learning_rate = tf.train.exponential_decay(self.config.lr0, self.global_step, self.config.decayStep, self.config.decayRate, staircase=True)
				self.trainStep = tf.train.AdamOptimizer(self.learning_rate,0.9,0.999,1e-8).minimize(self.cost, global_step=self.global_step)
		else:	
			with tf.name_scope('train'):
				self.trainStep = tf.train.AdamOptimizer(self.config.lr0,0.9,0.999,1e-8).minimize(self.cost)

	def trainQNetwork(self):
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,self.config.batchSize)
		state_batch = [data[0] for data in minibatch]  #dim: each item is multPerInput*stateDim
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInput:nextState_batch},session = self.session)
		# for i in range(0,self.config.batchSize):
		# 	terminal = minibatch[i][4]
		# 	if terminal:
		# 		y_batch.append(reward_batch[i])
		# 	else:
		# 		y_batch.append(reward_batch[i] + self.config.gamma * np.max(QValue_batch[i]))
		y_batch = reward_batch + (1-np.array(minibatch)[:,4])*self.config.gamma * np.max(QValue_batch, axis=1)
		# dim yInput: batchSize*1
		# dim actionInput: batchSize*actionListLen
		# dim stateInput: batchSize**multPerInput*stateDim
		
		# check if lr < Minlr, stop its decreasing procedure
		lr = self.learning_rate.eval(session=self.session)
		if lr < self.config.Minlr and not self.iflrReseted:
			self.iflrReseted = True
			self.learning_rate = tf.train.exponential_decay(lr, self.global_step, 10000000, 1, staircase=True)
		
		feed_dict={
			self.yInput : y_batch,			
			self.actionInput : action_batch,
			self.stateInput : state_batch
			}
		if self.config.TB and (self.timeStep % self.config.tbLogInterval == 1):
			# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()			
			# summary, _ = self.session.run([self.merged, self.trainStep], feed_dict, feed_dictoptions=run_options,
			# 								run_metadata=run_metadata)
			summary, _ = self.session.run([self.merged, self.trainStep], feed_dict,
											run_metadata=run_metadata)			
			self.train_writer.add_run_metadata(run_metadata, 'step%03d' % self.timeStep)
			self.train_writer.add_summary(summary, self.timeStep)			
			if self.config.INFO_print: 
				print('Adding run metadata for', self.timeStep)
		else:
			summary, _ = self.session.run([self.merged, self.trainStep], feed_dict)
			if self.config.TB and (self.timeStep%self.config.tbLogInterval==1):
				self.train_writer.add_summary(summary, self.timeStep)
			# self.trainStep.run(feed_dict, session=self.session)
			# self.session.run([self.trainStep], feed_dict)
			
			
		# grad_w= self.session.run([tf.norm(tf.gradients(self.cost, self.W_fc[3]))], feed_dict)
		# grad_b= self.session.run([tf.norm(tf.gradients(self.cost, self.b_fc[3]))], feed_dict)
		# print('grad is ', grad_w, grad_b)
		"""trainResult = self.session.run(self.cost,feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			},session = self.session)	
		print("TRAIN_RESULT", trainResult) """
		
		# save network every saveInterval iteration
		if (self.timeStep+1) % self.config.saveInterval == 0:
			self.saver.save(self.session, self.addressName, global_step = self.timeStep)
			print("network weights are saved")

		if self.timeStep % self.config.dnnUpCnt == 0:
			self.copyTargetQNetwork()
		
	def train(self,nextObservation,action,reward,terminal,playType):
		# Considering the multi-period observation idea, merges the last $m-1$ periods with the new state. 
		newState = np.append(self.currentState[1:,:],[nextObservation],axis = 0)
		
		if playType == "train":
			if self.config.MultiAgent:
				if self.config.MultiAgentRun[self.agentNum]:
					self.replayMemory.append([self.currentState,action,reward,newState,terminal])
					self.replaySize = len(self.replayMemory)
			else:
				self.replayMemory.append([self.currentState,action,reward,newState,terminal])
				self.replaySize = len(self.replayMemory)

			if self.replaySize > self.config.maxReplayMem and self.config.MultiAgentRun[self.agentNum]:
				self.replayMemory.popleft()
				self.trainQNetwork()
				state = "train"
				self.timeStep += 1
			
			elif self.replaySize >= self.config.minReplayMem and self.config.MultiAgentRun[self.agentNum]:
				# Train the network
				state = "train"
				self.trainQNetwork()
				self.timeStep += 1
			else:
				state = "observe" 
				
			if terminal and state == "train":
				self.epsilonReduce()	
					
			# print(info)
			#print("AGENT", self.agentNum,"/TRAINING_ITER", self.timeStep, "/ STATE", state, \)
			#"/ EPSILON", self.epsilon

		self.currentState = newState
		
	def getDNNAction(self,playType):
		action = np.zeros(self.config.actionListLen)
		action_index = 0
		if playType == "train":
			if (random.random() <= self.epsilon) or (self.replaySize < self.config.minReplayMem):
				action_index = random.randrange(self.config.actionListLen)
				action[action_index] = 1
			else:
				QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]},session = self.session)[0]
				action_index = np.argmax(QValue)
				action[action_index] = 1
		elif playType == "test"	:
			QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]},session = self.session)[0]
			action_index = np.argmax(QValue)
			action[action_index] = 1

		return action
			
	# this functions sets the current state of the game in the begining of each game	
	def setInitState(self,observation):
		self.currentState = np.stack([observation for _ in range(self.config.multPerdInpt)], axis = 0) # multPerdInpt observations stacked. each row is an observation


	def epsilonBuild(self): # this function specifies how much we should deduct from /epsilon at each game
		betta = 0.8
		if self.config.maxEpisodesTrain != 0:
			epsilon_red = (self.config.epsilonBeg - self.config.epsilonEnd)/(self.config.maxEpisodesTrain*betta)
		else:
			epsilon_red = 0
		return epsilon_red

	def epsilonReduce(self):
		# Reduces the values of epsilon at each iteration of episode
		if self.epsilon >self.config.epsilonEnd:
			self.epsilon -= self.epsilonRed

	def deleteGraph(self):
		tf.reset_default_graph()
		self.sess.close()



	def fc_layer(self, input_tensor, input_dim, output_dim, layer_name, j_, init_w=[], init_b=[], act=tf.nn.relu):
		"""Reusable code for making a simple fully connected neural net layer.

		It does a matrix multiply, bias add, and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		def variable_summaries(var):
			"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
			with tf.name_scope('summaries'):
				mean = tf.reduce_mean(var)
				tf.summary.scalar('mean', mean)
				with tf.name_scope('stddev'):
					stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
				tf.summary.scalar('stddev', stddev)
				tf.summary.scalar('max', tf.reduce_max(var))
				tf.summary.scalar('min', tf.reduce_min(var))
				tf.summary.histogram('histogram', var)
		
		def weight_variable(shape, j_, init_w=None):
			"""Create a weight variable with appropriate initialization."""
			if not list(init_w):
				initial = tf.random.truncated_normal(shape, stddev = 0.1)
			else:
				initial = tf.constant(init_w)
			if self.config.iftl and j_ < self.config.NoFixedLayer:
				return tf.Variable(initial, trainable=False)
			else:
				return tf.Variable(initial, trainable=True)

		def bias_variable(shape, j_, init_b=None):
			"""Create a bias variable with appropriate initialization."""
			if not list(init_b):
				initial = tf.constant(0.1, shape = shape)
			else:
				initial = tf.constant(init_b)
			if self.config.iftl and j_ < self.config.NoFixedLayer:
				return tf.Variable(initial, trainable=False)
			else:
				return tf.Variable(initial, trainable=True)

		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = weight_variable([input_dim, output_dim], j_, init_w)
				variable_summaries(weights)
			with tf.name_scope('biases'):
				biases = bias_variable([output_dim], j_, init_b)
				variable_summaries(biases)
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.matmul(input_tensor, weights) + biases
				tf.summary.histogram('pre_activations', preactivate)
				activations = act(preactivate, name='activation')
				tf.summary.histogram('activations', activations)
			return activations, weights, biases

