# python optimize_bs.py --demandDistribution=1  --demandUp=9 --demandMu=10 
# python optimize_bs.py --demandDistribution=3  --data_id=6 

import sys
from clBeergame import *
from utilities import *
import numpy as np 
#from clGeneralParameters import generalParameters
import random
import tensorflow as tf 
from config import get_config, update_config
from sensitivity_run import set_config

config=None

def main(config):
	random.seed(10) 

	# set a non-dqn default value 
	config.brainTypes = 26	
	config = set_config(config)
	config.gamma=0.99
	config = update_config(config)
	
	# prepare loggers and directories
	prepare_dirs_and_logger(config)

	# save the current configuration of the problem in a json file
	save_config(config)	
	
	# we assume have observed just 100 observations, created a frequency vector based on that, 
	# and then created a randomly generated sample with size 60000 based on that. Then, we have trained the 
	# beer game using this new samples and the results are provided. In both cases, we run a test based on unseen 
	# data set of the known dataset. 
	if config.observation_data:
		adsr = '../data/demandTr-obs-'
	elif config.demandDistribution == 3:
		if config.scaled:
			adsr = '../data/basket_data/scaled'
		else:
			adsr = '../data/basket_data'
	elif config.demandDistribution == 4:
		if config.scaled:
			adsr = '../data/forecast_data/scaled'
		else:
			adsr = '../data/forecast_data'
	else:
		adsr = '../data/demandTr'
		
	# load demands
	# demandTr = np.load('demandTr'+str(config.demandDistribution)+'-'+str(config.demandUp)+'.npy')
	if config.MultiAgent and 1==2:
		
		if config.demandDistribution == 0:
			direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(config.demandUp)+'-'+str(config.maxEpisodesTrain)+'.npy')
			if not os.path.exists(direc):
				direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(config.demandUp)+'.npy')
		elif config.demandDistribution == 1:
			direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(int(config.demandMu))+'-'+str(int(config.demandSigma))+'-'+str(config.maxEpisodesTrain)+'.npy')
			if not os.path.exists(direc):
				direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(int(config.demandMu))+'-'+str(int(config.demandSigma))+'.npy')
		elif config.demandDistribution == 2:
			direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(config.maxEpisodesTrain)+'.npy')
			if not os.path.exists(direc):
				direc = os.path.realpath(adsr+str(config.demandDistribution)+'.npy')
	elif config.demandDistribution == 0:
		direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(config.demandUp)+'-'+str(config.maxEpisodesTrain)+'.npy')
		if not os.path.exists(direc):
			direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(config.demandUp)+'.npy')
	elif config.demandDistribution == 1:
		direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(int(config.demandMu))+'-'+str(int(config.demandSigma))+'.npy')
	elif config.demandDistribution == 2:
		direc = os.path.realpath(adsr+str(config.demandDistribution)+'.npy')	
	elif config.demandDistribution == 3:
		direc = os.path.realpath(adsr+'/demandTr-'+str(config.data_id)+'.npy')
	elif config.demandDistribution == 4:
		direc = os.path.realpath(adsr+'/demandTr-'+str(config.data_id)+'.npy')
	demandTr = np.load(direc)	
	print("loaded training set=", direc)
	if config.demandDistribution == 0:
		direc = os.path.realpath('../data/demandTs'+str(config.demandDistribution)+'-'+str(config.demandUp)+'-'+str(config.maxEpisodesTrain)+'.npy')
		if not os.path.exists(direc):
			direc = os.path.realpath('../data/demandTs'+str(config.demandDistribution)+'-'+str(config.demandUp)+'.npy')
	elif config.demandDistribution == 1:
		direc = os.path.realpath('../data/demandTs'+str(config.demandDistribution)+'-'+str(int(config.demandMu))+'-'+str(int(config.demandSigma))+'.npy')
	elif config.demandDistribution == 2:
		direc = os.path.realpath('../data/demandTs'+str(config.demandDistribution)+'.npy')
	elif config.demandDistribution == 3:
		direc = os.path.realpath(adsr+'/demandTs-'+str(config.data_id)+'.npy')
		direcVl = os.path.realpath(adsr+'/demandVl-'+str(config.data_id)+'.npy')
		demandVl = np.load(direcVl)	
	elif config.demandDistribution == 4:
		direc = os.path.realpath(adsr+'/demandTs-'+str(config.data_id)+'.npy')
		direcVl = os.path.realpath(adsr+'/demandVl-'+str(config.data_id)+'.npy')
		demandVl = np.load(direcVl)	
	demandTs = np.load(direc)	
	print("loaded test set=", direc)


	# initilize an instance of Beergame
	beerGame = clBeerGame(config)

	if config.demandDistribution == 1 or config.demandDistribution == 3 or config.demandDistribution == 4:	
		bs_range = range(int(config.demandMu - 25*config.demandSigma), int(config.demandMu + 25*config.demandSigma))
	elif config.demandDistribution == 0 or config.demandDistribution == 2:
		bs_range = range(int(-25*config.demandUp), int(25*config.demandUp))

	train_episode = 50
	res_train = np.zeros((4,len(bs_range),train_episode,2))
	res_test = np.zeros((4,config.testRepeatMid))
	config.actionListLenOpt = len(bs_range)

	brains = [26,27,28,29]
	# brains = [30,31,32,33]
	min_brain = min(brains)
	for brain in brains:
		print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "brain=", brain )
		config.brainTypes = brain 
		config = update_config(config)
		for k in range(0,config.NoAgent):
			beerGame.players[k].compTypeTest = config.agentTypes[k]

		for c, bs in enumerate(bs_range):
			beerGame.players[brain-min_brain].optmlBaseStock = bs 
			for i in range(train_episode):
				demand = demandTr[i]
				result = beerGame.playGame(demand,"test")
				res_train[brain-min_brain,c,i,0] = bs 
				res_train[brain-min_brain,c,i,1] = sum(result)

		avg_cost = np.mean(res_train[brain-min_brain,:,:,1], axis=1)
		min_cost = np.min(avg_cost)
		best_bs_index = np.argmin(avg_cost)
		best_bs = res_train[brain-min_brain,best_bs_index,0,0]
		print("train result: brain=", brain - min_brain, "best_bs= ", best_bs, "min_cost= ", min_cost)

		beerGame.players[brain-min_brain].optmlBaseStock = best_bs 
		for i in range(config.testRepeatMid):
			demand = demandTs[i]
			result = beerGame.playGame(demand,"test")
			res_test[brain-min_brain,i] = sum(result)
                print("test_cost", np.mean(res_test[brain-min_brain,:]))
                        
	if config.demandDistribution == 3: 
		np.save('bs_search-'+str(config.demandDistribution)+ '-' + str(config.data_id) + '.npy',res_train)
	else:
		np.save('bs_search-'+str(config.demandDistribution)+'.npy',res_train)
	# get the average over xx episodes. 
	avg=np.mean(res_test, axis=1)	
	print(avg)




if __name__ == '__main__':		
	# load parameters
	config, unparsed = get_config()

	# run main
	main(config)






