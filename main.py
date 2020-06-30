from __future__ import print_function
import sys
from clBeergame import *
from utilities import *
import numpy as np 
#from clGeneralParameters import generalParameters
import random
from config import get_config, update_config
import tensorflow as tf 
from sensitivity_run import set_config

config = None

#def main(config, beerGame):
def main(config):
	random.seed(10) 

	# prepare loggers and directories
	prepare_dirs_and_logger(config)


	if config.preload_config:
		config = set_config(config)

	config = update_config(config)

	# save the current configuration of the problem in a json file
	save_config(config)	
	
	# we assume have observed just 100 observations, created a frequency vector based on that, 
	# and then created a randomly generated sample with size 60000 based on that. Then, we have trained the 
	# beer game using this new samples and the results are provided. In both cases, we run a test based on unseen 
	# data set of the known dataset. 
	if config.observation_data:
		adsr = 'data/demandTr-obs-'
	elif config.demandDistribution == 3:
		if config.scaled:
			adsr = 'data/basket_data/scaled'
		else:
			adsr = 'data/basket_data'
	elif config.demandDistribution == 4:
		if config.scaled:
			adsr = 'data/forecast_data/scaled'
		else:
			adsr = 'data/forecast_data'
	else:
		adsr = 'data/demandTr'

		
	# load demands
	# demandTr = np.load('demandTr'+str(config.demandDistribution)+'-'+str(config.demandUp)+'.npy')
	if config.demandDistribution == 0:
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
		direc = os.path.realpath('data/demandTs'+str(config.demandDistribution)+'-'+str(config.demandUp)+'-'+str(config.maxEpisodesTrain)+'.npy')
		if not os.path.exists(direc):
			direc = os.path.realpath('data/demandTs'+str(config.demandDistribution)+'-'+str(config.demandUp)+'.npy')
	elif config.demandDistribution == 1:
		direc = os.path.realpath('data/demandTs'+str(config.demandDistribution)+'-'+str(int(config.demandMu))+'-'+str(int(config.demandSigma))+'.npy')
	elif config.demandDistribution == 2:
		direc = os.path.realpath('data/demandTs'+str(config.demandDistribution)+'.npy')
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

	if config.maxEpisodesTrain == -1:
	# demandTs = np.array([[2,6,6,1,3,1,0,8,0,6,6,3,0,8,0,1,2,5,8,2,6,2,4,4,8,1,0,6,4,2,2,7,7,4,7,7,2,2,5,0,3,4,6,1,1,5,2,0,0,7,3,1,0,6,3,8,2,4,1,8,6,2,3,7,5,4,7,8,8,3,5,5,5,0,1,7,4,5,3,1,3,3,1,0,0,1,4,5,5,0,4,1,3,8,5,1,4,0,4,2]])
	# demandTs = np.array([[4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]])
	# demandTs = np.array([[9,11,9,9,8,7,12,11,12,13,6,11,8,11,11,12,12,9,10,10,10,8,13,9,8,12,12,8,8,9,10,8,7,12,10,5,7,11,13,13,6,8,13,11,9,10,8,10,8,9,13,11,9,6,10,9,11,9,7,10,13,9,8,12,9,9,11,10,5,9,7,12,14,10,11,10,10,12,8,10,8,11,9,7,11,10,13,10,13,13,9,9,14,11,12,7,12,11,9,12]])
		# demandTs = np.load('data/demandTs003.npy')
		demandTs = np.load('data/demandTs110.npy')
		# demandTs = np.concatenate((4*np.ones((10,4)) ,8*np.ones((10,config.Ttest+10))), axis=1).astype(int)
		# config.f = [config.f1, config.f2, config.f3, config.f4] 
		print("loaded auxiliary data")
		# config.agentTypes = ["optm", "optm","optm","optm"]

	# initilize an instance of Beergame
	beerGame = clBeerGame(config)
	
	# get the length of the demand.
	demand_len = np.shape(demandTr)[0] 
	# Do Initial tests
	beerGame.doTestMid(demandTs[0:config.testRepeatMid])
	
	# train the specified number of games
	for i in range(0, config.maxEpisodesTrain):
		beerGame.playGame(demandTr[i%demand_len],"train")
		# get the test results
		if (np.mod(beerGame.curGame,config.testInterval) == 0) and (beerGame.curGame>500):	
			beerGame.doTestMid(demandTs[0:config.testRepeatMid])			
		
	# do the last test on the middle test data set.
	beerGame.doTestMid(demandTs[0:config.testRepeatMid])
	if config.demandDistribution == 3:
		beerGame.doTestMid(demandVl[0:config.testRepeatMid])
	
if __name__ == '__main__':	
	# load parameters
	config, unparsed = get_config()

	# run main
	#tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
	main(config)
