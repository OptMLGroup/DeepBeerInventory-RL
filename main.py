from __future__ import print_function
from clBeergame import *
from utilities import *
import numpy as np 
#from clGeneralParameters import generalParameters
import random
from config import get_config, update_config
import tensorflow as tf 

config = None

#def main(config, beerGame):
def main(config):
	random.seed(10) 

	# prepare loggers and directories
	prepare_dirs_and_logger(config)
	config = update_config(config)
	# save the current configuration of the problem in a json file
	save_config(config)	

	# get the address of data	
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
	main(config)
