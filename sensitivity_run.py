# python sensitivity_run.py --demandDistribution=1 --demandMu=10 --INFO_print=False > sensitivity_run.out &> sensitivity_run.err
import sys
from clBeergame import *
from utilities import *
import numpy as np 
#from clGeneralParameters import generalParameters
import random
from config import get_config, update_config
import logging
# get TF logger
import tensorflow as tf 

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ['TF_CPP_MIN_VLOG_LEVEL']='0'
# tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)

config = None
from pymatbridge import Matlab

def set_config(config):

	config.iftl=False
	config.ifSaveFigure=True
	# config.maxEpisodesTrain=0 # comment it out to save records
	#config.ifUsePreviousModel=True
	#config.ifUsePreviousModel=False
	config.optimize_bs = True
	config.gamma=0.99 
	config.Ttest=100
	#config.maxEpisodesTest=0 # comment it out to save records 
	#config.ifsaveHistInterval=False # comment it out to save records 

	if config.demandDistribution == 0 and config.demandUp==3:
		config.f1=8.0 
		config.f2=8.0 
		config.f3=0 
		config.f4=0  

		config.actionUp=2  
		config.actionLow=-2 
		config.ch1=2 
		config.ch2=2 
		config.ch3=2 
		config.ch4=2 
		config.cp1=2.0 
		config.cp2=0 
		config.cp3=0 
		config.cp4=0 
		config.leadRecItem1=2 
		config.leadRecItem2=2 
		config.leadRecItem3=2 
		config.leadRecItem4=2 
		config.leadRecOrder1=2 
		config.leadRecOrder2=2 
		config.leadRecOrder3=2 
		config.leadRecOrder4=2 
		config.ILInit1=0 
		config.ILInit2=0 
		config.ILInit3=0 
		config.ILInit4=0 
		config.AOInit1=0 
		config.AOInit2=0 
		config.AOInit3=0 
		config.AOInit4=0 
		config.ASInit1=0 
		config.ASInit2=0 
		config.ASInit3=0 
		config.ASInit4=0 
		config.ifOptimalSolExist=True 
		config.if_use_AS_t_plus_1 = True

	elif config.demandDistribution == 0 and config.demandUp==9:
		config.f1=19.0
		config.f2=20.0
		config.f3=20.0 
		config.f4=14.0 

		config.actionUp=8
		config.actionLow=-8
		config.ch1=0.5
		config.ch2=0.5 
		config.ch3=0.5 
		config.ch4=0.5 
		config.cp1=1.0
		config.cp2=1.0 
		config.cp3=1.0 
		config.cp4=1.0 
		config.leadRecItem1=2 
		config.leadRecItem2=2 
		config.leadRecItem3=2 
		config.leadRecItem4=2 
		config.leadRecOrder1=2 
		config.leadRecOrder2=2 
		config.leadRecOrder3=2 
		config.leadRecOrder4=1
		config.ILInit1=12 
		config.ILInit2=12 
		config.ILInit3=12 
		config.ILInit4=12 
		config.AOInit1=4 
		config.AOInit2=4 
		config.AOInit3=4 
		config.AOInit4=4 
		config.ASInit1=4 
		config.ASInit2=4 
		config.ASInit3=4 
		config.ASInit4=4 
		config.ifOptimalSolExist=True 

	elif config.demandDistribution == 1 and config.demandMu==10:
		config.f1=48.0 
		config.f2=43.0 
		config.f3=41. 
		config.f4=30. 

		config.demandSigma=2
		config.actionUp=5
		config.actionLow=-5
		config.ch1=1.0
		config.ch2=0.75 
		config.ch3=0.5 
		config.ch4=0.25 
		config.cp1=10.0
		config.cp2=.0 
		config.cp3=.0 
		config.cp4=.0 
		config.leadRecItem1=2 
		config.leadRecItem2=2 
		config.leadRecItem3=2 
		config.leadRecItem4=2 
		config.leadRecOrder1=2 
		config.leadRecOrder2=2 
		config.leadRecOrder3=2 
		config.leadRecOrder4=1
		config.ILInit1=10
		config.ILInit2=10
		config.ILInit3=10
		config.ILInit4=10
		config.AOInit1=10
		config.AOInit2=10
		config.AOInit3=10
		config.AOInit4=10
		config.ASInit1=10
		config.ASInit2=10
		config.ASInit3=10
		config.ASInit4=10
		config.ifOptimalSolExist=True 

	elif config.demandDistribution == 2:
		config.f1=32.0 
		config.f2=32.0 
		config.f3=32 
		config.f4=24 

		config.actionUp=8
		config.actionLow=-8
		config.demandUp=9 
		config.demandLow=0 
		config.ch1=0.5
		config.ch2=0.5 
		config.ch3=0.5 
		config.ch4=0.5 
		config.cp1=1.0
		config.cp2=1.0 
		config.cp3=1.0 
		config.cp4=1.0 
		config.leadRecItem1=2 
		config.leadRecItem2=2 
		config.leadRecItem3=2 
		config.leadRecItem4=2 
		config.leadRecOrder1=2 
		config.leadRecOrder2=2 
		config.leadRecOrder3=2 
		config.leadRecOrder4=1
		config.ILInit1=12
		config.ILInit2=12
		config.ILInit3=12
		config.ILInit4=12
		config.AOInit1=4
		config.AOInit2=4
		config.AOInit3=4
		config.AOInit4=4
		config.ASInit1=4
		config.ASInit2=4
		config.ASInit3=4
		config.ASInit4=4
		config.ifOptimalSolExist=True 

	elif config.demandDistribution == 3:
		config.actionUp=5
		config.actionLow=-5
		config.ch1=1.0
		config.ch2=0.75 
		config.ch3=0.5 
		config.ch4=0.25 
		config.cp1=10.0
		config.cp2=.0 
		config.cp3=.0 
		config.cp4=.0 
		config.leadRecItem1=2 
		config.leadRecItem2=2 
		config.leadRecItem3=2 
		config.leadRecItem4=2 
		config.leadRecOrder1=2 
		config.leadRecOrder2=2 
		config.leadRecOrder3=2 
		config.leadRecOrder4=1
		config.ifOptimalSolExist=True 
		config.testRepeatMid = 200 

		if config.data_id==6:
			if config.scaled:
				config.f1=19.0 
				config.f2=12.0 
				config.f3=12. 
				config.f4=8. 
				config.demandMu=3
				config.demandSigma=2
				config.ILInit1=3
				config.ILInit2=3
				config.ILInit3=3
				config.ILInit4=3
				config.AOInit1=3
				config.AOInit2=3
				config.AOInit3=3
				config.AOInit4=3
				config.ASInit1=3
				config.ASInit2=3
				config.ASInit3=3
				config.ASInit4=3
			else:
				config.f1=181.0 
				config.f2=119.0 
				config.f3=110. 
				config.f4=74. 
				config.demandMu=25
				config.demandSigma=18
				config.ILInit1=25
				config.ILInit2=25
				config.ILInit3=25
				config.ILInit4=25
				config.AOInit1=25
				config.AOInit2=25
				config.AOInit3=25
				config.AOInit4=25
				config.ASInit1=25
				config.ASInit2=25
				config.ASInit3=25
				config.ASInit4=25

		elif config.data_id==13:
			if config.scaled:
				config.f1=19.0 
				config.f2=13.0 
				config.f3=11. 
				config.f4=8. 
				config.demandMu=3
				config.demandSigma=2
				config.ILInit1=2
				config.ILInit2=2
				config.ILInit3=2
				config.ILInit4=2
				config.AOInit1=2
				config.AOInit2=2
				config.AOInit3=2
				config.AOInit4=2
				config.ASInit1=2
				config.ASInit2=2
				config.ASInit3=2
				config.ASInit4=2
				config.actionUp=5
				config.actionLow=-5
			else:
				config.f1=272.0 
				config.f2=179.0 
				config.f3=166. 
				config.f4=113. 
				config.demandMu=37
				config.demandSigma=27
				config.ILInit1=40
				config.ILInit2=40
				config.ILInit3=40
				config.ILInit4=40
				config.AOInit1=40
				config.AOInit2=40
				config.AOInit3=40
				config.AOInit4=40
				config.ASInit1=40
				config.ASInit2=40
				config.ASInit3=40
				config.ASInit4=40
				config.actionUp=8
				config.actionLow=-8

		elif config.data_id==22:
			if config.scaled:
				config.f1=14.0 
				config.f2=9.0 
				config.f3=9. 
				config.f4=5. 
				config.demandMu=2
				config.demandSigma=2
				config.ILInit1=2
				config.ILInit2=2
				config.ILInit3=2
				config.ILInit4=2
				config.AOInit1=2
				config.AOInit2=2
				config.AOInit3=2
				config.AOInit4=2
				config.ASInit1=2
				config.ASInit2=2
				config.ASInit3=2
				config.ASInit4=2
			else:
				config.f1=71.0 
				config.f2=44.0 
				config.f3=42. 
				config.f4=28. 
				config.demandMu=10
				config.demandSigma=7
				config.ILInit1=10
				config.ILInit2=10
				config.ILInit3=10
				config.ILInit4=10
				config.AOInit1=10
				config.AOInit2=10
				config.AOInit3=10
				config.AOInit4=10
				config.ASInit1=10
				config.ASInit2=10
				config.ASInit3=10
				config.ASInit4=10

	elif config.demandDistribution == 4:
		config.actionUp=5
		config.actionLow=-5
		config.ch1=1.0
		config.ch2=0.75 
		config.ch3=0.5 
		config.ch4=0.25 
		config.cp1=10.0
		config.cp2=.0 
		config.cp3=.0 
		config.cp4=.0 
		config.leadRecItem1=2 
		config.leadRecItem2=2 
		config.leadRecItem3=2 
		config.leadRecItem4=2 
		config.leadRecOrder1=2 
		config.leadRecOrder2=2 
		config.leadRecOrder3=2 
		config.leadRecOrder4=1
		config.ifOptimalSolExist=True 
		config.testRepeatMid = 200 

		if config.data_id==5:
			if config.scaled:
				config.f1=21.0 
				config.f2=16.0 
				config.f3=16. 
				config.f4=11. 
				config.demandMu=4
				config.demandSigma=2
				config.ILInit1=4
				config.ILInit2=4
				config.ILInit3=4
				config.ILInit4=4
				config.AOInit1=4
				config.AOInit2=4
				config.AOInit3=4
				config.AOInit4=4
				config.ASInit1=4
				config.ASInit2=4
				config.ASInit3=4
				config.ASInit4=4

			else:
				config.f1=104.0 
				config.f2=82.0 
				config.f3=79. 
				config.f4=55. 
				config.demandMu=18
				config.demandSigma=7
				config.ILInit1=18
				config.ILInit2=18
				config.ILInit3=18
				config.ILInit4=18
				config.AOInit1=18
				config.AOInit2=18
				config.AOInit3=18
				config.AOInit4=18
				config.ASInit1=18
				config.ASInit2=18
				config.ASInit3=18
				config.ASInit4=18

		if config.data_id==34:
			if config.scaled:
				config.f1=18.0 
				config.f2=15.0 
				config.f3=14. 
				config.f4=10. 
				config.demandMu=4
				config.demandSigma=2
				config.ILInit1=4
				config.ILInit2=4
				config.ILInit3=4
				config.ILInit4=4
				config.AOInit1=4
				config.AOInit2=4
				config.AOInit3=4
				config.AOInit4=4
				config.ASInit1=4
				config.ASInit2=4
				config.ASInit3=4
				config.ASInit4=4

			else:
				config.f1=144.0 
				config.f2=114.0 
				config.f3=110. 
				config.f4=78. 
				config.demandMu=26
				config.demandSigma=10
				config.ILInit1=26
				config.ILInit2=26
				config.ILInit3=26
				config.ILInit4=26
				config.AOInit1=26
				config.AOInit2=26
				config.AOInit3=26
				config.AOInit4=26
				config.ASInit1=26
				config.ASInit2=26
				config.ASInit3=26
				config.ASInit4=26

		if config.data_id==46:
			if config.scaled:
				config.f1=21.0 
				config.f2=16.0 
				config.f3=18. 
				config.f4=12. 
				config.demandMu=4
				config.demandSigma=2
				config.ILInit1=4
				config.ILInit2=4
				config.ILInit3=4
				config.ILInit4=4
				config.AOInit1=4
				config.AOInit2=4
				config.AOInit3=4
				config.AOInit4=4
				config.ASInit1=4
				config.ASInit2=4
				config.ASInit3=4
				config.ASInit4=4

			else:
				config.f1=320.0 
				config.f2=259.0 
				config.f3=248. 
				config.f4=178. 
				config.demandMu=59
				config.demandSigma=20
				config.ILInit1=59
				config.ILInit2=59
				config.ILInit3=59
				config.ILInit4=59
				config.AOInit1=59
				config.AOInit2=59
				config.AOInit3=59
				config.AOInit4=59
				config.ASInit1=59
				config.ASInit2=59
				config.ASInit3=59
				config.ASInit4=59

	return config



#def main(config, beerGame):
def main(config):


	random.seed(10) 

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
	else:
		if config.demandDistribution == 0:
			direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(config.demandUp)+'-'+str(config.maxEpisodesTrain)+'.npy')
			if not os.path.exists(direc):
				direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(config.demandUp)+'.npy')
		elif config.demandDistribution == 1:
			direc = os.path.realpath(adsr+str(config.demandDistribution)+'-'+str(int(config.demandMu))+'-'+str(int(config.demandSigma))+'.npy')
		elif config.demandDistribution == 2:
			direc = os.path.realpath(adsr+str(config.demandDistribution)+'.npy')	
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
	demandTs = np.load(direc)	
	print("loaded test set=", direc)

	if config.maxEpisodesTrain == -1:
		# demandTs = np.load('../data/demandTs003.npy')
		demandTs = np.load('../data/demandTs110.npy')
		demandTs = np.load('../data/demandTs2-50000.npy')
		print("loaded auxiliary data")

	# set the sensitivity analysis configs and update the rest of configs regarding new ones. 
	config = set_config(config)
	config = update_config(config)
	
	
	config.cpch_deviation = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
	# config.lead_deviation = [-0.2, -0.1, 0.1, 0.2]
	# config.cpch_deviation = [0]
	config.lead_deviation = [1]

	original_cp = config.c_p 
	original_ch = config.c_h 

	for cp_deviation in config.cpch_deviation:
		for ch_deviation in config.cpch_deviation:
			# if cp_deviation in [-0.2, -0.1, 0.1, 0.2]:
			# 	if ch_deviation in [-0.2, -0.1, 0.1, 0.2]:
			# 		continue
			for l_deviation in config.lead_deviation:
				# start the Matlab connection to get the optimal solution.
				mlab = Matlab()
				mlab = Matlab(executable='/usr/local/matlab/latest/bin/matlab')
				mlab.start()
				for brain in [3,4,5,6,7,8,9,10]:
					# reset the setting with the current brain type 
					# update the cost coefficients 
					config.brainTypes = brain
					# config = set_config(config)
					config = update_config(config)

					config.c_p = [_*(1+cp_deviation) for _ in original_cp]
					config.c_h = [_*(1+ch_deviation) for _ in original_ch]

					res = mlab.run_func('../BS_matlab_codes/pycaller.m', 
						{'demandDistribution': config.demandDistribution, 'arg': 10, 'cp_dev':(1.+cp_deviation), 
						'ch1': config.c_h[0], 'ch2': config.c_h[1], 
						'ch3': config.c_h[2], 'ch4': config.c_h[3]})
					config.f = np.squeeze(res.items()[1][1])
					if len(res.items()[1][1]) == 0:
						continue

					print("SUMMARY; ", "cp_deviation; ", cp_deviation, "; ch_deviation; ", ch_deviation, \
						"; l_deviation; ", l_deviation, "; cp=", config.c_p, "; ch=", config.c_h, \
						"; brain; ", brain, "; BS=", config.f)
					print("---------------------------------------------------------------------------------------")
					# initilize an instance of Beergame
					beerGame = clBeerGame(config)
					# run the tests
					beerGame.doTestMid(demandTs[0:config.testRepeatMid])

				mlab.stop()

		
if __name__ == '__main__':
	# load parameters
	config, unparsed = get_config()

	# run main
	# tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
	#tf.app.run(main=main)
	main(config)
