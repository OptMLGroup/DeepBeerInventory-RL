# A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization

The code of the paper `A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization` is presented at this project. This code does not work with TensorFlow 2+. 
The `main.py` is the file to call to start the training. `BGAgent.py` provides the beer-game agent which involves all the properties and functionality of an agent. `clBeergame.py` instanciates the agents and runs the beer-game simulation. Also, once the number of observations in the replay buffer filled by the minimum requirement, it calls the train-step of the SRDQN algorithm. The DNN approximator and SRDQN algorithm are implemented in `SRDQN.py`. `config.py` introduce all arguments and their default values, as well as some functions to properly build the simulation scenarios for different instances of the game. In the following the procedure to run the training and setting different values for the arguments is described. 

## Some Notations
Each agent can use either of the `srdqn`, `bs`, `Ster`, or `Rnd` algorithms to decide about the action (order quantity). So, there are 256 combination of agent-types from which we consider 23 cases in this study. To determine each of these cases, we have used `config.gameConfig` to select one of pre-defined type of four agents in the game. For example, `config.gameConfig=3`, sets `config.agentTypes = ["srdqn", "bs","bs","bs"]`, in which the retailer follows the `srdqn` algorithm and the rest of agents use the base-stock policy to decide for the order quantity. The main `gameConfig` are as below:

Base-stock co-players 

	if config.gameConfig == 3: 
		config.agentTypes = ["srdqn", "bs","bs","bs"]
	if config.gameConfig == 4: 
		config.agentTypes = ["bs", "srdqn","bs","bs"]
	if config.gameConfig == 5: 
		config.agentTypes = ["bs", "bs","srdqn","bs"]
	if config.gameConfig == 6: 
		config.agentTypes = ["bs", "bs","bs","srdqn"]
Sterman co-players 

	if config.gameConfig == 7: 
		config.agentTypes = ["srdqn", "Strm","Strm","Strm"]
	if config.gameConfig == 8: 
		config.agentTypes = ["Strm", "srdqn","Strm","Strm"]
	if config.gameConfig == 9: 
		config.agentTypes = ["Strm", "Strm","srdqn","Strm"]
	if config.gameConfig == 10: 
		config.agentTypes = ["Strm", "Strm","Strm","srdqn"]
Random co-players 

	if config.gameConfig == 11: 
		config.agentTypes = ["srdqn", "rnd","rnd","rnd"]
	if config.gameConfig == 12: 
		config.agentTypes = ["rnd", "srdqn","rnd","rnd"]
	if config.gameConfig == 13: 
		config.agentTypes = ["rnd", "rnd","srdqn","rnd"]
	if config.gameConfig == 14: 
		config.agentTypes = ["rnd", "rnd","rnd","srdqn"]

The full list of all `gameConfig` is defined in `setAgentType()` function in `config.py`. 

Since the `d+x` rule is used to train the `SRDQN` model, we use the upper and lower limit for `x`. `config.actionLow` and `config.actionUp` are used to set these values.  

In addition, for each agent one can determine the lead time for receving order as well as receving the shimpement via `config.leadRecItem1`, `config.leadRecItem2`, `config.leadRecItem3`, `config.leadRecItem4` and `config.leadRecOrder1`, `config.leadRecOrder2`, `config.leadRecOrder3`, `config.leadRecOrder4` for four agents. Similarly, the initial inventory level, initial arriving order, and initial arriving shipment can be set by `config.ILInit1`, `config.ILInit2`, `config.ILInit3`, `config.ILInit4`, `config.AOInit1`, `config.AOInit2`, `config.AOInit3`, `config.AOInit4`, `config.ASInit1`, `config.ASInit2`, `config.ASInit3`, `config.ASInit4`, respectively for the four agents.

`config.maxEpisodesTrain` determines the number of episodes to train the `srdqn` agent. 

TO run the baseStock policy (`bs`), you need to set the value of the base-stock level for each agent by `config.f1`, `config.f2`, `config.f3`, `config.f4`. We obtained those values by running the Clark-Scarf algorithm for each instance. 

## unzip the data
`data.zip` includes all the required dataset to train the model on basic case, literature cases, basket dataset, and forecasting dataset. Unzipping this file creates `data` directory, in which there is a python file (`createDemand.py`) as well as the mentioned datasets. `createDemand.py` can be used to create datasets of any size for the literature cases.

## Train the basic model
The basic model used the Uniform distribution `U[0,2]` with action space of `{-2, -1, 0, 1, 2}`. All the default values are set to run this experiment for the case that `srdqn` plays the retailer and other agents follow base-stock policy. For any other case the training can be started by setting the corresponding arguments. For example, to train a `srdqn` Warehouse with the initial inventory of 10 units which plays with Sterman co-players, the following line can be used to run the training for 50000 episodes: 

	python main.py --gameConfig=8 --maxEpisodesTrain=50000 config.ILInit2=10 --batchSize=128

## Train the literature cases 
To train each of the literature cases, first you need to set `config.demandDistribution`, `actionUp`, and `actionLow`, as well as the other parameter for the agents as following:

For U[0,8]:

	python main.py --demandDistribution=0 --demandUp=9  --actionUp=8  --actionLow=-8 --ch1=0.5 --ch2=0.5 --ch3=0.5 --ch4=0.5 --cp1=1.0 --cp2=1.0 --cp3=1.0 --cp4=1.0 --f1=19.0 --f2=20.0 --f3=20.0 --f4=14.0  --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --ILInit1=12 --ILInit2=12 --ILInit3=12 --ILInit4=12 --AOInit1=4 --AOInit2=4 --AOInit3=4 --AOInit4=4 --ASInit1=4 --ASInit2=4 --ASInit3=4 --ASInit4=4 --gameConfig=6 

For N(10,2):

	python main.py --demandDistribution=1 --demandMu=10  --demandSigma=2 --actionUp=5  --actionLow=-5 --ch1=1 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0 --cp3=0 --cp4=0 --f1=48.0 --f2=43.0 --f3=41.0 --f4=30.0 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --ILInit1=10 --ILInit2=10 --ILInit3=10 --ILInit4=10 --AOInit1=10 --AOInit2=10 --AOInit3=10 --AOInit4=10 --ASInit1=10 --ASInit2=10 --ASInit3=10 --ASInit4=10 --gameConfig=6

For C(4,8):

	python main.py --demandDistribution=2 --actionUp=8  --actionLow=-8 --ch1=0.5 --ch2=0.5 --ch3=0.5 --ch4=0.5 --cp1=1.0 --cp2=1.0 --cp3=1.0 --cp4=1.0 --demandUp=9 --f1=32.0 --f2=32.0 --f3=32.0 --f4=24.0 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --ILInit1=12 --ILInit2=12 --ILInit3=12 --ILInit4=12 --AOInit1=4 --AOInit2=4 --AOInit3=4 --AOInit4=4 --ASInit1=4 --ASInit2=4 --ASInit3=4 --ASInit4=4 --gameConfig=6

## Train the basket dataset 


## Train the forecasting dataset 


## Use Transfer Learning 

	python main.py --gameConfig=3  --iftl=True --ifUsePreviousModel=True

use the transfer learning 
	
	python main.py --gameConfig=3  --iftl=True --maxEpisodesTrain=0 --ifUsePreviousModel=True
to test to see how four stand-alone pre-trained agents work together. 


## Other utilities 

	--ifSaveFigure=True --if_reset_replay=False 