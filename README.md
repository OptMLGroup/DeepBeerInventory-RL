# A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization

The code of the paper `A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization` is presented at this repository. The paper is available online in https://pubsonline.informs.org/doi/abs/10.1287/msom.2020.0939. The code works with `Python2.7` and `Python3.4-Python3.7`. For more information see the list of the requirments (You can install them `pip install -r requirements.txt`). 
The `main.py` is the file to call to start the training. `BGAgent.py` provides the beer-game agent which involves all the properties and functionality of an agent. `clBeergame.py` instanciates the agents and runs the beer-game simulation. Also, once the number of observations in the replay buffer filled by the minimum requirement, it calls the train-step of the SRDQN algorithm. The DNN approximator and SRDQN algorithm are implemented in `SRDQN.py`. `config.py` introduce all arguments and their default values, as well as some functions to properly build the simulation scenarios for different instances of the game. In the following the procedure to run the training and setting different values for the arguments is described. 

###Play beer-game and compare your result with AI!
You can play the beer-game and compare your result on the same game with the result that our RL algorithm achieves. See https://beergame.opexanalytics.com/
 

Note that this code does not work with TensorFlow 2+. 
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
For the basket dataset you need to set `config.demandDistribution=3`, and then `config.data_id` can be either `6, 13`, or `22`. For training with the scaled dataset, which is reported in the paper, `config.scaled=True` is required too. See the following commands for three cases: 

	python main.py --demandDistribution=3 --data_id=6 --demandMu=3 --demandSigma=2 --demandUp=3 --actionUp=5 --actionLow=-5 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --scaled=True --ch1=1.0 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0.0 --cp3=0.0 --cp4=0.0 --f1=19.0 --f2=12.0 --f3=12.0 --f4=8.0 --ILInit1=3 --ILInit2=3 --ILInit3=3 --ILInit4=3 --AOInit1=3 --AOInit2=3 --AOInit3=3 --AOInit4=3 --ASInit1=3 --ASInit2=3 --ASInit3=3 --ASInit4=3

	python main.py --demandDistribution=3 --data_id=13 --demandMu=3  --demandSigma=2  --demandUp=3 --actionUp=5 --actionLow=-5 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --scaled=True --ch1=1.0 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0.0 --cp3=0.0 --cp4=0.0 --f1=19.0 --f2=13.0 --f3=11.0 --f4=8.0 --ILInit1=3  --ILInit2=3  --ILInit3=3  --ILInit4=3  --AOInit1=3  --AOInit2=3  --AOInit3=3  --AOInit4=3  --ASInit1=3  --ASInit2=3  --ASInit3=3  --ASInit4=3 

	python main.py --demandDistribution=3 --data_id=22 --demandMu=2  --demandSigma=2  --demandUp=3 --actionUp=5 --actionLow=-5       --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --scaled=True --ch1=1.0 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0.0 --cp3=0.0 --cp4=0.0 --f1=14.0 --f2=9.0 --f3=9.0 --f4=5.0 --ILInit1=2  --ILInit2=2  --ILInit3=2  --ILInit4=2  --AOInit1=2  --AOInit2=2  --AOInit3=2  --AOInit4=2  --ASInit1=2  --ASInit2=2  --ASInit3=2  --ASInit4=2 

## Train the forecasting dataset 
For the forecasting dataset you need to set `config.demandDistribution=4`, and then `config.data_id` can be either `5, 34`, or `46`. For training with the scaled dataset, which is reported in the paper, `config.scaled=True` is required too. See the following commands for three cases: 

	python main.py --demandDistribution=4 --data_id=5 --demandMu=4 --demandSigma=2 --demandUp=3 --actionUp=5 --actionLow=-5 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --scaled=True --ch1=1.0 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0.0 --cp3=0.0 --cp4=0.0 --f1=21.0 --f2=16.0 --f3=16.0 --f4=11.0 --ILInit1=4  --ILInit2=4  --ILInit3=4  --ILInit4=4  --AOInit1=4  --AOInit2=4  --AOInit3=4  --AOInit4=4  --ASInit1=4  --ASInit2=4  --ASInit3=4  --ASInit4=4 

	python main.py --demandDistribution=4 --data_id=34 --demandMu=4 --demandSigma=2 --demandUp=3 --actionUp=5 --actionLow=-5 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --scaled=True --ch1=1.0 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0.0 --cp3=0.0 --cp4=0.0 --f1=18.0 --f2=15.0 --f3=14.0 --f4=10.0 --ILInit1=4  --ILInit2=4  --ILInit3=4  --ILInit4=4  --AOInit1=4  --AOInit2=4  --AOInit3=4  --AOInit4=4  --ASInit1=4  --ASInit2=4  --ASInit3=4  --ASInit4=4 

	python main.py --demandDistribution=4 --data_id=46 --demandMu=4 --demandSigma=2 --demandUp=3 --actionUp=5 --actionLow=-5 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --scaled=True --ch1=1.0 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0.0 --cp3=0.0 --cp4=0.0 --f1=21.0 --f2=16.0 --f3=18.0 --f4=12.0 --ILInit1=4  --ILInit2=4  --ILInit3=4  --ILInit4=4  --AOInit1=4  --AOInit2=4  --AOInit3=4  --AOInit4=4  --ASInit1=4  --ASInit2=4  --ASInit3=4  --ASInit4=4 

## Use Transfer Learning 
We have provided the trained model of the basic model which are used in the transfer learning section. The saved models are available in `pre_model\uniform\0-3\brainX` in which `X` is in `{3, 4, 5, 6}`. The value of `X` follows the same pattern as of `config.gameConfig`. To train a new with either of these trained models, you need to set `config.tlBaseBrain` that determines which trained should be used as the base model. For example: 

	python main.py --gameConfig=3  --iftl=True --ifUsePreviousModel=True  --tlBaseBrain=3 --baseDemandDistribution=0

Besides, if you trained a model with another demand distribution, e.g., `N(10,2)`, you need to move the saved models into `pre_model\normal\10-2\brainX` and then for a new training set `config.baseDemandDistribution=1`. The `config.baseDemandDistribution` follows the same pattern as of `config.demandDistribution`. 

## Other utilities 
If you set `config.ifSaveFigure=True`, it saves the trajectories of inventory-level, reward, action, open-order, and order-upto-level for each agent in an episode. `config.saveFigIntLow` and `config.saveFigIntUp` determine the range of eprisode to save the figures. 

Setting `config.ifsaveHistInterval=True`, activate saving of trajectory of the received order, received shipment, inventory-level, reward, action, open-order, and order-upto-level for each agent in an episode. With this argument, you need to determine the interval between every two epsiode to save the history with `config.saveHistInterval`.


## Paper citation
If you used this code for your experiments or found it helpful, consider citing the following paper:

	@article{oroojlooyjadid2017deep,
	title={A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization},
	author={Oroojlooyjadid, Afshin and Nazari, MohammadReza and Snyder, Lawrence and Tak{\'a}{\v{c}}, Martin},
	journal = {Manufacturing \& Service Operations Management},
	volume = {0},
	number = {0},
	pages = {null},
	year = {0},
	doi = {10.1287/msom.2020.0939},

	URL = { 
		https://doi.org/10.1287/msom.2020.0939

	},
	eprint = { 
		https://doi.org/10.1287/msom.2020.0939

	}
	  year={2021}
	}
