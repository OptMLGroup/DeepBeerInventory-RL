# A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization

The code of the paper `A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization` is presented at this project.

## unzip the data

## Train the basic model

## Train the literature cases 

## Train the basket dataset 

## Train the forecasting dataset 


## Use Transfer Learning 
use the transfer learning 
	
	python main.py --brainTypes=3  --iftl=True --maxEpisodesTrain=0 --ifUsePreviousModel=True
to test to see how four stand-alone pre-trained agents work together. 






Call classic demand with:
	python main.py --maxEpisodesTrain=0 --actionUp=8  --actionLow=-8 --batchSize=64 --epsilonBeg=0.9 --distCoeff=2 --lr0=0.00025 --NoHiLayer=3 --ch1=0.5 --ch2=0.5 --ch3=0.5 --ch4=0.5 --cp1=1.0 --cp2=1.0 --cp3=1.0 --cp4=1.0 --demandDistribution=2 --demandUp=9 --iftl=True --ifUsePreviousModel=True --NoFixedLayer=3 --f1=32.0 --f2=32.0 --f3=32.0 --f4=24.0  --ifSaveFigure=True --if_reset_replay=False --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --demandMu=0  --demandSigma=1 --action_step=1  --ILInit1=12 --ILInit2=12 --ILInit3=12 --ILInit4=12 --AOInit1=4 --AOInit2=4 --AOInit3=4 --AOInit4=4 --ASInit1=4 --ASInit2=4 --ASInit3=4 --ASInit4=4 --ifOptimalSolExist=True --MultiAgent=False --multAgUpCnt=100 --gamma=1 --Ttest=99 --iftl=False --ifUsePreviousModel=True --brainTypes=6


Call normal(10,2) demand with:
	python main.py --maxEpisodesTrain=0 --actionUp=5  --actionLow=-5 --batchSize=64 --epsilonBeg=0.9 --distCoeff=2 --lr0=0.00025 --NoHiLayer=3 --node1=180 --node2=130 --node3=61 --ch1=1 --ch2=0.75 --ch3=0.5 --ch4=0.25 --cp1=10.0 --cp2=0 --cp3=0 --cp4=0 --demandDistribution=1 --demandUp=9 --ifUsePreviousModel=True --NoFixedLayer=3 --f1=48.0 --f2=43.0 --f3=41.0 --f4=30.0 --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --demandMu=10  --demandSigma=2 --action_step=1  --ILInit1=10 --ILInit2=10 --ILInit3=10 --ILInit4=10 --AOInit1=10 --AOInit2=10 --AOInit3=10 --AOInit4=10 --ASInit1=10 --ASInit2=10 --ASInit3=10 --ASInit4=10 --ifOptimalSolExist=True --multAgUpCnt=100 --gamma=1 --Ttest=99 --iftl=False --ifUsePreviousModel=True --brainTypes=6


Call unifroms[0,8] demand with:
	python main.py --maxEpisodesTrain=0 --actionUp=8  --actionLow=-8 --batchSize=64 --epsilonBeg=0.9 --distCoeff=2 --lr0=0.00025 --NoHiLayer=3 --node1=180 --node2=130 --node3=61 --ch1=0.5 --ch2=0.5 --ch3=0.5 --ch4=0.5 --cp1=1.0 --cp2=1.0 --cp3=1.0 --cp4=1.0 --demandDistribution=0 --demandUp=9 --iftl=False --ifUsePreviousModel=True --NoFixedLayer=3 --f1=19.0 --f2=20.0 --f3=20.0 --f4=14.0  --leadRecItem1=2 --leadRecItem2=2 --leadRecItem3=2 --leadRecItem4=2 --leadRecOrder1=2 --leadRecOrder2=2 --leadRecOrder3=2 --leadRecOrder4=1 --demandMu=0  --demandSigma=1 --action_step=1  --ILInit1=12 --ILInit2=12 --ILInit3=12 --ILInit4=12 --AOInit1=4 --AOInit2=4 --AOInit3=4 --AOInit4=4 --ASInit1=4 --ASInit2=4 --ASInit3=4 --ASInit4=4 --ifOptimalSolExist=True --gamma=1 --Ttest=99 --brainTypes=6 
