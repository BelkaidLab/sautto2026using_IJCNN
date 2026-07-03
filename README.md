# Using Disinhibition versus Direct Control (Sautto et al., IJCNN, 2026)
This repository contains code and materials related to article Sautto et al., IJCNN, 2026.

## About the article
[![DOI](https://doi.org/TBD)](https://doi.org/TBD) _(TBD)_

### Using Disinhibition versus Direct Control in a Spiking Neural Model of Dopamine-Driven Reinforcement Learning
**Abstract:** Dopaminergic signalling is central to value learning and decision making. It has been observed that multiple pathways with different patterns of connectivity project to midbrain dopaminergic neurons, some involving direct excitatory projections while others involve disinhibition. However, the respective contributions of these patterns to dopamine control, and their computational and functional advantages remain unclear. In the current work we simulate and evaluate two fully spiking neural models of dopaminergic control, based either solely on disinhibition, or solely on direct inhibitory and excitatory projections. We compare these models in terms of their engineering properties, their resulting spiking profiles, and their ability to successfully acquire representations of expected value in a 3-armed bandit task. We find that both models are able to operate at an asynchronous-irregular firing regime, but that the firing profile of the direct integration model is less resilient to disruption and more sensitive to incoming signals. In addition, the disinhibition model performs better in the learning task. We conclude that while the direct model is more parsimonious, disinhibition-based control remains advantageous in the operational context. Our results have implications for the study of decision-making brain circuits as well as for the design of brain-inspired systems.

### Citation
_(TBD)_

## About the code
This codebase is programmed in Python, and requires dependencies as listed in the file 'environment.yml', which can be used to generate an appropriate virtual environment.

To run this code and generate the results shown in the paper, once all dependencies have been installed, the following operations will need to be performed:
1. Open the file simulation.py and set the 'mode' variable to 2. Run the python script to generate the parameter variations used for the random search. 
2. Set the 'mode' variable to 1 and close the file.
3. Open the file 'Run.sh' and modify line 14 so that the fist token points to the python executable of your virtual environment, and the second one to the file 'simulation.py'
4. To run Experiment 1, leave lines 3 and 6 uncommented, and lines 4 and 7 commented; then run the script.
5. To run Experiment 2, comment out lines 3 and 6, and uncomment lines 4 and 7; then run the script.

The code is based on the spiking neural network model found developed by Belkaid and Krichmar (2020). The project overall is inspired by the animal study published by Naudé et al., 2016.

### References 
- Belkaid and Krichmar, "Modeling Uncertainty-Seeking Behavior Mediated by Cholinergic Influence on Dopamine", Neural Networks, 2020.
- Naudé et al., "Nicotinic receptors in the ventral tegmental area promote uncertainty-seeking", Nature Neuroscience, 2016.




