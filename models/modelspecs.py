from brian2 import *
from models.modelequations import *

import traceback as tb
import pickle as pk

# CONSTANTS
## Experiment parameters
HABITUATION_TIME = 10000 # Idle time before the session starts
MAX_TRIAL_TIME = 10000 # Maximum trial duration in ms
LEARNING_TIME = 200 # Time of presentation of reward in ms
INTERTRIAL_TIME = 500 # Rest time betwen trial presentation
FAKE_TRIAL_TIME = 1000 # Time for the PartialExperiment
P_REWARDS = [.25, .5, 1.]

## Currently unused
INTERSESSION_TIME = 0 # Rest time between sessions

## Architecture
NO_CHANNELS = 3 # Number of segregated channels/distinct inputs
CHAN_SIZE = 15 # Number of neurons in each
pSize = NO_CHANNELS*CHAN_SIZE
P_CONNECT = .7 # Connection probability
POPULATIONS = ['Target', 'Estimation', 'DecisionExc', 'DecisionInh', 'ctlExc', 'ctlInh', 'Dopamine', 'Readout','Reward']

## Neuron parameters
EXC_RATIO = '.2' # NMDA/AMPA ratio
E_EXC = 0*mV # Resting potential for excitation
E_INH = -70*mV # Resting potential for inhibition
SIGMA = 1.*mV # Intrinsic noise; .5 was suggested
REFRACTORY = 2*ms # Standard refractory period


neuronPars = ['E_L',  'R',     'tau', 'V_th', 'V_res', 'tau_ampa', 'tau_nmda', 'tau_gaba', 'I_ext']
genPars =    [-70*mV, 50*Mohm, 15*ms, -55*mV, -70*mV,  3*ms,       100*ms,     5*ms] #,       .20*nA ]
strPars =    [-80*mV, 60*Mohm, 10*ms, -60*mV, -80*mV,  3*ms,       100*ms,     5*ms] #,       .23*nA ]
DAPars =     [-65*mV, 30*Mohm, 10*ms, -50*mV, -65*mV,  3*ms,       100*ms,     6*ms] #,       .13*nA ]
parInits = [('V','V_res + rand()*(V_th-V_res)'), ('g_ampa',0*siemens), ('g_nmda',0*siemens), ('g_gaba',0*siemens), ('DA', 0), ('DATonic', 0)]

extCurrents = {'Target':.33*nA, 'Estimation': .28 * nA, 'DecisionExc':.33 * nA, 'DecisionInh':.295 * nA, 'ctlExc': .24*nA, 'ctlInh': .31*nA, 'Dopamine': .467*nA}
weights = {'TarEst': .04, 'excDop':3.*nS, 'inhDop':2.*nS, 'avgValue':.4*nS }
DAEffects = {'lbd':1, 'eta':{}}