from models.modelspecs import *

class DACFullInh:
    def makePopulation(self, name, number, parameters, parNames = neuronPars, inits = parInits, equation = eqLIF, reset = 'V = V_res', threshold = 'V > V_th'):
        self.net.add(NeuronGroup(number, equation, 'euler', threshold=threshold, reset=reset, refractory=REFRACTORY, name = name))
        for parName, parValue in list(zip(parNames, parameters))+inits: setattr(self.net[name], parName, parValue)

    def makeProjection(self, name, source, target, mod, syn, parameters, cond = '((i//CHAN_SIZE) == (j//CHAN_SIZE)) and rand() < '+str(P_CONNECT), delay = 0):

        self.net.add(Synapses(self.net[source], self.net[target], model = mod, on_pre=syn, name = name, delay=delay*ms))
        self.net[name].connect(cond)
        for parName, parValue in parameters: setattr(self.net[name], parName, parValue)

    def __init__(self, noTrials, eta = {}, # Experiment parameters
                 MINIMAL_RECORDING = False, RECORD_CONTINUOUS = True, # recording flags
                 steady = True, # Wether or not to have a steady state epoch
                 extCurrents = extCurrents,
                 weights = weights,
                 runType = "Trials"): # Trials, Unrewarded, Rewarded
        self.noTrials = noTrials
        self.RECORD_CONTINUOUS = RECORD_CONTINUOUS
        self.MINIMAL_RECORDING = MINIMAL_RECORDING
        self.monitors = {}
        self.weights = []
        self.net = Network()
        self.runType = runType
        self.toRecord = ['Estimation', 'Dopamine', 'Reward', 'ctlExc', 'ctlInh']

        BrianLogger.log_level_error()

        # Neural populations
        ## Core Network
        self.makePopulation('Target', pSize, genPars+[0*nA, 0, extCurrents['Target']], neuronPars + ['gamble', 'targetExt'], 
                            equation='gamble : 1\ntargetExt : amp\n'+eqLIF)
        self.makePopulation('Estimation', pSize, strPars+[extCurrents['Estimation'], 0], neuronPars+['chosen'], 
                            equation='alpha : siemens\nchosen : 1\n'+eqLIF) # alpha is here to be changed by the controller after maxTrials
        # Can contain one more channel (+CHAN_SIZE) to test competition by tying mutual inhibition to the empty channel and seeing the difference
        readoutEq = 'gamble : 1 \n \n fire : 1 \n dwinner/dt = ((gamble+1+int(rand()>.5))%NO_CHANNELS - winner)/ms : 1'
        readoutReset = 'gamble = (gamble+1)%NO_CHANNELS \n fire = 0' # Fire set every 1200ms
        self.makePopulation('Readout', 1, [0, 1, 1], ['gamble', 'winner', 'fire'], [], equation=readoutEq, 
                            threshold='(t > (HABITUATION_TIME+FAKE_TRIAL_TIME)*ms) and (fire > .5)', reset=readoutReset)

        ## DA Subsystem
        self.makePopulation('ctlExc', pSize, genPars+[extCurrents['ctlExc']])
        self.makePopulation('ctlInh', pSize, genPars+[extCurrents['ctlInh']])
        self.makePopulation('Dopamine', pSize, DAPars+[extCurrents['Dopamine']])
        ## Environment
        self.makePopulation('Reward', 1, genPars+[0*amp], equation='choice : 1\nrewardExt : amp\n'+eqLIF, inits=parInits+[('choice',-1)])
        self.makePopulation('Controller', 1, [0, noTrials], ['trialCount', 'maxTrials'], [], 'trialCount : 1 \n maxTrials : 1', 'trialCount = 0', 'trialCount >= maxTrials')

        # Projections
        ## Core Network 
        self.makeProjection('TarEst', 'Target', 'Estimation', learningModel, excSyn, # stochastic initialisation, var=.04
                            [('w', '('+str(weights['TarEst'])+'+.08*(rand()-.5))*nS'), ('ExcRatio',EXC_RATIO), 
                             ('lbd', DAEffects['lbd']), ('eta', eta.get('TarEst', 0))])
        ## DA Subsystem: ideally, the more neurons, the less weights, the smoother
        rewDopSyn = 'g_ampa += int((j//CHAN_SIZE)==choice_pre)*w \n g_nmda += int((j//CHAN_SIZE) == choice_pre)*w*ExcRatio'
        if runType == "Rewarded": rewDopSyn = excSyn

        self.makeProjection('excDop', 'ctlExc', 'ctlInh', inhModel, inhSyn, [('w',weights['excDop']), ('eta', eta.get('excDop', 0))])
    #    self.makeProjection('ctlBalance', 'ctlInh', 'ctlExc', inhMod, inhSyn, {[('w',1.25 *nS), ('eta', eta.get('TarEst', 0))]})
        self.makeProjection('inhDop', 'ctlInh', 'Dopamine', inhModel, inhSyn, [('w',weights['inhDop']), ('eta', eta.get('inhDop', 0))]) 
        self.makeProjection('avgValue', 'Estimation', 'ctlExc', excModel, excSyn, 
                            [('w',weights['avgValue']), ('eta', eta.get('avgValue', 0))], 'rand() < '+str(P_CONNECT))
        self.makeProjection('rewDop', 'Reward', 'ctlExc', excModel, rewDopSyn, [('eta', eta.get('rewDop', 0))], True)
        self.makeProjection('expDop', 'Estimation', 'ctlInh', excModel, 
                            'g_ampa += chosen_pre*w \n g_nmda += chosen_pre*w*ExcRatio', 
                            [('eta', eta.get('expDop', 0))])
        self.makeProjection('dopEst', 'Dopamine', 'Estimation', '', 'DA += .001', []) # Parset1 has ~.001

        ## Environment
        ### Readout control
        self.makeProjection('nextFiring', 'Readout', 'Readout', '', 'fire = 1', [],  True, LEARNING_TIME+INTERTRIAL_TIME+FAKE_TRIAL_TIME)
        ### Target control
        self.makeProjection('stopTrial', 'Readout', 'Target', '', 'I_ext = 0 * nA', [], True, LEARNING_TIME)
        self.makeProjection('startTrial', 'Readout', 'Target', '', 'gamble = (gamble+1)%NO_CHANNELS\nI_ext = int((j//CHAN_SIZE)!=gamble)*targetExt', [], True, LEARNING_TIME+INTERTRIAL_TIME)
        ### Rewards and trials
        # P_REWARDS RENDERED AS (0.125*winner**2 + 0.125*winner + 0.25)
        self.makeProjection('getReward', 'Readout', 'Reward', '', 'choice = winner_pre \n I_ext = int(rand()<=(0.125*choice**2 + 0.125*choice + 0.25))*rewardExt', [], True)
        self.makeProjection('stopReward', 'Readout', 'Reward', '', 'choice = -1 \n I_ext = 0*nA', [], True, LEARNING_TIME)
        self.makeProjection('getEst', 'Readout', 'Estimation', '', 'chosen = int(winner_pre == j//CHAN_SIZE)', [], True)
        self.makeProjection('stopEst', 'Readout', 'Estimation', '', 'chosen = 0', [], True, LEARNING_TIME+INTERTRIAL_TIME)
        self.makeProjection('countTrials', 'Readout', 'Controller', '', 'trialCount += 1', [], True, LEARNING_TIME+INTERTRIAL_TIME)
        self.makeProjection('steadyPhase', 'Controller', 'Estimation', '', 'alpha = 0*nS', [], True)

        # Activity
        self.net.add(SpikeMonitor(self.net['Readout'], 'winner', when = 'after_resets', name = 'spReadout'))
        self.monitors['spReadout'] = self.net['spReadout']
        # TODO: there is no other way, you have to generate TarEst connectivity separately, save it, and use it
#        self.net.add(StateMonitor(self.net['TarEst'], 'w', record=self.net['TarEst']['i>=0'], name='EstW'))
#        self.monitors['EstW'] = self.net['EstW']
        if not MINIMAL_RECORDING: 
            for layer in self.toRecord:
                self.net.add(SpikeMonitor(self.net[layer], name = 'sp'+layer))
                self.monitors['sp'+layer] = self.net['sp'+layer]
            #NOTE: weights/trial requires setting connectivity externally
            if RECORD_CONTINUOUS and self.runType == 'Trials':
                self.net.add(StateMonitor(self.net['Estimation'], 'DA', True, name='estDAPhasic'))
                self.monitors['estDAPhasic'] = self.net['estDAPhasic']
                self.net.add(StateMonitor(self.net['Estimation'], 'DATonic', True, name='estDATonic'))
                self.monitors['estDATonic'] = self.net['estDATonic']
        
        device.apply_run_args()

        # Running
        runTime = ms*((FAKE_TRIAL_TIME+LEARNING_TIME+INTERTRIAL_TIME)*noTrials+200)
        if runType=="Trials": self.net["Readout"].fire = 1
        self.net.run(HABITUATION_TIME*ms) # Needed for the DA levels
        self.net['Target'].I_ext = 'int((i//CHAN_SIZE)!=gamble)*targetExt'
        if runType=="Rewarded": 
            self.net['Reward'].I_ext = 'rewardExt'
            self.net['Estimation'].chosen = 1
        if runType == "Trials": self.net.run(runTime)  # Standalone needs to run full
        else: self.net.run(HABITUATION_TIME*ms) # Needed for the DA levels
        
        device.build(run=False)
        self.device = get_device()

    def runExperiment(self, rPath, session, pars, experiment = 0): 
        from brian2.devices import device_module
        device_module.active_device = self.device
        resDir = 'exp'+str(experiment)+'Result'+str(session)
        args = [
            (self.net['Reward'].rewardExt, pars[1]),
            (self.net['rewDop'].w, pars[2]),
            (self.net['expDop'].w, pars[3]),
            (self.net['Estimation'].alpha, pars[6])
        ]
        
        for pop in ['Target', 'Estimation', 'Dopamine','Reward', 'ctlExc', 'ctlInh']:
            args.append((self.net[pop].tauDA, pars[4]))
            args.append((self.net[pop].tauDATonic, pars[5]))
        
        self.device.run(run_args = dict(args), results_directory = resDir)
        

        if self.runType == "Trials":
            #NOTE: saving stocastic projections for weights analysis
            with open(rPath+'Connectivity/Session'+str(session), 'wb') as f:
                pk.dump({'src':np.array(self.net['TarEst'].i), 'trg':np.array(self.net['TarEst'].j)}, f)
            try:
                data = self.parseSessionMonitors(session)
                with open(rPath+'Recordings/Session'+str(session), 'wb') as f:
                    pk.dump(data, f)
            except Exception as ex:
                print(tb.format_exc())
                print('Could not parse Session '+str(session))
                return False
        else:
            with open(rPath+'Recordings/Session'+str(session), 'wb') as f:
                tmp = self.monitors['spDopamine'].spike_trains()
                pk.dump([ tmp[i] for i in range(len(tmp.keys())) ], f)

        return True

    ## Extracts data from the monitors and saves raw data;
    ## Uses local functions as scope for memory management
    def parseSessionMonitors(self, session):
        sessionData = []
        startTime = HABITUATION_TIME*ms
        mistrials = []
        count = 0

        # Separate into trials
        decisions = self.monitors['spReadout'].get_states()
        for c, time in enumerate(decisions['t']):
            endTime = time+(LEARNING_TIME+INTERTRIAL_TIME)*ms
            #print(startTime)
            #print(time)
            #print()
            decisionTime = time - startTime

            # When DT <=0 this is a mistrial and must be penalised
            # Possibly due to too high intrinsic excitability of the Readouts
            # The trial effects should be effectively be attributed to the last recorded c
            if decisionTime/ms <= 0: 
                print('Session'+str(session)+': mistrial in trial '+str(count)+' with DT='+str(decisionTime))
                mistrials.append((count, decisions['i'][c]))
                continue
            trialRecord = {}

            # Trial metrics
            trialRecord['Choice'] = int(decisions['winner'][c])
            trialRecord['DT'] = decisionTime/ms
            #TODO: this should be recording the gamble variable from Readout at spike time
            # but I can't be arsed right now
            trialRecord['Gamble'] = c%3 #NOTE: C SHOULD WORK EVEN WITH MISTRIALS
            #trialRecord['EstW'] = np.array(self.weights)
            
            #TODO: re-check this when needed
            if not self.MINIMAL_RECORDING:
                # Spike Trains for the Network
                for population in self.toRecord:
                    if population == 'Reward': continue
                    # For some silly reason this is a dictionary
                    tmp = self.monitors['sp'+population].spike_trains()
                    trialRecord[population] = [ (tmp[i][(tmp[i] >= startTime) & (tmp[i] < endTime)] - startTime)/ms for i in range(len(tmp.keys())) ]
                tmp = self.monitors['spReward'].spike_trains()[0]
                trialRecord['Rewarded'] = len(tmp[(tmp >= startTime) & (tmp < endTime)]) > 0

                if self.RECORD_CONTINUOUS and self.runType == 'Trials':
                    ## Match times to indices independent of time step
                    indices = np.where((self.monitors['estDAPhasic'].t >= startTime) & (self.monitors['estDAPhasic'].t < endTime))[0]
                    # Decision progress:
                    #rOut = self.monitors['ReadoutProgress'].get_states() # Nah we're not recording this, it's silly
                    trialRecord['t'] = (self.monitors['estDAPhasic'].t[indices]-startTime)/ms
                    
                    trialRecord['DAPhasic'] = self.monitors['estDAPhasic'].DA[:, indices]
                    trialRecord['DATonic'] = self.monitors['estDATonic'].DATonic[:, indices]
                    trialRecord['RPE'] = (self.monitors['estDAPhasic'].DA[:, indices] - self.monitors['estDATonic'].DATonic[:, indices]) # TODO: get lambda and alpha here
                    #trialRecord['RPE'] = rpe
                    #trialRecord['RPEDecision'] = rpe[:, :dtIdx].sum(axis=1)
                    #trialRecord['RPELearning'] = rpe[:, dtIdx:lrnIdx].sum(axis=1)
                    #trialRecord['RPERest'] = rpe[:, lrnIdx:].sum(axis=1)
                pass

            startTime = endTime
            sessionData.append(trialRecord)
            count += 1
        sessionData.append(mistrials)
        sessionData.append(self.net['TarEst'].w/nS)

        return sessionData
