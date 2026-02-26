from models.modelspecs import *

class Model:
    def makePopulation(self, name, number, parameters, parNames = neuronPars, inits = parInits, equation = eqLIF, reset = 'V = V_res', threshold = 'V > V_th'):
        self.net.add(NeuronGroup(number, equation, 'euler', threshold=threshold, reset=reset, refractory=REFRACTORY, name = name))
        for parName, parValue in list(zip(parNames, parameters))+inits: setattr(self.net[name], parName, parValue)

    def makeProjection(self, name, source, target, mod, syn, parameters, cond = '((i//CHAN_SIZE) == (j//CHAN_SIZE)) and rand() < '+str(P_CONNECT), delay = 0):
        self.net.add(Synapses(self.net[source], self.net[target], model = mod, on_pre=syn, name = name, delay=delay*ms))
        self.net[name].connect(cond)
        for parName, parValue in parameters: setattr(self.net[name], parName, parValue)

    def __init__(self, noTrials, eta = {}, # Experiment parameters
                 MINIMAL_RECORDING = False, RECORD_CONTINUOUS = True, # recording flags
                 steady = True): # Wether or not to have a steady state epoch
        self.noTrials = noTrials
        self.RECORD_CONTINUOUS = RECORD_CONTINUOUS
        self.MINIMAL_RECORDING = MINIMAL_RECORDING
        self.monitors = {}
        self.net = Network()

        BrianLogger.log_level_error()
        defaultclock.dt = 1. * ms

        # Neural populations
        ## Core Network
        self.makePopulation('Target', pSize, genPars+[0*nA, 0, extCurrents['Target']], neuronPars + ['gamble', 'targetExt'], 
                            equation='gamble : 1\ntargetExt : amp\n'+eqLIF)
        self.makePopulation('Estimation', pSize, strPars+[extCurrents['Estimation'], 0], neuronPars+['chosen'], 
                            equation='alpha : siemens\nchosen : 1\n'+eqLIF) # alpha is here to be changed by the controller after maxTrials
        # Can contain one more channel (+CHAN_SIZE) to test competition by tying mutual inhibition to the empty channel and seeing the difference
        self.makePopulation('DecisionExc', pSize, strPars+[extCurrents['DecisionExc']])  
        self.makePopulation('DecisionInh', pSize, strPars+[extCurrents['DecisionInh']]) 
        self.makePopulation('Readout', NO_CHANNELS, genPars+[0*nA], equation=eqAction, 
                            inits=[('V','V_res')]+parInits[1:]+[('last',HABITUATION_TIME*ms), ('gamble',0), ('p',P_REWARDS)])
        ## DA Subsystem: should be able to burst at 20-40Hz (i.e. from 'underlying' FR), but there's not enough neurons to inhibit it
        self.makePopulation('ctlExc', pSize, genPars+[extCurrents['ctlExc']])
        self.makePopulation('ctlInh', pSize, genPars+[extCurrents['ctlInh']])
        self.makePopulation('Dopamine', pSize, DAPars+[extCurrents['Dopamine']])
        ## Environment
        self.makePopulation('Reward', 1, genPars+[0*amp], equation='choice : 1\nrewardExt : amp\n'+eqLIF, inits=parInits+[('choice',-1)])
        ## Stopping condition
        self.makePopulation('Controller', 1, [0, noTrials], ['trialCount', 'maxTrials'], [], 'trialCount : 1 \n maxTrials : 1', 'trialCount = 0', 'trialCount >= maxTrials')
        ## Cannot use this with standalone
        #def endRun():
        #    if self.net["Controller"].trialCount[0] >= self.net["Controller"].maxTrials[0]: 
        #        self.net.stop()
        #self.net.add(NetworkOperation(endRun, 10*ms, name = "endRun"))

        # Projections
        ## Core Network 
        self.makeProjection('TarEst', 'Target', 'Estimation', learningModel, excSyn, # stochastic initialisation with a min mean of .4
                            [('w', '('+str(weights['TarEst'])+'+.08*(rand()-.5))*nS'), ('ExcRatio',EXC_RATIO), 
                             ('lbd', DAEffects['lbd']), ('eta', eta.get('TarEst', 0))])
        self.makeProjection('EstDec', 'Estimation', 'DecisionExc', excModel, excSyn, [('ExcRatio',EXC_RATIO), ('eta', eta.get('EstDec', 0))])
        self.makeProjection('DecLat', 'DecisionExc', 'DecisionInh', excModel, excSyn, [('ExcRatio',EXC_RATIO), ('eta', eta.get('DecLat', 0))])
        self.makeProjection('LatInh', 'DecisionInh', 'DecisionExc', inhModel, inhSyn, [('eta', eta.get('LatInh', 0))], 
                            '((i//CHAN_SIZE) != (j//CHAN_SIZE)) and rand() < '+str(P_CONNECT))
        self.makeProjection('LatExc', 'DecisionExc', 'DecisionExc', excModel, excSyn, [('ExcRatio',EXC_RATIO), ('eta', eta.get('LatExc', 0))],
                            '((i//CHAN_SIZE) != (j//CHAN_SIZE)) and rand() < '+str(P_CONNECT))
        self.makeProjection('DecOut', 'DecisionExc', 'Readout', excModel, excSyn, [('ExcRatio',EXC_RATIO),('eta', eta.get('DecOut', 0))], 
                            '((i//CHAN_SIZE) == j) and rand() < '+str(P_CONNECT))
        ## DA Subsystem: ideally, the more neurons, the less weights, the smoother
        self.makeProjection('excDop', 'ctlExc', 'ctlInh', inhModel, inhSyn, [('w',weights['excDop']), ('eta', eta.get('excDop', 0))])
    #    self.makeProjection('ctlBalance', 'ctlInh', 'ctlExc', inhMod, inhSyn, {[('w',1.25 *nS), ('eta', eta.get('TarEst', 0))]})
        self.makeProjection('inhDop', 'ctlInh', 'Dopamine', inhModel, inhSyn, [('w',weights['inhDop']), ('eta', eta.get('inhDop', 0))]) 
        # The proportion between rewDop and expDop sets the final weights, while the magnitude acts as learning rate
        self.makeProjection('avgValue', 'Estimation', 'ctlExc', excModel, excSyn, 
                            [('w',weights['avgValue']), ('eta', eta.get('avgValue', 0))], 'rand() < '+str(P_CONNECT))
        self.makeProjection('rewDop', 'Reward', 'ctlExc', excModel, 
                            'g_ampa += int((j//CHAN_SIZE)==choice_pre)*w \n g_nmda += int((j//CHAN_SIZE) == choice_pre)*w*ExcRatio', 
                            [('eta', eta.get('rewDop', 0))], True)
        self.makeProjection('expDop', 'Estimation', 'ctlInh', excModel, 
                            'g_ampa += chosen_pre*w \n g_nmda += chosen_pre*w*ExcRatio', 
                            [('eta', eta.get('expDop', 0))])
        self.makeProjection('dopEst', 'Dopamine', 'Estimation', '', 'DA += .0005', [])

        ## Environment
        ### Readout control
        self.makeProjection('commitOut', 'Readout', 'Readout', '', 'last = t\nV = V_res \n I_ext = -10 *nA', [], True)
        self.makeProjection('repriseOut', 'Readout', 'Readout', '', 
                            'last = t\ngamble = (gamble+1)%NO_CHANNELS\nV = V_res \n g_ampa = 0*siemens \n g_nmda = 0*siemens\n I_ext = int(j!=gamble)*IEreset', 
                            [],  True, LEARNING_TIME+INTERTRIAL_TIME)
        ### Target control
        self.makeProjection('stopTrial', 'Readout', 'Target', '', 'I_ext = 0 * nA', [], True, LEARNING_TIME)
        self.makeProjection('startTrial', 'Readout', 'Target', '', 'gamble = (gamble+1)%NO_CHANNELS\nI_ext = int((j//CHAN_SIZE)!=gamble)*targetExt', [], True, LEARNING_TIME+INTERTRIAL_TIME)
        ### Rewards and trials
        self.makeProjection('getReward', 'Readout', 'Reward', '', 'choice = i \n I_ext = int(rand()<=p_pre)*rewardExt', [], True)
        self.makeProjection('stopReward', 'Readout', 'Reward', '', 'choice = -1 \n I_ext = 0*nA', [], True, LEARNING_TIME)
        self.makeProjection('getEst', 'Readout', 'Estimation', '', 'chosen = int(i == j//CHAN_SIZE)', [], True)
        self.makeProjection('stopEst', 'Readout', 'Estimation', '', 'chosen = 0', [], True, LEARNING_TIME+INTERTRIAL_TIME)
        self.makeProjection('countTrials', 'Readout', 'Controller', '', 'trialCount += 1', [], True, LEARNING_TIME+INTERTRIAL_TIME)
        self.makeProjection('steadyPhase', 'Controller', 'Estimation', '', 'alpha = 0*nS', [], True)
        
        # Activity
        self.net.add(SpikeMonitor(self.net['Readout'], name = 'spReadout'))
        self.monitors['spReadout'] = self.net['spReadout']
        if not MINIMAL_RECORDING: 
            for layer in ['Target', 'Estimation', 'DecisionExc', 'DecisionInh','Reward', 'Dopamine', 'ctlExc', 'ctlInh']:
                self.net.add(SpikeMonitor(self.net[layer], name = 'sp'+layer))
                self.monitors['sp'+layer] = self.net['sp'+layer]
            # self.net.add(StateMonitor(self.net['Readout'], 'V', True, name = 'vReadout')
            #TODO: experiment with these monitors, as they might have to be declared after the connect has been set; 
            #TODO: at the same time, understand if the connect can be called in the other function after building the code
            self.net.add(StateMonitor(self.net['TarEst'], 'w', True, 'w'))
            self.monitors['w'] = self.net['w']
            if RECORD_CONTINUOUS:
                self.net.add(StateMonitor(self.net['Estimation'], 'DA', True, name='estDAPhasic'))
                self.monitors['estDAPhasic'] = self.net['estDAPhasic']
                self.net.add(StateMonitor(self.net['Estimation'], 'DATonic', True, name='estDATonic'))
                self.monitors['estDATonic'] = self.net['estDATonic']
        
        device.apply_run_args()

        # Running
        runTime = ms*(MAX_TRIAL_TIME+LEARNING_TIME+INTERTRIAL_TIME)*int(.7*self.noTrials*(1+int(steady)))
        self.net.run(HABITUATION_TIME*ms) # Needed for the DA levels
        self.net['Target'].I_ext = 'int((i//CHAN_SIZE)!=gamble)*targetExt'
        self.net['Readout'].I_ext = 'int(i!=gamble)*IEreset'
        self.net.run(runTime)  # Standalone needs to run full, so Learning + Steady at something around fulltime
        
        device.build(run=False)
        self.device = get_device()

    def runExperiment(self, rPath, session, pars, experiment = 0): 
        from brian2.devices import device_module
        device_module.active_device = self.device
        resDir = 'exp'+str(experiment)+'Result'+str(session)
        args = [
            (self.net['Readout'].IEreset, pars[0]),
            (self.net['Reward'].rewardExt, pars[1]),
            (self.net['EstDec'].w, pars[2]),
            (self.net['DecLat'].w, pars[3]),
            (self.net['LatInh'].w, pars[4]),
            (self.net['LatExc'].w, pars[5]),
            (self.net['DecOut'].w, pars[6]),
            (self.net['rewDop'].w, pars[7]),
            (self.net['expDop'].w, pars[8]),
            (self.net['Estimation'].alpha, pars[11])
        ]
        for pop in POPULATIONS:
            args.append((self.net[pop].tauDA, pars[9]))
            args.append((self.net[pop].tauDATonic, pars[10]))
        
        self.device.run(run_args = dict(args), results_directory = resDir)

        #prepareFolders(rPath, self.parameterSet.toDict())  # NOTE: should I save the list instead?
        # Should probably save the connectivity of the whole thing at some point
        # This is needed for weights analysis, otherwise you don't know the projections
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
            trialRecord['Choice'] = -1 if decisionTime >= MAX_TRIAL_TIME*ms else decisions['i'][c]
            trialRecord['DT'] = decisionTime/ms
            #TODO: this should be recording the gamble variable from Readout at spike time
            # but I can't be arsed right now
            trialRecord['Gamble'] = c%3 #NOTE: C SHOULD WORK EVEN WITH MISTRIALS
            
            #TODO: re-check this when needed
            if not self.MINIMAL_RECORDING:
                # Spike Trains for the Network
                for population in ['Target', 'Estimation', 'DecisionExc', 'DecisionInh', 
                                   'Dopamine', 'ctlExc', 'ctlInh']:
                    # For some silly reason this is a dictionary
                    tmp = self.monitors['sp'+population].spike_trains()
                    trialRecord[population] = [ (tmp[i][(tmp[i] >= startTime) & (tmp[i] < endTime)] - startTime)/ms for i in range(len(tmp.keys())) ]
                tmp = self.monitors['spReward'].spike_trains()[0]
                trialRecord['Rewarded'] = len(tmp[(tmp >= startTime) & (tmp < endTime)]) > 0

                ## Match times to indices independent of time step
                indices = np.where((self.monitors['w'].t >= startTime) & (self.monitors['w'].t < endTime))[0]
                trialRecord['w'] = self.monitors['w'].w[:, indices[-1]]/nS
                if self.RECORD_CONTINUOUS:
                    # Decision progress:
                    #rOut = self.monitors['ReadoutProgress'].get_states() # Nah we're not recording this, it's silly
                    trialRecord['t'] = (self.monitors['w'].t[indices]-startTime)/ms
                    
                    trialRecord['DAPhasic'] = self.monitors['estDAPhasic'].DA[:, indices]
                    trialRecord['DATonic'] = self.monitors['estDATonic'].DATonic[:, indices]
                    # This stuff shouldn't be calculated here, it should be done afterwards, in pre-analysis
                    #rpe = ((monitors['fDA'].fDA[:, fullIndices] - monitors['sDA'].sDA[:, fullIndices])**PARAMETERSET['model'][1]['lbd']) #*PARAMETERSET#['model'][1]['alpha']
                    #trialRecord['RPE'] = rpe
                    #trialRecord['RPEDecision'] = rpe[:, :dtIdx].sum(axis=1)
                    #trialRecord['RPELearning'] = rpe[:, dtIdx:lrnIdx].sum(axis=1)
                    #trialRecord['RPERest'] = rpe[:, lrnIdx:].sum(axis=1)
                pass

            startTime = endTime
            sessionData.append(trialRecord)
            count += 1
        sessionData.append(mistrials)

        return sessionData
