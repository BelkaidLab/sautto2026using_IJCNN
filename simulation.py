from dataproc import *
from random import uniform

os.environ["MAKEFLAGS"] = "-j20"

def runExperiment(runType, model, prefix, PARSET, fixedCurrents, fixedWeights):
    set_device('cpp_standalone', build_on_run=False)
    defaultclock.dt = 1.*ms

    if runType == 'Trials': savePath = './LearningTask/'+prefix+'_s'+str(SESSIONS)+'t'+str(TRIALS)+'/'
    else: savePath = './DAFiring/'+prefix+'/'

    print("Generating simulation environment at "+time.strftime("%d/%m/%Y - %H:%M:%S",time.localtime()))
#    
    constArguments = (TRIALS, {}, False, False, False, fixedCurrents, fixedWeights, runType)
    if model == 'FULLDIR':
        sim = DACFullDir(*constArguments)
    elif model == 'FULLINH':
        sim = DACFullInh(*constArguments)
    else:
        sim = Model(TRIALS, MINIMAL_RECORDING=False)  # NOTE: This is most likely broken now

    print("Simulation starting at "+time.strftime("%d/%m/%Y - %H:%M:%S",time.localtime()))
    prepareFolders(savePath, [PARSET, fixedCurrents, fixedWeights])
    with mp.get_context('spawn').Pool(10) as p:
        args = [ (savePath, s, PARSET) for s in range(SESSIONS)]
        p.starmap(sim.runExperiment, args)
    print("Simulation ended at "+time.strftime("%d/%m/%Y - %H:%M:%S",time.localtime()))
    if runType == 'Trials': standardAnalysis(savePath, SESSIONS, TRIALS, steady=False, MINIMAL_RECORDING = True, TMax = FAKE_TRIAL_TIME+100)
    

def generateParameters(rPath = './DAFiring/', noVariations = 100, varRange = .1, models = ['FULLINH', 'FULLDIR']):
    # Relevant parameters are rewDop expDop tauDA tauDATonic avgValue Dopamine
    for model in models:
        os.makedirs(rPath+model+'/', exist_ok = True)
        for i in range(noVariations+1):
            parVariation = {}
            if model == 'FULLDIR':
                parVariation['parset'] = list(FULLDIR)
                parVariation['fixedCurrents'] = {'Target':.33*nA, 'Estimation': .28 * nA, 'ctlExc': .24*nA, 'ctlInh': .31*nA, 'Dopamine': .412*nA}
                parVariation['fixedWeights'] = {'TarEst': .04, 'avgValue':.4*nS }
            elif model == 'FULLINH':
                parVariation['parset'] = list(FULLINH)
                parVariation['fixedCurrents'] = dict(extCurrents)
                parVariation['fixedWeights'] = dict(weights)
            if i!=noVariations:
                parVariation['parset'][2] *= 1 + varRange*uniform(-1, 1)
                parVariation['parset'][3] *= 1 + varRange*uniform(-1, 1)
                parVariation['parset'][4] *= 1 + varRange*uniform(-1, 1)
                parVariation['parset'][5] *= 1 + varRange*uniform(-1, 1)
                parVariation['fixedCurrents']['Dopamine'] *= 1 + varRange*uniform(-1, 1)
                parVariation['fixedWeights']['avgValue'] *= 1 + varRange*uniform(-1, 1)
            else:
                print(parVariation)
            with open(rPath+model+'/ParVar'+str(i), 'wb') as f:
                pk.dump(parVariation, f)



# Originally Readout was .21 for both
#         'Readout', 'Reward', 'EstDec', 'DecLat', 'LatInh', 'LatExc', 'DecOut', 'rewDop', 'expDop', 'tauDA', 'tauDATonic', 'alpha' 
#PARSET1 = [ .1*nA,     .9*nA,  .1*nS,      .1*nS,      .4*nS,  .1*nS,     .07*nS, 6.*nS,   2.*nS,      500*ms, 3000*ms, .005*nS ]
#PARSET2 = [ .1*nA,     .7*nA,  .1*nS,      .1*nS,      .4*nS,  .1*nS,     .06*nS, 2.*nS,   .7*nS,      100*ms,  500*ms, .01*nS ]
PARSET = []

avgVal = {'Hi':.1, 'Mi':.07, 'Lo':.04}
#        'Readout', 'Reward', 'rewDop', 'expDop', 'tauDA', 'tauDATonic', 'alpha'
#FULLINH = [ .1*nA,     .7*nA,    2.*nS,   .7*nS,      100*ms,  500*ms, .01*nS ]
FULLINH = [ .1*nA,     .8*nA,     4.*nS,   1.*nS,      100*ms,  500*ms, .01*nS ]
FULLDIR = [ .1*nA,     .8*nA,     4.*nS,   2.5*nS,      100*ms,  500*ms, .01*nS ] 
parsets = {'FULLINH':FULLINH, 'FULLDIR':FULLDIR}

SESSIONS = 30
TRIALS = 90
if __name__ == "__main__":
    mode = 0
    if mode == 0: # Base parset  
        varNo = 101
        runType = 'Trials'
        model = 'FULLINH'
        envValue = .4

        if model == 'FULLDIR':
            PARSET = FULLDIR
            fixedCurrents = {'Target':.33*nA, 'Estimation': .28 * nA, 'ctlExc': .24*nA, 'ctlInh': .31*nA, 'Dopamine': .412*nA}
            fixedWeights = {'TarEst': .1, 'avgValue':.4*nS }
        elif model == 'FULLINH':
            PARSET = FULLINH
            #{'Target':.33*nA, 'Estimation': .28 * nA, 'DecisionExc':.33 * nA, 'DecisionInh':.295 * nA, 'ctlExc': .24*nA,  'ctlInh': .31*nA, 'Dopamine': .47*nA}
            fixedCurrents = extCurrents
            fixedWeights = weights
        if runType != 'Trials': fixedWeights['TarEst'] = avgVal[envValue]
        prefix = './' #model+runType+envValue+str(varNo)
        runExperiment(runType, model, prefix, PARSET, fixedCurrents, fixedWeights)

    if mode == 1: # Random Search
        # TODO: test if with different classes it's possible to run these one after another
        runType = sys.argv[2]
        model = sys.argv[3]
        envValue = sys.argv[4]
        varNo = sys.argv[1]
        with open('./DAFiring/'+model+'/ParVar'+str(varNo), 'rb') as f:
            parVariation = pk.load(f)
        parVariation['fixedWeights']['TarEst'] = avgVal[envValue]
        prefix = model+runType+envValue+str(varNo)
        if runType == 'Trials': prefix +="_s"+str(SESSIONS)+"t"+str(TRIALS)
        runExperiment(runType, model, prefix, parVariation['parset'], parVariation['fixedCurrents'], parVariation['fixedWeights'])


    if mode == 2: # Random Search parset generation 
        generateParameters()
        quit()
