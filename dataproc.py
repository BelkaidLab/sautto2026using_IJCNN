from models.models import *

import os
import json
import multiprocessing as mp
import seaborn as sns
import pandas as pd
import scipy as sp 
import statsmodels.formula.api as fml
import statsmodels.api as sm 


## Creates folder structure for an experiment run
def prepareFolders(path, PARAMETERSET = None):
    os.makedirs(path, exist_ok = True)
    os.makedirs(path+'Recordings/', exist_ok = True)
    os.makedirs(path+'Plots/', exist_ok = True)
    os.makedirs(path+'Connectivity/', exist_ok = True)     
    if PARAMETERSET:
        with open(path+'parameterset.json', 'w') as f:
            f.write(str(PARAMETERSET))

## Loads raw data of a given experiment
def loadExperimentData(rPath, sessions, noTrials, prefix = '', training = True, steady = True):
    rawData = []
    mistrials = []
    EstW = []
    for session in range(sessions):
        with open(rPath+'Recordings/'+prefix+'Session'+str(session), 'rb') as f:
            data = pk.load(f)
            data, mst, weights = data[:-2], data[-2], data[-1]
            if training and steady:
                rawData.append(data)
            elif training:
                rawData.append(data[:noTrials])
            elif steady:
                rawData.append(data[noTrials:noTrials*2])
            mistrials.append(mst)
            EstW.append(weights)

    return rawData, mistrials, EstW

# PLOTS
## Plots a single trial as a raster plot
def plot_raster(trial, T, MODULATION, sNo = -1, tNo = -1, rPath = None):
    fig, ax = plt.subplots(3, 2, figsize = (12, 7.5), tight_layout=True)
    ax = ax.flatten()
    timeMax = trial['DT']+T['learn']/ms

    for k, population in enumerate(['Target', 'Estimation', 'Lateral', 'Decision', 'Dopamine']):
        spikes = [trial[population][i] for i in range(pSize)]
        spikes = [s[s <= timeMax] for s in spikes]
        rate = np.array([ len(np.concatenate(spikes[i*CHAN_SIZE:(i+1)*CHAN_SIZE])) for i in range(NO_CHANNELS)])*1000/(timeMax*CHAN_SIZE)

        for i in range(pSize):
            ax[k].plot(spikes[i], [i]*len(spikes[i]), '|', color = 'C'+str(i//CHAN_SIZE))
        
        ax[k].vlines(trial['DT'], trial['Choice']*CHAN_SIZE, (trial['Choice']+1)*CHAN_SIZE, color='red')

        #ax[k].set_ylabel('Neuron Id')
        #ax[k].set_xlabel('Time (ms)')
        ax[k].set_title(population)
        ax[k].set_xlim(0,timeMax)
        ax[k].set_ylim(-.1, pSize+.1)
        tmp = ax[k].twinx()
        tmp.set_ylim(-.1, pSize+.1)
        tmp.set_yticks([(i+.5)*CHAN_SIZE for i in range(NO_CHANNELS)])
        tmp.set_yticklabels([str(round(r,2))+'Hz' for r in rate])
    if MODULATION:
        # TODO: see if signal actually goes here
        pass
    
    ax[-1].set_ylabel('')
    ax[-1].set_xlabel('')
    ax[-1].set_yticks([])
    ax[-1].set_xticks([])
    
    if rPath != None: 
        fig.savefig(rPath+'Plots/'+'Raster S'+str(sNo)+'T'+str(tNo)+'.svg')
        plt.close()
    else: return fig 
## Plots a single trial as instantaneous firing rate
def plot_rates(trial, T, MODULATION, sNo = -1, tNo = -1, rPath = None, binsize = 20, windowsize = 100):
    fig, ax = plt.subplots(3, 2, figsize = (12, 12), tight_layout=True)
    ax = ax.flatten()
    maxFrequency = 40
    timeMax = int(trial['DT']+T['learn']/ms)

    for k, population in enumerate(['Target', 'Estimation', 'Lateral', 'Decision', 'Dopamine']):
        ax[k].set_title(population)
        ax[k].set_ylim(0, NO_CHANNELS)
        ax[k].set_yticks(np.arange(0, NO_CHANNELS+.0001, 1./4))
        ax[k].set_yticklabels(['0']+[str(int(maxFrequency*.25)),str(int(maxFrequency*.50)),str(int(maxFrequency*.75)),str(int(maxFrequency))]*NO_CHANNELS)

        ax[k].vlines(trial['DT'], trial['Choice'], (trial['Choice']+1), color='red')

        spikes = [trial[population][i] for i in range(pSize)]
        popSpikes = [ np.sort(np.concatenate(spikes[i*CHAN_SIZE:(i+1)*CHAN_SIZE])) for i in range(NO_CHANNELS)]
        
        for i in range(NO_CHANNELS):
            spCounts, t = np.histogram(popSpikes[i],range(0, timeMax+1, binsize))
            rate = sp.signal.savgol_filter(spCounts/(CHAN_SIZE*1.), int(windowsize/binsize), 1)
            
            visualize = (rate*1000/(maxFrequency*binsize) + i)
            ax[k].plot(t[:-1], visualize, color = 'C'+str(i))
            if i != 0:
                ax[k].hlines(i, 0, timeMax, color='black')
            
    rax = ax[-1] 
    rax.set_ylim(0, NO_CHANNELS)
    vRange = (-genPars['V_res']+genPars['V_th'])/mV
    time = trial['t'][trial['t'] <= trial['DT']]
    for i in range(NO_CHANNELS):
        visualize = trial['Evidence'][:len(time), i]
        visualize = (visualize-genPars['V_res']/mV)/((genPars['V_th']-genPars['V_res'])/mV) + i
        #visualize[0] = i
        rax.plot(time, visualize, color = 'C'+str(i))
        if i != 0:
            rax.hlines(i, 0, timeMax, color='black')
    # These don't show up and it's fine I don't care
    rax.set_yticks(np.arange(0, NO_CHANNELS+.0001, 1./3))
    rax.set_yticklabels(['-70']+[str(-70+int(vRange*.34)),str(-70+int(vRange*.67)),str(-70+int(vRange))]*NO_CHANNELS)

    ax[-1].set_ylabel('')
    ax[-1].set_xlabel('')
    ax[-1].set_yticks([])
    ax[-1].set_xticks([])

    if rPath != None: 
        fig.savefig(rPath+'Plots/'+'Rates S'+str(sNo)+'T'+str(tNo)+'.svg')
        plt.close()
    else: return fig 
## Plots the layers related to the Dopamine control mechanism
def plot_dopamine(trial, T, MODULATION, sNo = -1, tNo = -1, rPath = None, binsize = 20, windowsize = 100):
    raster, ax = plt.subplots(3, 2, figsize = (12, 5), tight_layout=True)
    rax, ax = ax[0][0], ax.flatten()[1:]
    timeMax = trial['DT']+T['learn']/ms

    for k, population in enumerate(['Estimation', 'ctlExc', 'ctlInh', 'Dopamine']):
        spikes = [trial[population][i] for i in range(pSize)]
        spikes = [s[s <= timeMax] for s in spikes]
        rate = np.array([ len(np.concatenate(spikes[i*CHAN_SIZE:(i+1)*CHAN_SIZE])) for i in range(NO_CHANNELS)])*1000/(timeMax*CHAN_SIZE)

        for i in range(pSize):
            ax[k].plot(spikes[i], [i]*len(spikes[i]), '|', color = 'C'+str(i//CHAN_SIZE), markersize = 3.)
        
        ax[k].vlines(trial['DT'], trial['Choice']*CHAN_SIZE, (trial['Choice']+1)*CHAN_SIZE, color='red')

        #ax[k].set_ylabel('Neuron Id')
        #ax[k].set_xlabel('Time (ms)')
        ax[k].set_title(population)
        ax[k].set_xlim(0,timeMax)
        ax[k].set_ylim(-.1, pSize+.1)
        tmp = ax[k].twinx()
        tmp.set_ylim(-.1, pSize+.1)
        tmp.set_yticks([(i+.5)*CHAN_SIZE for i in range(NO_CHANNELS)])
        tmp.set_yticklabels([str(round(r,2))+'Hz' for r in rate])
    
    ax[-1].set_title('RPE for Chosen')
    ax[-1].plot(trial['RPE'][CHAN_SIZE*trial['Choice']:CHAN_SIZE*(trial['Choice']+1)].T, color = 'C0', alpha = .05)
    ax[-1].plot(trial['RPE'][CHAN_SIZE*trial['Choice']:CHAN_SIZE*(trial['Choice']+1)].mean(axis=0), color = 'C0', alpha = 1)
    ax[-1].vlines([trial['DT'], trial['DT']+T['learn']/ms], -.03, .03, color='red', alpha=.2, linewidth = 1)
    ax[-1].hlines(0, 0, trial['DT']+T['learn']/ms+T['rest']/ms  , color='red', linestyles = 'dashed', linewidth=1, alpha = .2)


    spikes = np.array(trial['Reward'][0])
    rate = len(spikes)*1000/(T['learn']/ms)
    rax.plot(spikes, [0]*len(spikes), '|', color = 'limegreen')
    rax.vlines(trial['DT'], -.5, .5, color='red')

    rax.set_title('Reward')
    rax.set_xlim(0,timeMax)
    rax.set_ylim(-.6, .6)
    tmp = rax.twinx()
    tmp.set_ylim(-.6, .6)
    tmp.set_yticks([0])
    tmp.set_yticklabels([str(round(rate, 2))+'Hz'])

    if rPath != None: 
        raster.savefig(rPath+'Plots/'+'DARaster S'+str(sNo)+'T'+str(tNo)+'.svg')
        plt.close()

    '''
    rates, ax = plt.subplots(2, 2, figsize = (12, 7), tight_layout=True)
    ax = ax.flatten()
    timeMax = int(trial['DT']+T['learn']/ms)

    for k, population in enumerate(['Estimation', 'ctlExc', 'ctlInh', 'Dopamine']):
        ax[k].set_title(population)
        ax[k].set_ylim(0, noChans)
        ax[k].set_yticks(np.arange(0, noChans+.0001, 1./4))

        maxFrequency = 40
        if population == 'Dopamine': maxFrequency = 20
        ax[k].set_yticklabels(['0']+[str(int(maxFrequency*.25)),str(int(maxFrequency*.50)),str(int(maxFrequency*.75)),str(int(maxFrequency))]*noChans)

        ax[k].vlines(trial['DT'], trial['Choice'], (trial['Choice']+1), color='red')

        spikes = [trial[population][i] for i in range(noChans*chanSize)]
        popSpikes = [ np.sort(np.concatenate(spikes[i*chanSize:(i+1)*chanSize])) for i in range(noChans)]
        
        for i in range(noChans):
            spCounts, t = np.histogram(popSpikes[i],range(0, timeMax+1, binsize))
            rate = sp.signal.savgol_filter(spCounts/(chanSize*1.), int(windowsize/binsize), 1)
            
            visualize = (rate*1000/(maxFrequency*binsize) + i)
            ax[k].plot(t[:-1], visualize, color = 'C'+str(i))
            if i != 0:
                ax[k].hlines(i, 0, timeMax, color='black')

    if rPath != None: 
        rates.savefig(rPath+'Plots/'+'DARates S'+str(sNo)+'T'+str(tNo)+'.svg')
        plt.close() '''

    if rPath == None:
        return raster #, rates

## Plots the distrbution of decision times
def dtDist(df, rPath, prefix, maxTime, iax = None):
    if iax is not None: ax = iax
    else: f, ax = plt.subplots()

#    outliers = len(df[df['Choice'] == -1]['DT'].values)
    dts = df[df['Choice'] != -1]['DT'].values # Failed trials are not taken into account
#    outliers = outliers/(len(dts)+outliers)
#    outliers = ((outliers*10000)//1)/100.
    n, b, _ = ax.hist(dts, bins=50)
    n = np.array(n)

    ax.axvline(dts.min(), 0, n.max(), color = 'C1')
    ax.text(dts.min(), np.max(n)+1, str(dts.min().round(2))+'ms')
    ax.axvline(dts.max(), 0, n.max(), color = 'C1')
    ax.text(dts.max()/1.2, n.max()+1, str(dts.max().round(2))+'ms')

    ax.axvline(dts.mean(), 0, n.max(), color = 'C2')
    ax.text(dts.mean()*1.1, n.max()*.9, '~'+str(dts.mean().round(2))+'ms')
    ax.axvline(b[np.where(n == n.max())[0][0]], 0, n.max(), color = 'C3')
    ax.text(b[np.where(n == n.max())[0][0]]*1.1, n.max()*.8, str(b[np.where(n == n.max())[0][0]].round(2))+'ms')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Number of Trials')
    ax.set_yticks(np.array(ax.get_yticks()).astype(int))
    ax.set_xlim(0, int(maxTime/ms))

    if rPath != None: 
        ax.figure.savefig(rPath+'Plots/'+prefix+'Decision Times.svg')
        plt.close()
## Choice percentage plot (times chosen / times available)
def choicePercentage(df, rPath, prefix, iax = None):
    if iax is not None: ax = iax
    else: f, ax = plt.subplots(tight_layout=True)

    # Calculating how many times the choice i was available per session; i.e. how many times Gamble != i per session
    gambCount = df.groupby('S')['G'].value_counts()
    gambCount = pd.DataFrame(gambCount).reset_index()
    available = []
    for i in df['Choice'].unique():
        tmp = gambCount[gambCount['G'] != i].groupby('S').sum()[['count']].reset_index()
        tmp['Choice'] = i
        available.append(tmp.rename({'count':'Available'}, axis=1))
    
    available = pd.concat(available).set_index(['S', 'Choice'])
    # Calculating the choice percentage of each option
    choices = df.groupby('S')['Choice'].value_counts().reset_index().rename({'count':'Chosen'}, axis=1)
    choices = choices[choices['Choice'] != -1] # Outliers
    choices = choices.set_index(['S', 'Choice']).join(available)
    choices['Chosen %'] = 100.*choices['Chosen']/choices['Available']
    choices = choices.reset_index()
    choices['Reward Probability'] = [ P_REWARDS[choice] for choice in choices['Choice'] ]
    ratio = choices.groupby('Choice').agg({'Chosen %':['mean', 'std']}).reset_index()
    ratio.columns = ratio.columns.to_flat_index()
    ratio.columns = ['Choice', 'Chosen %', 'std']
    ratio.reindex(columns=ratio.columns)
    ratio['Reward Probability'] = P_REWARDS

    reg = sp.stats.linregress(choices['Reward Probability'], choices['Chosen %'])
    ax.text(.05, .75,'p = '+'%.2e' % (reg.pvalue)+'\nR2 = '+str(round(reg.rvalue**2, 2))+'\nslope = '+str(round(reg.slope, 2)), transform = ax.transAxes)
    #sns.lineplot(data = choices, x = 'Reward Probability', y = 'Chosen %', marker = 'o', ax = ax)
    sns.regplot(data = choices, x = 'Reward Probability', y = 'Chosen %', ax = ax)
    #ratio.plot('Reward Probability','Chosen %', yerr='std', legend=False, ax = ax, capsize=6, ylim=(0,100), xlim=(0,1.1), xticks=ratio['Reward Probability'])
    for i, p in enumerate(ratio['Chosen %']):
        ax.text(ratio['Reward Probability'][i], ratio['Chosen %'][i], np.round(p, 2).astype(str)+'%')
    ax.set_ylim(0, 100)
    ax.set_xlim(0,1.1)
    ax.set_ylabel('Times chosen/Choice available')

    if rPath != None: 
        ax.figure.savefig(rPath+'Plots/'+prefix+'Choice Percentage.svg')
        plt.close()
    else: return ax

## Plot the distribution of Dopmanine firing rates
def plotFR(net, exp, pop, rPath, prefix, outThs = (0, 40)):
    fr = net[net['Population'] == pop].set_index(['S', 'T']).join(exp[['S', 'T', 'Choice','G', 'R']].set_index(['S', 'T']))
    fr = fr[fr['Channel'] != fr['G']] # Channel = Gamble is only affected in tonic firing
    fr['Chosen'] = fr['Channel'] == fr['Choice']
    fr = fr[['R', 'Decision', 'Learning', 'Chosen']].melt(id_vars=['R', 'Chosen']).rename(columns={'value':'Firing Rate (Hz)', 'variable':'Time'})
    # Outliers:
    fr = fr[(fr['Firing Rate (Hz)'] > outThs[0]) & (fr['Firing Rate (Hz)'] < outThs[1])]
    fr = fr[(fr['Time'] == 'Learning') | ((fr['Time'] == 'Decision') & (fr['Firing Rate (Hz)'] > 0))]
    fr = fr.rename(columns={'R':'Reward'})

    dafr = sns.displot(data = fr, x = 'Firing Rate (Hz)', hue='Time', col='Chosen', row = 'Reward', rug=True, common_bins = False)
    specs = fr.groupby(['Time', 'Chosen', 'Reward'])['Firing Rate (Hz)'].agg(['mean', 'median']).reset_index()

    for ax in dafr.axes.flatten():
        pars = [ p.split(' = ')[1] for p in ax.get_title().split(' | ') ]
        medi = specs[(specs['Time'] == 'Decision') & (specs['Reward'] == (pars[0]==True)) & (specs['Chosen'] == (pars[1] == 'True'))]['median']
        ax.axvline(x = medi.values[0], c = 'blue')
        medi = specs[(specs['Time'] == 'Learning') & (specs['Reward'] == (pars[0]==True)) & (specs['Chosen'] == (pars[1] == 'True'))]['median']
        ax.axvline(x = medi.values[0], c = 'orange')

    if rPath != None: 
        dafr.figure.savefig(rPath+'Plots/'+prefix+'DAFR.svg')
        plt.close()

## Weights-Value plot
def hasLearned(weights, rPath):
    lastWeights = weights[weights['Trial'] == weights['Trial'].max()].drop('Trial', axis = 1)
    fig, ax = plt.subplots(figsize = (6, 3))
    reg = sp.stats.linregress(lastWeights['Reward Probability'], lastWeights['wEnd'])
    #fg = sns.lineplot(data = lastWeights, x = 'Reward Probability', y = 'wEnd', marker = 'o', ax = ax)
    plt.title('Learned Weights')
    ax.text(.05, .75,'p = '+'%.2e' % (reg.pvalue)+'\nR2 = '+str(round(reg.rvalue**2, 2))+'\nslope = '+str(round(reg.slope, 2)), transform = ax.transAxes)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, .15)
    ax.set_xticks([.25, .5, 1.])
    sns.regplot(data = lastWeights, x = 'Reward Probability', y = 'wEnd', ax = ax)
    plt.savefig(rPath+'Plots/learnedWeights.svg', bbox_inches = 'tight')
    plt.close()

# DATA PROCESSING
## Extracting experiment statistics
def processExperiment(raw):
    sNo, tNo, decT, gamble, choice, reward = [], [], [], [], [], []
    for s, session in enumerate(raw):
        for t, trial in enumerate(session):
            reward.append(trial['Rewarded'])
            sNo.append(s), tNo.append(t), gamble.append(trial['Gamble']),
            decT.append(trial['DT']), choice.append(trial['Choice'])
    return pd.DataFrame({'S':sNo, 'T':tNo, 'DT':decT, 'G':gamble, 'Choice':choice, 'R':reward})

## Extracting network statistics (i.e. FR) for the experiment
def processNetwork(raw, learn, rest):
    sNo, tNo, pop, ch, frDec, frLrn, frRst, frTrl, frPst, frAll, pR = [], [], [], [], [], [], [], [], [], [], []
    for s, session in enumerate(raw):
        for t, trial in enumerate(session):
            dt = trial['DT']
            for p in ['Estimation', 'Decision', 'Dopamine']:
                try:
                    spikesDecision = [ neuron[neuron < dt] for neuron in trial[p]]
                    spikesLearning = [ neuron[(neuron >= dt) & (neuron < (dt+learn))] for neuron in trial[p]]
                    spikesRest = [ neuron[(neuron >= (dt+learn))] for neuron in trial[p]]
                except:
                    spikesDecision = [[]]*CHAN_SIZE
                    spikesLearning = [[]]*CHAN_SIZE
                    spikesRest = [[]]*CHAN_SIZE
                
                for i in range(NO_CHANNELS):
                    sNo.append(s), tNo.append(t), pop.append(p), ch.append(i), pR.append(P_REWARDS[i])
                    # Firing Rates calculations
                    chDec = len([sp for n in spikesDecision[i*CHAN_SIZE:(i+1)*CHAN_SIZE] for sp in n])/CHAN_SIZE
                    chLrn = len([sp for n in spikesLearning[i*CHAN_SIZE:(i+1)*CHAN_SIZE] for sp in n])/CHAN_SIZE
                    chRst = len([sp for n in spikesRest[i*CHAN_SIZE:(i+1)*CHAN_SIZE] for sp in n])/CHAN_SIZE
                    frDec.append(chDec*1000/dt)
                    frLrn.append(chLrn*1000/learn)
                    frRst.append(chRst*1000/rest)
                    frTrl.append((chDec + chLrn)*1000/(dt+learn))
                    frPst.append((chLrn + chRst)*1000/(learn+rest))
                    frAll.append((chDec + chLrn + chRst)*1000/(dt+learn+rest))

    return pd.DataFrame({'S':sNo, 'T':tNo, 'Population':pop, 'Channel':ch, 'Reward Probability':pR, 
                       'Decision':frDec, 'Learning':frLrn, 'Rest':frRst, 'Trial':frTrl, 'Nonlearning':frPst, 'Overall':frAll}) 

## Extracting information about learning. Only considers 'mean' weight
def processChanges(raw, connectivity):
    sNo, tNo, nId, w, pR = [], [], [], [], []
    # TODO: needs constant weights recording
#    for s, session in enumerate(raw):
    for s in range(len(raw)):
        trgs = connectivity[s]['trg']//CHAN_SIZE
#        for t, trial in enumerate(session):
        for t in range(1):
            for i in range(3):
                sNo.append(s)
                tNo.append(t)
                nId.append(i)
                pR.append(P_REWARDS[i])
#                print(trial['EstW'])
#                w.append(trial['EstW'][np.where(trgs == i)].mean())
                w.append(raw[s][np.where(trgs == i)].mean())


    return pd.DataFrame({'Session':sNo, 'Trial':tNo, 'nId':nId, 'wEnd':w, 'Reward Probability':pR})
## Learning process checks
def plotChanges(weights, net, exp, rPath):
    line = sns.relplot(data = weights, kind = 'line', x = 'Trial', y = 'wEnd', hue = 'Reward Probability', height= 2.5, aspect=3)
    line.ax.set_ylabel('Average weight after trial')
    line.ax.set_ylim(0,.2)
    line.set(title = 'Average weights over trials')
    plt.savefig(rPath+'Plots/TarEst over trials.svg', bbox_inches='tight')
    plt.close()

    tmp = net[net['Population'] == 'Estimation'].set_index(['S','T']).join(exp[['S', 'T', 'G']].set_index(['S', 'T'])).reset_index()
    tmp = tmp[tmp['Channel'] != tmp['G']]
    line = sns.relplot(data = tmp, kind = 'line', x = 'T', y = 'Trial', hue = 'Reward Probability', height = 2.5, aspect = 3)
    line.ax.set_ylabel('Average FR (Hz)')
    line.ax.set_ylim(0, 40)
    line.set(title = 'Trial Average - Estimation Firing Rates')
    plt.savefig(rPath+'Plots/EstFR over trials.svg', bbox_inches='tight')
    plt.close()

    tmp = net[net['Population'] == 'Decision'].set_index(['S','T']).join(exp[['S', 'T', 'G']].set_index(['S', 'T'])).reset_index()
    tmp = tmp[tmp['Channel'] != tmp['G']]
    line = sns.relplot(data = tmp, kind = 'line', x = 'T', y = 'Trial', hue = 'Reward Probability', height = 2.5, aspect = 3)
    line.ax.set_ylabel('Average FR (Hz)')
    line.ax.set_ylim(20, 50)
    line.set(title = 'Trial Average - Decision Firing Rates')
    plt.savefig(rPath+'Plots/DecFR over trials.svg', bbox_inches='tight')
    plt.close()

def estDaEffect(net, rPath, prefix, limit = 30):
    tmp = pd.DataFrame(net[(net['Population'] == 'Estimation') | (net['Population'] == 'Dopamine')]).drop(columns=['Rest','Nonlearning', 'Overall', 'Learning'])
    tmp = tmp[tmp['T'] < limit]
    est = pd.DataFrame(tmp[tmp['Population'] == 'Estimation']).drop(columns=['Population', 'Channel', 'Reward Probability'])
    est = est.groupby(['S','T']).agg('mean').reset_index()
    est['G'] = est['T']%3

    dop = pd.DataFrame(tmp[tmp['Population'] == 'Dopamine']).drop(columns=['Population']).melt(id_vars=['S', 'T', 'Channel'], value_vars=['Decision', 'Trial'], var_name='Time', value_name='Average FR (Hz)')
    # Do average estimation levels change with the gamble (i.e. which targets are active)?
    estLevels = est.melt(id_vars=['S','T','G'], value_vars=['Decision','Trial'], var_name='Time', value_name='Average FR (Hz)')
    print(sp.stats.f_oneway(estLevels[estLevels['G'] == 0]['Average FR (Hz)'], estLevels[estLevels['G'] == 1]['Average FR (Hz)'], estLevels[estLevels['G'] == 2]['Average FR (Hz)']))
    print(sp.stats.tukey_hsd(estLevels[estLevels['G'] == 0]['Average FR (Hz)'], estLevels[estLevels['G'] == 1]['Average FR (Hz)'], estLevels[estLevels['G'] == 2]['Average FR (Hz)']))
    sns.boxplot(estLevels, y = 'Average FR (Hz)', hue = 'Time', x ='G')
    plt.title('Overall Expected Value (FR) per Gamble')
    plt.savefig(rPath+'Plots/'+prefix+'OEVonG.svg')

    tmp = dop.merge(estLevels, on=['S', 'T', 'Time'], suffixes=[' Dopamine', ' Estimation'])
    dec = tmp[tmp['Time'] == 'Decision']
    tri = tmp[tmp['Time'] == 'Trial']

    fg = sns.lmplot(data=tmp, x = 'Average FR (Hz) Estimation', y = 'Average FR (Hz) Dopamine', markers='.', col = 'Time')

    reg = sp.stats.linregress(dec['Average FR (Hz) Estimation'], dec['Average FR (Hz) Dopamine'])
    fg.axes[0][0].text(.05, .75,'p = '+'%.2e' % (reg.pvalue)+'\nR2 = '+str(round(reg.rvalue**2, 2))+'\nslope = '+str(round(reg.slope, 2)), transform = fg.axes[0][0].transAxes)
    reg = sp.stats.linregress(tri['Average FR (Hz) Estimation'], tri['Average FR (Hz) Dopamine'])
    fg.axes[0][1].text(.05, .75,'p = '+'%.2e' % (reg.pvalue)+'\nR2 = '+str(round(reg.rvalue**2, 2))+'\nslope = '+str(round(reg.slope, 2)), transform = fg.axes[0][1].transAxes)
    plt.title('Overall Expected Value Influences DA')
    plt.savefig(rPath+'Plots/'+prefix+'OEVonDA.svg')

    dec = dec.rename(columns={'Average FR (Hz) Estimation':'Estimation', 'Average FR (Hz) Dopamine':'Dopamine'})
    mod = fml.ols('Estimation ~ Channel * Dopamine', data = dec).fit()
    with open(rPath+'OEVtoDAresults.txt', 'w') as f:
        print(str(sm.stats.anova_lm(mod, type = 2)) + '\n' + str(mod.summary()), file=f)

# ANALYSIS PIPELINES
def standardAnalysis(rPath, nS, nT, steady = True, learning = True, MINIMAL_RECORDING = False, RECORD_CONTINUOUS = True, TMax = MAX_TRIAL_TIME):
    tarEstConnectivity = []
    for s in range(nS):
        with open(rPath+'Connectivity/Session'+str(s), 'rb') as f:
            tarEstConnectivity.append(pk.load(f))

    if learning: # Learning Data
        learningData, lrnMst, EstW = loadExperimentData(rPath, nS, 'Plastic')
        lrnExp = processExperiment(learningData)
        lrnExp.to_csv(rPath+'PlasticExperimentData.csv', index=False)
        lrnNwk = processNetwork(learningData, LEARNING_TIME, INTERTRIAL_TIME)
        lrnNwk.to_csv(rPath+'PlasticNetworkStats.csv', index=False)
    
        dtDist(lrnExp, rPath, 'Plastic', TMax)
        choicePercentage(lrnExp, rPath, 'Plastic')
        plotFR(lrnNwk, lrnExp, 'Dopamine', rPath, 'Plastic')

        try:
            weights = processChanges(EstW, tarEstConnectivity)
            #weights = processChanges(learningData, tarEstConnectivity)
            weights.to_csv(rPath+'WeightsStats.csv', index=False)
            # TODO: requires continuous weights
            #plotChanges(weights, lrnNwk, lrnExp, rPath)
            hasLearned(weights, rPath)
        except Exception as ex:
            print(tb.format_exc())
            print("Learning informaion was not available. ")

    
    if steady: # Steady Data
        steadyData, stdMst, _ = loadExperimentData(rPath, nS, nT, '', False, True)
        stdExp = processExperiment(steadyData)
        stdExp.to_csv(rPath+'SteadyExperimentData.csv', index=False)
        stdNwk = processNetwork(steadyData, LEARNING_TIME, INTERTRIAL_TIME)
        stdNwk.to_csv(rPath+'SteadyNetworkStats.csv', index=False)

        dtDist(stdExp, rPath, 'Steady', TMax)
        choicePercentage(stdExp, rPath, 'Steady')
        plotFR(stdNwk, stdExp, 'Dopamine', rPath, 'Steady')
        estDaEffect(stdNwk, rPath, 'Steady')