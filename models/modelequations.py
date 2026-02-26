## Equations
lifPars = '''
E_L : volt
R : ohm
tau : second
V_th : volt
V_res : volt
ExcRatio : 1
tau_ampa : second 
tau_nmda : second
tau_gaba : second
I_ext : amp
tauDA : second
tauDATonic : second
'''

lifSyns = '''
dg_ampa/dt = -g_ampa / tau_ampa : siemens
dg_nmda/dt = -g_nmda / tau_nmda : siemens
dg_gaba/dt = -g_gaba / tau_gaba : siemens
dDA/dt = -DA/tauDA : 1 
dDATonic/dt = (DA-DATonic)/tauDATonic : 1
'''

eqLIF = lifPars +'''
dV/dt = ( I_L + R * (I_AMPA + I_NMDA + I_GABA + I_ext)) / tau + SIGMA*sqrt(2/tau)*xi  : volt (unless refractory)
I_L = (E_L - V) : volt
I_AMPA = g_ampa * (E_EXC - V) : volt*siemens
I_NMDA = g_nmda * (E_EXC - V) : volt*siemens
I_GABA = g_gaba * (E_INH - V) : volt*siemens
''' + lifSyns

eqAction = lifPars + '''
IEreset : amp 
gamble : 1
p : 1
last : second
dV/dt = ((E_L - V) + R * (g_ampa * (E_EXC - V) + g_nmda * (E_EXC - V) + g_gaba * (E_INH - V) + I_ext*(1+clip((t-last)/(1000*ms)-10, 0, 10)))) / tau + SIGMA*sqrt(2/tau)*xi : volt (unless refractory)
''' + lifSyns # MaxTime is 10 seconds

# NOTE: this learning method, using the chosen flag, restricts the usage of the signal to learning time
learningModel = '''
    lbd : 1 \n eta : 1
    dw/dt = ((chosen*(DA - DATonic)**lbd)*alpha - int((w/siemens) < 0)*w)/dt : siemens (clock-driven)
    '''

excModel = 'w : siemens \n eta : 1'
excSyn = '''
    g_ampa += w
    g_nmda += w*ExcRatio'''
modExcSyn = '''
    g_ampa += w*(1+eta*DA)
    g_nmda += w*(1+eta*DA)*ExcRatio'''


inhModel = 'w : siemens \n eta : 1'
inhSyn = 'g_gaba += w'
modInhSyn = 'g_gaba += w*(1+eta*DA)'
