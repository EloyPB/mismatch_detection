# Figures 3.10-12

import numpy as np
import matplotlib.pyplot as plt
from SNeurons.SNeurons10 import SNeurons


transientBlock = True
later = False

groupNames = ["1", "2"]
sizes = [1,1]
numGroups = len(sizes)
numDendrites = [1,1]
sPairs = [2,1]
sPairsMax = max(sPairs)
sPairHasWeights = [[True,False], [True]]

a = SNeurons("A", size=sizes[0], numDendrites=numDendrites[0], sPairs=sPairs[0], tau=100, 
             sPairWeights=np.array([0,1]), snPairWeights=np.array([0,1]), ageingTau=10000, alpha=0, taggingRate=0.0006,
             captureRate=1, noiseTau = 1000, transientBlock=transientBlock)

b = SNeurons("B", size=sizes[1], numDendrites=numDendrites[1], sPairs=sPairs[1], tau=100, 
             sPairWeights=np.array([1]), snPairWeights=np.array([1]), ageingTau=10000, alpha=0, taggingRate=0.0006, 
             captureRate=1, noiseTau = 1000, transientBlock=transientBlock)

        
a.enableConnections(fromGroups=[b], fromNeurons=[0], fromSPairs=[0], toSPair=0)
a.initializeWeights()

b.enableConnections(fromGroups=[a], fromNeurons=[0], fromSPairs=[0], toSPair=0)
b.initializeWeights()


groups = [a, b]


H = [[np.zeros(sizes[groupNum])] for groupNum in range(numGroups)]
HPast = [[] for _ in range(numGroups)]
S = [[] for _ in range(numGroups)]
uS = [[] for _ in range(numGroups)]
SN = [[] for _ in range(numGroups)]
WS = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
WSN = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
          
def run(steps, hInput, sInput, enabled, learn):
    for _ in range(steps):
        for groupNum, group in enumerate(groups):
            h, hP, s, sn, us = group.step(hInput[groupNum], sInput[groupNum], enabled[groupNum], 0, learn[groupNum])
            H[groupNum].append(h)
            HPast[groupNum].append(hP+0)
            S[groupNum].append(s)
            SN[groupNum].append(sn)
            uS[groupNum].append(us)
            for sPair in range(sPairs[groupNum]):
                if sPairHasWeights[groupNum][sPair]:
                    WS[groupNum][sPair].append(group.shouldW[sPair]+0)
                    WSN[groupNum][sPair].append(group.shouldntW[sPair]+0)


# RUN SIMULATION

# learn and off
enabled = [np.array([1,1]), np.array([1])]
learn = [np.array([[1,1],[0,0]]), np.array([[1,1]])]

hInput = [np.zeros(sizes[0]), np.ones(sizes[1])]
sInput = [np.array([[0],[1]]), np.array([[0]])]
run(1000, hInput, sInput, enabled, learn)

if later:
    asw = a.shouldW
    asnw = a.shouldntW
    bsw = b.shouldW
    bsnw = b.shouldntW
    
    H = [[np.zeros(sizes[groupNum])] for groupNum in range(numGroups)]
    HPast = [[] for _ in range(numGroups)]
    S = [[] for _ in range(numGroups)]
    uS = [[] for _ in range(numGroups)]
    SN = [[] for _ in range(numGroups)]
    WS = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
    WSN = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
    
    A = SNeurons("A", size=sizes[0], numDendrites=numDendrites[0], sPairs=sPairs[0], tau=100, 
                 sPairWeights=np.array([0,1]), snPairWeights=np.array([0,1]), ageingTau=10000, alpha=0, 
                 taggingRate=0.00012, captureRate=1, noiseTau = 1000, transientBlock=transientBlock)
    
    B = SNeurons("B", size=sizes[1], numDendrites=numDendrites[1], sPairs=sPairs[1], tau=100, 
                 sPairWeights=np.array([1]), snPairWeights=np.array([1]), ageingTau=10000, alpha=0, 
                 taggingRate=0.00012, captureRate=1, noiseTau = 1000, transientBlock=transientBlock)
    
    A.enableConnections(fromGroups=[B], fromNeurons=[0], fromSPairs=[0], toSPair=0)
    A.initializeWeights()
    
    B.enableConnections(fromGroups=[A], fromNeurons=[0], fromSPairs=[0], toSPair=0)
    B.initializeWeights()
    
    A.shouldW = asw
    A.shouldntW = asnw
    B.shouldW = bsw 
    B.shouldntW = bsnw 
    
    groups = [A, B]
    
else:
    hInput = [np.zeros(sizes[0]), np.zeros(sizes[1])]
    sInput = [np.array([[0],[0]]), np.array([[0]])]
    run(500, hInput, sInput, enabled, learn)


reps = 3 if later else 2
for rep in range(reps):
    # A On
    hInput = [np.zeros(sizes[0]), np.zeros(sizes[1])]
    sInput = [np.array([[0],[1]]), np.array([[0]])]
    run(600, hInput, sInput, enabled, learn)
     
    # A Off
    hInput = [np.zeros(sizes[0]), np.zeros(sizes[1])]
    sInput = [np.array([[0],[0]]), np.array([[0]])]
    run(300, hInput, sInput, enabled, learn)
                         

# PLOT ACTIVITIES

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')

ratio = 1.2 if transientBlock else 0.6
def figsize(scale):
    fig_width_pt = 437.46112                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio                    # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

# plt.rc('figure', figsize=figsize(1))

plotSizes = [1,1,0.5,1] if transientBlock else [1,1]

fig, ax = plt.subplots(4 if transientBlock else 2, 2, sharex='col', sharey='row', 
                       gridspec_kw = {'height_ratios': plotSizes})

inputNames = ["2","1"]
titles = ["Circuit 1", "Circuit 2"]
tSim = len(uS[0])
for groupNum in range(numGroups):
    ax[0,groupNum].plot(H[groupNum], label="$H_"+groupNames[groupNum]+"$")
    ax[0,groupNum].plot(np.array(S[groupNum])[:,0], 'C2', label="$S_"+groupNames[groupNum]+"$")
    ax[0,groupNum].plot(np.array(SN[groupNum])[:,0], 'C1', label="$N_"+groupNames[groupNum]+"$")
    
    ax[0,groupNum].set_ylim([-0.05,1.05])
    ax[0,groupNum].legend(loc='lower right')
    ax[0,groupNum].set_title(titles[groupNum], size='medium')
    
    if transientBlock:        
        ps = 0
        dif = np.zeros(tSim)
        blockC = np.zeros(tSim)
        enable = np.zeros(tSim)
        for t in range(tSim):
            dif[t] = np.array(uS[groupNum])[t,0]-ps
            ps = np.array(uS[groupNum])[t,0]
            blockC[t] = blockC[t-1]-1 if abs(dif[t]) < 0.018 else 150
            enable[t] = 1 if blockC[t] <= 0 else 0
            
        ax[1,groupNum].plot(dif, color='C7')
        ax[1,groupNum].plot([0.018 for _ in range(tSim)], '--', linewidth=0.5, color='xkcd:grey') 
        ax[1,groupNum].plot([-0.018 for _ in range(tSim)], '--', linewidth=0.5, color='xkcd:grey') 
        
        ax[2,groupNum].plot(enable, color='C7')
        
        if groupNum == 0:
            ax[1,groupNum].set_ylabel("Diff(S)")
            ax[2,groupNum].set_ylabel("Enable \n learning in S")
        
    ax[-1,groupNum].plot(np.array(WS[groupNum][0])[:,0,0,0], color='k',
                         label="$w_{H_"+inputNames[groupNum]+r"\to S_"+groupNames[groupNum]+"}$")
    ax[-1,groupNum].legend(loc='lower right')
    

    if not later:
        ax[-1,groupNum].set_ylim([-0.05,1.05])
    
    ax[-1,groupNum].set_xlabel("Simultaion steps")
    if groupNum == 0:
        ax[0,groupNum].set_ylabel("Neuron activations")
        ax[-1,groupNum].set_ylabel("Weights onto S")

if transientBlock and not later:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/enable3.pdf')
elif not transientBlock and not later:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/enable1.pdf')
elif not transientBlock and later:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/enable2.pdf')
plt.show()
