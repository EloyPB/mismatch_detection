# Figures 3.14,15

import numpy as np
import matplotlib.pyplot as plt
from SNeurons.SNeurons10 import SNeurons

delay = True
saveFigs = False

if delay:
    alpha = 0.99
    taggingRate = 0.4
    captureRate = 0.005
    noiseTau = 200
    dend = 1
else:
    alpha = 0.97
    taggingRate = 0.05
    captureRate = 0.007
    noiseTau = 300
    dend = 2
    
reps = 10

groupNames = ["A", "B"]
sizes = [1,1]
numGroups = len(sizes)
numDendrites = [1,dend]
sPairs = [2,2]
sPairsMax = max(sPairs)
sPairHasWeights = [[False,False], [False,True]]

A = SNeurons("A", size=sizes[0], numDendrites=numDendrites[0], sPairs=sPairs[0], tau=100, 
             sPairWeights=np.array([1,0]), snPairWeights=np.array([1,0]), ageingTau=5000, alpha=0, taggingRate=0.001, 
             captureRate=1, noiseTau = 100, transientBlock=True)

B = SNeurons("B", size=sizes[1], numDendrites=numDendrites[1], sPairs=sPairs[1], tau=100, 
             sPairWeights=np.array([1,0]), snPairWeights=np.array([1,0]), ageingTau=5000, alpha=alpha, 
             taggingRate=taggingRate, captureRate=captureRate, noiseTau = noiseTau, transientBlock=False)


A.initializeWeights()
        
B.enableConnections(fromGroups=[A], fromNeurons=[2], fromSPairs=[1], toSPair=1)
B.initializeWeights()


groups = [A, B]


H = [[np.zeros(sizes[groupNum])] for groupNum in range(numGroups)]
HPast = [[] for _ in range(numGroups)]
S = [[] for _ in range(numGroups)]
SN = [[] for _ in range(numGroups)]
WS = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
WSTagUp = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
WSTagDown = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
          
def run(steps, hInput, sInput, enabled, learn):
    for _ in range(steps):
        for groupNum, group in enumerate(groups):
            h, hP, s, sn = group.step(hInput[groupNum], sInput[groupNum], enabled[groupNum], 0, learn[groupNum])
            H[groupNum].append(h)
            HPast[groupNum].append(hP+0)
            S[groupNum].append(s)
            SN[groupNum].append(sn)
            for sPair in range(sPairs[groupNum]):
                if sPairHasWeights[groupNum][sPair]:
                    WS[groupNum][sPair].append(group.shouldW[sPair]+0)
                    WSTagUp[groupNum][sPair].append(group.shouldWTagUp[sPair]+0)
                    WSTagDown[groupNum][sPair].append(group.shouldWTagDown[sPair]+0)


hInput = [np.zeros(sizes[0]), np.zeros(sizes[1])]
learn = [np.array([[0,0],[0,0]]), np.array([[0,0],[0,0]])]


if delay == False:
    for rep in range(reps):
        # all off
        enabled = [np.array([1,1]), np.array([1,1])]
        sInput = [np.array([[0],[0]]), np.array([[0],[0]])]
        run(300, hInput, sInput, enabled, learn)
          
        # want A
        learn = [np.array([[0,0],[0,0]]), np.array([[0,0],[1,0]])]
        sInput = [np.array([[0],[1]]), np.array([[0],[0]])]
        run(100, hInput, sInput, enabled, learn)
        # does B
        sInput = [np.array([[0],[1]]), np.array([[1],[0]])]
        run(200, hInput, sInput, enabled, learn)
        # gets A
        sInput = [np.array([[1],[1]]), np.array([[1],[0]])]
        while(H[0][-1][0] < 0.05):
            run(1, hInput, sInput, enabled, learn)
        learn = [np.array([[0,0],[0,0]]), np.array([[0,0],[0,1]])]
        run(300, hInput, sInput, enabled, learn)
else:
    for rep in range(reps):
        # all off
        enabled = [np.array([1,1]), np.array([1,1])]
        sInput = [np.array([[0],[0]]), np.array([[0],[0]])]
        run(300, hInput, sInput, enabled, learn)
          
        # want A
        learn = [np.array([[0,0],[0,0]]), np.array([[0,0],[1,0]])]
        sInput = [np.array([[0],[1]]), np.array([[0],[0]])]
        run(100, hInput, sInput, enabled, learn)
        # does B
        sInput = [np.array([[0],[1]]), np.array([[1],[0]])]
        run(500, hInput, sInput, enabled, learn)
        sInput = [np.array([[0],[1]]), np.array([[0],[0]])]
        enabled = [np.array([1,1]), np.array([1,0])]
        run(150, hInput, sInput, enabled, learn)
        # gets A
        sInput = [np.array([[1],[1]]), np.array([[0],[0]])]
        while(H[0][-1][0] < 0.05):
            run(1, hInput, sInput, enabled, learn)
        learn = [np.array([[0,0],[0,0]]), np.array([[0,0],[0,1]])]
        run(300, hInput, sInput, enabled, learn)

# PLOT ACTIVITIES

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')

def figsize(scale):
    fig_width_pt = 437.46112                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*1           # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

plt.rc('figure', figsize=figsize(1))


if not delay:
    fig, ax = plt.subplots(4, 1, sharex='col')
    
    weightLabel = [r"$w_{S_A \to S_{B,d_1}}$", r"$w_{S_A \to S_{B,d_2}}$"]
    titles = ["Circuit A: S-Pair 2 receiving unmodelled motivation input",
              "Circuit B: S-Pair 2 receiving input from S-Pair 2 of Circuit A",
              "Weights onto dendrite 1 of $S_B$", "Weights onto dendrite 2 of $S_B$"]
    
    for groupNum in range(numGroups):
        ax[groupNum].plot(H[groupNum], label="$H_"+groupNames[groupNum]+"$")
        ax[groupNum].plot(np.array(S[groupNum])[:,1], 'C2', label="$S_"+groupNames[groupNum]+"$")
        
        ax[groupNum].set_ylim([-0.05,1.05])
        ax[groupNum].legend(loc='lower right')
        ax[groupNum].set_title(titles[groupNum], size='medium')
    
    
    ax[2].plot(np.array(WS[groupNum][1])[:,0,0,0], label=weightLabel[0], color='k')
    ax[2].legend(loc='lower right')
    ax[2].set_title(titles[2], size='medium')
    ax[2].set_ylim([-0.05,1.05])
    

    ax[3].plot(np.array(WS[groupNum][1])[:,0,0,1], label=weightLabel[1], color='k')
    ax[3].legend(loc='lower right')
    ax[3].set_title(titles[3], size='medium')
    ax[3].set_xlabel("Simulation steps")
    ax[3].set_ylim([-0.05,1.05])
    
    plt.tight_layout()
    if saveFigs:
        plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/action.pdf')

    
else:
    fig, ax = plt.subplots(4, 1, sharex='col')
    
    weightLabel = r"$w_{S_A \to S_{B}}$"
    titles = ["Circuit A: S-Pair 2 receiving unmodelled motivation input",
              "Circuit B: S-Pair 2 receiving input from S-Pair 2 of Circuit A",
              r"Synaptic Tags of $w_{S_A \to S_B}$", "Weights onto $S_B$"]
    
    for groupNum in range(numGroups):
        ax[groupNum].plot(H[groupNum], label="$H_"+groupNames[groupNum]+"$")
        ax[groupNum].plot(np.array(S[groupNum])[:,1], 'C2', label="$S_"+groupNames[groupNum]+"$")
        
        ax[groupNum].set_ylim([-0.05,1.05])
        ax[groupNum].legend(loc='lower right')
        ax[groupNum].set_title(titles[groupNum], size='medium')
    
    ax[2].plot(np.array(WSTagUp[groupNum][1])[:,0,0,0], label="Tag Up", color='xkcd:teal')
    ax[2].plot(np.array(WSTagDown[groupNum][1])[:,0,0,0], label="Tag Down", color='xkcd:tomato')
    ax[2].legend(loc='upper right')
    ax[2].set_title(titles[2], size='medium')
    
    
    ax[3].plot(np.array(WS[groupNum][1])[:,0,0,0], label=weightLabel, color='k')
    ax[3].legend(loc='lower right')
    ax[3].set_title(titles[3], size='medium')
    ax[3].set_ylim([-0.05,1.05])
    
    plt.tight_layout()
    
    if saveFigs:
        plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/actiondelay.pdf')
        
plt.show()
