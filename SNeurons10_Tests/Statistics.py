# Produces figures 3.7 and 3.8

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from SNeurons.SNeurons10 import SNeurons

u = []
uPast = []
should = []
shouldnt = []
wShould = []
wShouldSuccess = []
wShouldnt = []

def run(sneurons, t0, tf, uInput, sInput, learn):
    for t in range(t0,tf):
        uT = []
        uPastT = []
        shouldT = []
        shouldntT = []
        wShouldT = []
        wShouldSuccessT = []
        wShouldntT = []
        for n in range(len(sneurons)):
            uu, uuPast, sshould, sshouldnt = sneurons[n].step(uInput[n], sInput[n], enabled[n], 0, learn[n])
            uT.append(uu)
            uPastT.append(uuPast)
            shouldT.append(sshould)
            shouldntT.append(sshouldnt)
            wShouldT.append(deepcopy(sneurons[n].shouldW))
            wShouldntT.append(deepcopy(sneurons[n].shouldntW))
            wShouldSuccessT.append(deepcopy(sneurons[n].shouldWFitness))
        u.append(uT)
        uPast.append(uPastT)
        should.append(shouldT)
        shouldnt.append(list(shouldntT))
        wShould.append(wShouldT[:])
        wShouldnt.append(wShouldntT[:])
        wShouldSuccess.append(wShouldSuccessT[:])
#         print(t)



# THREE COMBINATIONS
sizes = [1,1]
numDendrites = [1,1]
sPairs = [1,1]
sPairWeights = snPairWeights = [np.array([1]), np.array([0])]
 
alpha = 0
taggingRate = 0.0001
captureRate = 1
# alpha = 0.95
# taggingRate = 0.2
# captureRate = 0.005


sneuron0 = SNeurons("A", sizes[0], numDendrites[0], sPairs[0], sPairWeights[0], snPairWeights[0],
                    tau=50, ageingTau=4000, alpha = alpha, taggingRate = taggingRate, 
                    captureRate = captureRate, noiseTau = 500)
    
sneuron1 = SNeurons("B", sizes[1], numDendrites[1], sPairs[1], sPairWeights[1], snPairWeights[1],
                    tau=50, ageingTau=4000, alpha = alpha, taggingRate = taggingRate, 
                    captureRate = captureRate, noiseTau = 300)
    

sneuron0.initializeWeights()
    
sneuron1.enableConnections(fromGroups=[sneuron0], fromNeurons=[0], fromSPairs=[0], toSPair=0)
sneuron1.initializeWeights()
    
sneurons = [sneuron0, sneuron1]

enabled = [np.array([1]), np.array([1])]

# modes: 0 -> p = 0.5
mode = 1
if mode == 0:
    reps = 15
    tstable = 500
    for rep in range(reps):
        print(rep)
        run(sneurons, 0, tstable, [np.zeros((1)),np.ones((1))], [[1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
        run(sneurons, 0, int(tstable/2), [np.zeros((1)),-1*np.ones((1))], [[1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
        run(sneurons, 0, int(tstable/2), [np.zeros((1)),np.zeros((1))], [[1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
elif mode == 1:
    reps = 15
    tup = 500
    tdown = 850
    for rep in range(reps):
        print(rep)
        run(sneurons, 0, tup, [np.zeros((1)),np.ones((1))], [[1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
        run(sneurons, 0, int(tdown/2), [np.zeros((1)),-1*np.ones((1))], [[1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
        run(sneurons, 0, int(tdown/2), [np.zeros((1)),np.zeros((1))], [[1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    
# PLOT ACTIVITIES AND WEIGHTS
# per groupNum one set of figures, per sPair one figure, per dendrite one plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')

def figsize(scale):
    fig_width_pt = 437.46112                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*0.6            # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

plt.rc('figure', figsize=figsize(1))

tMax = len(u)
inputNames = ["A", "B"]

fig, axes = plt.subplots(2, 1, sharex='col', gridspec_kw = {'height_ratios': (1,2)})

axes[0].set_xlim([0,tMax])
uABC = np.array([u[t][1] for t in range(tMax)])
axes[0].plot(uABC, label="$H_1$")
snABC = np.array([shouldnt[t][1][0] for t in range(tMax)])
axes[0].plot(snABC, label="$S_1$")
sABC = np.array([should[t][1][0] for t in range(tMax)])
axes[0].plot(sABC, label="$N_1$")
axes[0].legend()
axes[0].set_ylabel("Neuron\n Activations")

        
axes[1].set_ylim([-0.1,1.1])
for inputNum in range(sizes[0]):
    weights = np.array([wShould[t][1][0][inputNum,0,0] for t in range(tMax)])
    axes[1].plot(weights, label=r"$w_{H_2 \to S_1}$")
    axes[1].legend(loc='upper right', ncol=sizes[0])
    axes[1].set_ylabel("Weights onto $S_1$")
    
    axes[-1].set_xlabel("Simulation steps")

plt.tight_layout()

if mode == 0:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC//THESIS/WRITING/TEX/Chapter3/Figs/statistics05.pdf')
if mode == 1:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC//THESIS/WRITING/TEX/Chapter3/Figs/statistics03.pdf')
plt.show()



