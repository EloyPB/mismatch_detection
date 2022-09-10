# Figures 3.4-6

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
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
        print(t)



# THREE COMBINATIONS
sizes = [3,1]
numDendrites = [1,2]
sPairs = [1,1]
sPairWeights = snPairWeights = [np.array([1]), np.array([0])]
 
# alpha = 0
# taggingRate = 0.001
# captureRate = 1
alpha = 0.95
taggingRate = 0.2
captureRate = 0.005


sneuron0 = SNeurons("A", sizes[0], numDendrites[0], sPairs[0], sPairWeights[0], snPairWeights[0],
                    tau=50, ageingTau=1000000, alpha = alpha, taggingRate = taggingRate, 
                    captureRate = captureRate, noiseTau = 500)
    
sneuron1 = SNeurons("B", sizes[1], numDendrites[1], sPairs[1], sPairWeights[1], snPairWeights[1],
                    tau=50, ageingTau=1000000, alpha = alpha, taggingRate = taggingRate, 
                    captureRate = captureRate, noiseTau = 300)
    

sneuron0.initializeWeights()
    
sneuron1.enableConnections(fromGroups=[sneuron0], fromNeurons=[0], fromSPairs=[0], toSPair=0)
sneuron1.initializeWeights()
    
sneurons = [sneuron0, sneuron1]

enabled = [np.array([1]), np.array([1])]

# modes: 0 -> dendritic clustering, 1 -> weight homogenization
mode = 1
if mode == 0:
    reps = 2
    tstable = 2000
#     reps = 7
#     tstable = 500
    for rep in range(reps):
        print(rep)
        run(sneurons, 0, tstable, [np.zeros((3)),np.ones((1))], [[1,1,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
        run(sneurons, 0, 100, [np.zeros((3)),-1*np.ones((1))], [[0,0,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
        if rep < reps-1:
            run(sneurons, 0, tstable, [np.zeros((3)),np.ones((1))], [[0,1,1],[0]],
                [np.array([[0,0]]),np.array([[1,1]])])
            run(sneurons, 0, 100, [np.zeros((3)),-1*np.ones((1))], [[0,0,0],[0]],
                [np.array([[0,0]]),np.array([[1,1]])])

     
    t0 = 0
    run(sneurons, t0, t0+500, [np.zeros((3)),-np.ones((1))], [[0,0,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, t0+500, t0+1000, [np.zeros((3)),np.zeros((1))], [[1,0,1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, t0+1000, t0+1250, [np.zeros((3)),np.zeros((1))], [[0,0,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, t0+1250, t0+1750, [np.zeros((3)),np.zeros((1))], [[0,1,1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, t0+1750, t0+2000, [np.zeros((3)),np.zeros((1))], [[0,0,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, t0+2000, t0+2500, [np.zeros((3)),np.zeros((1))], [[1,1,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, t0+2500, t0+2750, [np.zeros((3)),np.zeros((1))], [[0,0,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
     
elif mode == 1:
    run(sneurons, 0, 2000, [np.zeros((3)),np.ones((1))], [[1,0,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, 2000, 6000, [np.zeros((3)),np.ones((1))], [[1,1,0],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    run(sneurons, 6000, 10000, [np.zeros((3)),np.ones((1))], [[1,1,1],[0]], [np.array([[0,0]]), np.array([[1,1]])])
    
# PLOT ACTIVITIES AND WEIGHTS
# per groupNum one set of figures, per sPair one figure, per dendrite one plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')

def figsize(scale):
    fig_width_pt = 437.46112                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*1            # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

plt.rc('figure', figsize=figsize(1))

tMax = len(u)
weightColors = ['C5', 'C7', 'k']
orangeCM = LinearSegmentedColormap.from_list('orangeCM', [colors.to_rgba('C1',0), colors.to_rgba('C1',1)], N=100)
greenCM = LinearSegmentedColormap.from_list('greenCM', [colors.to_rgba('C2',0), colors.to_rgba('C2',1)], N=100)
blueCM = LinearSegmentedColormap.from_list('blueCM', [colors.to_rgba('C0',0), colors.to_rgba('C0',1)], N=100)

inputNames = ["A", "B", "C"]

fig, axes = plt.subplots(2+numDendrites[1], 1, sharex='col', gridspec_kw = {'height_ratios': (6,2,10,10)})

# axes[0].set_xlim([0,tMax])
uABC = np.array([u[t][0] for t in range(tMax)])
axes[0].imshow(uABC.T, vmin=0, vmax=1, cmap=blueCM, origin='upper', aspect='auto')
sABC = np.array([should[t][0][0] for t in range(tMax)])
axes[0].imshow(sABC.T, vmin=0, vmax=1, cmap=greenCM, origin='upper', aspect='auto')
snABC = np.array([shouldnt[t][0][0] for t in range(tMax)])
axes[0].imshow(snABC.T, vmin=0, vmax=1, cmap=orangeCM, origin='upper', aspect='auto')

plt.sca(axes[0])
plt.yticks([0,1,2], inputNames)
axes[0].yaxis.set_ticks(np.arange(-0.5,2,1), minor="true")
axes[0].yaxis.grid(True, which="minor")
# axes[0].set_ylabel("Neuron\n activations")

uY = np.array([u[t][1] for t in range(tMax)])
axes[1].imshow(uY.T, vmin=0, vmax=1, cmap=blueCM, aspect='auto')
sY = np.array([should[t][1][0] for t in range(tMax)])
axes[1].imshow(sY.T, vmin=0, vmax=1, cmap=greenCM, aspect='auto')
snY = np.array([shouldnt[t][1][0] for t in range(tMax)])
axes[1].imshow(snY.T, vmin=0, vmax=1, cmap=orangeCM, aspect='auto')
plt.sca(axes[1])
plt.yticks([0], ['Y'])

for branchNum in range(numDendrites[1]):           
    axes[2+branchNum].set_ylim([-0.1,1.1])
    for inputNum in range(sizes[0]):
        weights = np.array([wShould[t][1][0][inputNum,0,branchNum] for t in range(tMax)])
        axes[2+branchNum].plot(weights, label="$w_{H_"+inputNames[inputNum]+r"\to S_Y,d_"+str(branchNum+1)+"}$",
                               color=weightColors[inputNum])
        axes[2+branchNum].legend(loc='upper right', ncol=sizes[0])
        axes[2+branchNum].set_ylabel("Weights onto\n dendrite "+str(branchNum+1)+" of $S_Y$")
    
    axes[-1].set_xlabel("Simulation steps")

fig.text(0.0425, 0.835, 'Neuron\n activations', ha='center', va='center', rotation='vertical', 
         linespacing=1)
plt.tight_layout()

if mode == 0:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC//THESIS/WRITING/TEX/Chapter3/Figs/dendrites'+str(tstable)+'.pdf')
elif mode == 1:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC//THESIS/WRITING/TEX/Chapter3/Figs/homogenization.pdf')
plt.show()



