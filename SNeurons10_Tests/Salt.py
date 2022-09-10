# Figure 3.2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from SNeurons.SNeurons10 import SNeurons


groupNames = ["taste features", "visual features", "salt"]
sizes = [3,3,1]
numGroups = len(sizes)
numDendrites = [1,1,1]
sPairs = [2,2,2]
sPairsMax = max(sPairs)
sPairHasWeights = [[False,True], [False,True], [True,True]]

taste = SNeurons("taste features", size=sizes[0], numDendrites=numDendrites[0], sPairs=sPairs[0], tau=100, 
                 sPairWeights=np.array([1,0]), snPairWeights=np.array([1,0]), ageingTau=10000, alpha=0, 
                 taggingRate=0.001, captureRate=1, noiseTau = 1000, transientBlock=False)

vision = SNeurons("visual features", size=sizes[1], numDendrites=numDendrites[1], sPairs=sPairs[1], tau=100, 
                  sPairWeights=np.array([1,0]), snPairWeights=np.array([1,0]), ageingTau=10000, alpha=0, 
                  taggingRate=0.001, captureRate=1, noiseTau = 1000, transientBlock=False)

salt = SNeurons("salt", size=sizes[2], numDendrites=numDendrites[2], sPairs=sPairs[2], tau=100, 
                sPairWeights=np.array([8,1]), snPairWeights=np.array([8,1]), ageingTau=10000, alpha=0, 
                taggingRate=0.001, captureRate=1, noiseTau = 1000, transientBlock=False)
        
taste.enableConnections(fromGroups=[salt], fromNeurons=[0], fromSPairs=[0], toSPair=1)
taste.initializeWeights()

vision.enableConnections(fromGroups=[salt], fromNeurons=[0], fromSPairs=[0], toSPair=1)
vision.initializeWeights()

salt.enableConnections(fromGroups=[taste], fromNeurons=[0], fromSPairs=[0], toSPair=0)
salt.enableConnections(fromGroups=[vision], fromNeurons=[0], fromSPairs=[0], toSPair=1)
salt.initializeWeights()

groups = [taste, vision, salt]


H = [[np.zeros(sizes[groupNum])] for groupNum in range(numGroups)]
HPast = [[] for _ in range(numGroups)]
S = [[] for _ in range(numGroups)]
SN = [[] for _ in range(numGroups)]
WS = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
WSN = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
          
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
                    WSN[groupNum][sPair].append(group.shouldntW[sPair]+0)

# learn features of salt
hInput = [np.zeros(sizes[0]), np.zeros(sizes[1]), 10*np.ones(sizes[2])]
sInput = [np.array([[1,0,0],[0,0,0]]), np.array([[0,1,1],[0,0,0]]), np.array([[0],[0]])]
enabled = [np.array([1,1]), np.array([1,1]), np.array([1,1])]
learn = [np.array([[0,0],[1,1]]), np.array([[0,0],[1,1]]), np.array([[1,1],[1,1]])]
run(600, hInput, sInput, enabled, learn)

# everything off
hInput = [np.zeros(sizes[0]), np.zeros(sizes[1]), np.zeros(sizes[2])]
sInput = [np.array([[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0]]), np.array([[0],[0]])]
learn = [np.array([[0,0],[0,0]]), np.array([[0,0],[0,0]]), np.array([[0,0],[0,0]])]
run(300, hInput, sInput, enabled, learn)

# visual alone
sInput = [np.array([[0,0,0],[0,0,0]]), np.array([[0,1,1],[0,0,0]]), np.array([[0],[0]])]
enabled = [np.array([1,0]), np.array([1,1]), np.array([0,1])]
run(300, hInput, sInput, enabled, learn)

# visual and wrong taste
sInput = [np.array([[0,1,0],[0,0,0]]), np.array([[0,1,1],[0,0,0]]), np.array([[0],[0]])]
tv = 0
while(np.sum(H[0][-1]) < 0.1):
    run(1, hInput, sInput, enabled, learn)
    tv += 1
enabled = [np.array([1,1]), np.array([1,1]), np.array([1,1])]
run(300, hInput, sInput, enabled, learn)

# everything off
sInput = [np.array([[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0]]), np.array([[0],[0]])]
run(300, hInput, sInput, enabled, learn)

# taste alone
sInput = [np.array([[1,0,0],[0,0,0]]), np.array([[0,0,0],[0,0,0]]), np.array([[0],[0]])]
enabled = [np.array([1,0]), np.array([1,0]), np.array([1,0])]
run(300, hInput, sInput, enabled, learn)

# taste and wrong visual 
sInput = [np.array([[1,0,0],[0,0,0]]), np.array([[1,0,0],[0,0,0]]), np.array([[0],[0]])]
tt = 0
while(np.sum(H[1][-1]) < 0.1):
    run(1, hInput, sInput, enabled, learn)
    tt += 1
enabled = [np.array([1,1]), np.array([1,1]), np.array([1,1])]
run(300, hInput, sInput, enabled, learn)



# PLOT ACTIVITIES

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')

def figsize(scale):
    fig_width_pt = 437.46112                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

plt.rc('figure', figsize=figsize(1))

# define colormaps
orangeCM = LinearSegmentedColormap.from_list('orangeCM', [colors.to_rgba('C1',0), colors.to_rgba('C1',1)], N=100)
greenCM = LinearSegmentedColormap.from_list('greenCM', [colors.to_rgba('C2',0), colors.to_rgba('C2',1)], N=100)
blueCM = LinearSegmentedColormap.from_list('blueCM', [colors.to_rgba('C0',0), colors.to_rgba('C0',1)], N=100)
beigeCM = LinearSegmentedColormap.from_list('beigeCM', [colors.to_rgba('xkcd:beige',0),
                                                        colors.to_rgba('xkcd:beige',1)], N=100)
    
fig, ax = plt.subplots(numGroups, sPairsMax, sharex='col', sharey='row', gridspec_kw = {'height_ratios':sizes})

ylabels = ["Taste features", "Visual features", "Salt"]
titles = [["S-Pair 1 receiving\n unmodelled low-level input", "S-Pair 2 receiving\n input from salt"], 
          ["unmodelled low-level input", "input from salt"],
          ["input from taste features", "input from visual features"]]

for groupNum in range(numGroups):
    for sPair in range(sPairs[groupNum]): 
        im0 = ax[groupNum,sPair].imshow(np.array(HPast[groupNum]).T, cmap=beigeCM, origin='lower', aspect='auto', 
                                        vmin=0, vmax=1)
        im1 = ax[groupNum,sPair].imshow(np.array(H[groupNum]).T, cmap=blueCM, origin='lower', aspect='auto', 
                                        vmin=0, vmax=1)
        im2 = ax[groupNum,sPair].imshow(np.array(S[groupNum])[:,sPair].T, cmap=greenCM, origin='lower', aspect='auto', 
                                        vmin=0, vmax=1)
        im3 = ax[groupNum,sPair].imshow(np.array(SN[groupNum])[:,sPair].T, cmap=orangeCM, origin='lower', 
                                        aspect='auto', vmin=0, vmax=1)
        
        # plot lines
        ax[groupNum,sPair].plot(np.array([600, 600]), np.array([-0.5, sizes[groupNum]-0.5]), '--',
                                linewidth=0.5, color='xkcd:grey') 
        ax[groupNum,sPair].plot(np.array([900, 900]), np.array([-0.5, sizes[groupNum]-0.5]), '--',
                                linewidth=0.5, color='xkcd:grey') 
        ax[groupNum,sPair].plot(np.array([1200, 1200]), np.array([-0.5, sizes[groupNum]-0.5]), '--',
                                linewidth=0.5, color='xkcd:grey') 
        ax[groupNum,sPair].plot(np.array([1500+tv, 1500+tv]), np.array([-0.5, sizes[groupNum]-0.5]), '--',
                                linewidth=0.5, color='xkcd:grey') 
        ax[groupNum,sPair].plot(np.array([1800+tv, 1800+tv]), np.array([-0.5, sizes[groupNum]-0.5]), '--',
                                linewidth=0.5, color='xkcd:grey') 
        ax[groupNum,sPair].plot(np.array([2100+tv, 2100+tv]), np.array([-0.5, sizes[groupNum]-0.5]), '--',
                                linewidth=0.5, color='xkcd:grey') 
        
                
        ax[groupNum,sPair].yaxis.set_ticks(range(0,sizes[groupNum]))
        ax[groupNum,sPair].yaxis.set_ticklabels(range(1,sizes[groupNum]+1))
        ax[groupNum,sPair].yaxis.set_ticks(np.arange(-.5, sizes[groupNum], 1), minor=True);
        ax[groupNum,sPair].yaxis.grid(which = 'minor')
        for tic in ax[groupNum,sPair].yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        
        ax[groupNum,sPair].set_title(titles[groupNum][sPair], size='medium')
        
        if sPair == 0:
            ax[groupNum,sPair].set_ylabel(ylabels[groupNum])
    else:
        sPair += 1
        while sPair < sPairsMax:
            ax[groupNum,sPair].axis('off')
            sPair += 1
            
for axis in ax[-1,:]:
    axis.set_xlabel("Simulation steps")
    

for axis in ax[0,:]:     
    plt.text(0.12, 5/6, "A", horizontalalignment='center', verticalalignment='center', 
             transform=axis.transAxes, fontsize='small')
    axis.add_artist(Ellipse((0.12, 0.85), 0.07, 0.17, transform=axis.transAxes, fill=False, linewidth=0.5))
    plt.text(0.41, 5/6, "B", horizontalalignment='center', verticalalignment='center', 
             transform=axis.transAxes, fontsize='small')
    axis.add_artist(Ellipse((0.41, 0.85), 0.07, 0.17, transform=axis.transAxes, fill=False, linewidth=0.5))
    plt.text(0.545, 5/6, "C", horizontalalignment='center', verticalalignment='center', 
             transform=axis.transAxes, fontsize='small')
    axis.add_artist(Ellipse((0.545, 0.85), 0.07, 0.17, transform=axis.transAxes, fill=False, linewidth=0.5))
    plt.text(0.795, 5/6, "D", horizontalalignment='center', verticalalignment='center', 
             transform=axis.transAxes, fontsize='small')
    axis.add_artist(Ellipse((0.795, 0.85), 0.07, 0.17, transform=axis.transAxes, fill=False, linewidth=0.5))
    plt.text(0.925, 5/6, "E", horizontalalignment='center', verticalalignment='center', 
             transform=axis.transAxes, fontsize='small')
    axis.add_artist(Ellipse((0.925, 0.85), 0.07, 0.17, transform=axis.transAxes, fill=False, linewidth=0.5))

plt.tight_layout()
    
# plot colorbars
fig.subplots_adjust(right=0.82)
cbar = np.linspace(0,1,100).reshape(100,1)

pos_high = ax[0,0].get_position()
pos_low = ax[-1,0].get_position()
y_span = pos_high.y0 + pos_high.height - pos_low.y0

def prettyColorBar(cbAxes, colorMap, name):
    cbAxes.imshow(cbar, cmap=colorMap, aspect='auto', origin='lower')
    cbAxes.axes.set_ylabel(name)
    cbAxes.yaxis.set_label_position("right")
    cbAxes.axes.get_xaxis().set_visible(False)
    cbAxes.yaxis.tick_right()
    cbAxes.yaxis.set_ticks(np.linspace(0,100,3))
    cbAxes.yaxis.set_ticklabels(np.linspace(0,1,3))

cbar_ax_sn = fig.add_axes([0.87, pos_low.y0, 0.015, y_span*4/15])
prettyColorBar(cbar_ax_sn, orangeCM, "should not")

pos = pos_low.y0 + y_span*4/15 + y_span/10
cbar_ax_s = fig.add_axes([0.87, pos, 0.015, y_span*4/15])
prettyColorBar(cbar_ax_s, greenCM, "should")

pos += y_span*4/15 + y_span/10
cbar_ax_h = fig.add_axes([0.87, pos, 0.015, y_span*4/15])
prettyColorBar(cbar_ax_h, blueCM, "head")

plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/salt.pdf')

# # PLOT WEIGHTS
# inputSizes = [[] for _ in range(numGroups)]
# for groupNum in range(numGroups):
#     sPairNum = 0
#     for sPair in range(sPairs[groupNum]):
#         if sPairHasWeights[groupNum][sPair]:
#             inputSizes[groupNum].append(len(WS[groupNum][sPair][0]))
#             sPairNum += 1
#    
# for groupNum, group in enumerate(groups):
#     if sum(sPairHasWeights[groupNum]):
#         figW, axW = plt.subplots(sum(sPairHasWeights[groupNum]), numDendrites[groupNum], sharex='col', sharey='row',
#                                  gridspec_kw = {'height_ratios': inputSizes[groupNum]})
#         figW.suptitle("weights to "+groupNames[groupNum])
#         sPairNum = 0
#         for sPair in range(sPairs[groupNum]):
#             if sPairHasWeights[groupNum][sPair]:
#                 for dendriteNum in range(numDendrites[groupNum]):
#                     size = sizes[groupNum]
#                     inputSize = inputSizes[groupNum][sPairNum]
#                        
#                     wr = np.zeros((size*inputSize, len(WS[groupNum][sPair])))
#                     for n in range(size):
#                         for i in range(inputSize):
#                             wr[n*inputSize+i,:] = np.array(WS[groupNum][sPair])[:,i,n,dendriteNum]
#                                    
#                     if sum(sPairHasWeights[groupNum])== 1 and numDendrites[groupNum] ==1:
#                         ax = axW
#                     elif sum(sPairHasWeights[groupNum]) == 1:
#                         ax = axW[dendriteNum]
#                     elif numDendrites[groupNum] == 1:
#                         ax = axW[sPairNum]
#                     else:
#                         ax = axW[sPairNum, dendriteNum]
#                              
#                     cm = ax.pcolormesh(wr, vmin=0, vmax=1)
# 
#                     axr = ax.twinx()
#                     axr.yaxis.set_ticks(np.arange(0.5, size*inputSize, 1))
#                     axr.yaxis.set_ticklabels(group.weightNames[sPair]*size)
#                     axr.yaxis.set_ticks(range(size*inputSize+1), minor=True)
#                     axr.yaxis.grid(which='minor')
#                     for tic in axr.yaxis.get_major_ticks():
#                         tic.tick1On = tic.tick2On = False
#                         
#                     ax.yaxis.set_ticks(np.arange(0.5*inputSize,size*inputSize,inputSize))
#                     ax.yaxis.set_ticklabels(range(size))
#                     for tic in ax.yaxis.get_major_ticks():
#                         tic.tick1On = tic.tick2On = False
#                     ax.yaxis.set_ticks(range(0,size*inputSize,inputSize), minor=True)
#                     ax.yaxis.grid(which='minor', linewidth=3, color='w')
# 
#                 sPairNum += 1
#                  
#         figW.subplots_adjust(right=0.82)
#         cbar_ax = figW.add_axes([0.90, 0.25, 0.02, 0.5])
#         figW.colorbar(cm, cax=cbar_ax)
#                 
#         cols = ['connections onto dendrite {}'.format(dendriteNum) for dendriteNum in range(numDendrites[groupNum])]
#         rows = ['connections\nonto s-pair {}'.format(sPair) for sPair in range(sPairs[groupNum]) if 
#                 sPairHasWeights[groupNum][sPair]]
#         if sum(sPairHasWeights[groupNum])== 1 and numDendrites[groupNum] ==1:
#             axW.set_title(cols[0], size='medium')
#             axW.set_ylabel(rows[0])
#         elif sum(sPairHasWeights[groupNum])== 1:
#             axW[0].set_ylabel(rows[0])
#             for ax, col in zip(axW, cols):
#                 ax.set_title(col, size='medium')
#         elif numDendrites[groupNum] == 1:
#             axW[0].set_title(cols[0], size='medium')
#             for ax, row in zip(axW, rows):
#                 ax.set_ylabel(row)
#         else:
#             for ax, col in zip(axW[0], cols):
#                 ax.set_title(col, size='medium')    
#             for ax, row in zip(axW[:,0], rows):
#                 ax.set_ylabel(row)
        
plt.show()
