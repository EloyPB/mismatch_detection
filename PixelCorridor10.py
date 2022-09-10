import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from SNeurons.SNeurons10 import SNeurons
import pickle

# MODE

# 0 -> learn place
# 1 -> localize & get lost
# 2 -> new odour

# 3 -> learn how to go to places
# 4 -> show 

mode = 4

savefigs = True

numPixels = 6
pixels = [1,2,3,4,5,4,3,2,1,0]
sizes = [3,2,4,3]
numGroups = len(sizes)
    
if mode in (0,1,2):
    numDendrites = [1,1,1,2]
    sPairs = [1,1,2,2]
    sPairsMax = max(sPairs)
    sPairWeights = [np.array([1]), np.array([1]), np.array([1,0]), np.array([4,1])]
    snPairWeights = [np.array([1]), np.array([1]), np.array([1,0]), np.array([1,1])]
    taus = [50, 50, 50, 100]
    alphas = [0, 0, 0, 0]
    if mode == 0:
        taggingRates = [0, 0, 0.0008, 0.0008]
        ageingTaus = [6000, 10000, 10000, 10000]
    elif mode == 1:
        taggingRates = [0, 0, 0.00001, 0.00001]
        ageingTaus = [1e10, 1e10, 1e10, 1e10]
    elif mode == 2:
        taggingRates = [0, 0, 0.0008, 0.0008]
        ageingTaus = [1e10, 1e10, 1e10, 1e10]
    captureRates = [0, 0, 1, 1]
    noiseTaus = [0, 0, 1000, 1000]
    
    sPairHasWeights = [[False], [False], [False, True], [True,True]]
    
else:
    numDendrites = [2,2,1,2]
    sPairs = [2,2,2,3]
    sPairsMax = max(sPairs)
    sPairWeights = [np.array([1,0]), np.array([1,0]), np.array([1,0]), np.array([6,1,0])]
    snPairWeights = [np.array([1,0]), np.array([1,0]), np.array([1,0]), np.array([1,1,0])]
    taus = [50, 50, 50, 100]
    alphas = [0.9, 0.97, 0, 0]
    taggingRates = [0.3, 0.4, 0.0005, 0.0005]
    captureRates = [0.002, 0.002, 1, 1]
    noiseTaus = [800, 800, 1000, 1000]
    ageingTaus = [4000, 5000, 10000, 10000]
    
    sPairHasWeights = [[False, True], [False, True], [False, True], [True, True, False]]
    

distances = SNeurons("distances", sizes[0], numDendrites[0], sPairs[0], sPairWeights[0], snPairWeights[0], taus[0], 
                     ageingTaus[0], alphas[0], taggingRates[0], captureRates[0], noiseTaus[0], transientBlock=False,
                     dendriteThreshold=0.5)

heading = SNeurons("heading", sizes[1], numDendrites[1], sPairs[1], sPairWeights[1], snPairWeights[1], taus[1], 
                   ageingTaus[1], alphas[1], taggingRates[1], captureRates[1], noiseTaus[1], transientBlock=False,
                   dendriteThreshold=0.5)

odours = SNeurons("odours", sizes[2], numDendrites[2], sPairs[2], sPairWeights[2], snPairWeights[2], taus[2], 
                  ageingTaus[2], alphas[2], taggingRates[2], captureRates[2], noiseTaus[2], transientBlock=True,
                  dendriteThreshold=0.1)

places = SNeurons("places", sizes[3], numDendrites[3], sPairs[3], sPairWeights[3], snPairWeights[3], taus[3], 
                  ageingTaus[3], alphas[3], taggingRates[3], captureRates[3], noiseTaus[3], transientBlock=True)

if mode > 2:
    distances.enableConnections(fromGroups=[places,places], fromNeurons=[1,2], fromSPairs=[0,2], toSPair=1)
    heading.enableConnections(fromGroups=[places,places], fromNeurons=[1,2], fromSPairs=[0,2], toSPair=1)
    
distances.initializeWeights()
heading.initializeWeights()

places.enableConnections(fromGroups=[distances,heading,places], fromNeurons=[0,0,1], fromSPairs=[0,0,0], toSPair=0)
places.enableConnections(fromGroups=[odours], fromNeurons=[0], fromSPairs=[0], toSPair=1)
places.initializeWeights()

odours.enableConnections(fromGroups=[places], fromNeurons=[0], fromSPairs=[0], toSPair=1)
odours.initializeWeights()

groups = [distances, heading, odours, places]

H = [[np.zeros(sizes[groupNum])] for groupNum in range(numGroups)]
HPast = [[] for _ in range(numGroups)]
S = [[] for _ in range(numGroups)]
SN = [[] for _ in range(numGroups)]
WS = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
WSN = [[[] for _ in range(sPairs[groupNum])] for groupNum in range(numGroups)]
          
def run(steps, hInput, sInput, sPairsEnabled, resetPast, learn):
    for _ in range(steps):
        for groupNum, group in enumerate(groups):
            h, hP, s, sn = group.step(hInput[groupNum], sInput[groupNum], sPairsEnabled[groupNum], 
                                      resetPast[groupNum], learn[groupNum])
            H[groupNum].append(h)
            HPast[groupNum].append(hP+0)
            S[groupNum].append(s)
            SN[groupNum].append(sn)
            for sPair in range(sPairs[groupNum]):
                if sPairHasWeights[groupNum][sPair]:
                    WS[groupNum][sPair].append(group.shouldW[sPair]+0)
                    WSN[groupNum][sPair].append(group.shouldntW[sPair]+0)
                    

remember = [0, 0, 0, 0]
resetPast = [0, 0, 0, 1]


if mode in (0,1,2):    
    enabled = [np.array([1]), np.array([1]), np.array([1,1]), np.array([1,1])]
    disabled = [np.array([1]), np.array([1]), np.array([1,1]), np.array([0,1])]
    learn = [np.array([[0,0]]), np.array([[0,0]]), np.array([[0,0],[1,1]]), np.array([[1,1],[1,1]])]
    firstPlace = True
                
    if mode != 0:
        file = open('/home/eloy/KheperaIV/SNeurons/Places/weights.pkl', 'rb')
        places.shouldW, odours.shouldW = pickle.load(file)
        file.close()
        
    if mode == 0:
        passes = 2
        d = 3
    elif mode == 1:
        passes = 3
        d = 3
    elif mode == 2:
        passes = 4
        d = 0

        
    previousPixel = 0
    
    pSInput = np.zeros((sPairs[3], sizes[3]))

    dHInput = np.zeros((sizes[0]))
    hHInput = np.zeros((sizes[1]))
    oHInput = np.zeros((sizes[2]))
    pHInput = np.zeros((sizes[3]))
    
    for passNum in range(passes):
        for pixelNum in pixels:
                
            print("pass: ", passNum, " pixelNum: ", pixelNum)
            
            pHInput = np.zeros((sizes[3]))
            
            # inputs to odours
            oSInput = np.zeros((sPairs[2], sizes[2]))
            if mode == 0 or mode == 1 and passNum == 0:
                if pixelNum == 0:
                    oSInput[0][0] = 1
                elif pixelNum == 2:
                    oSInput[0][1] = 1
                elif pixelNum == 5:
                    oSInput[0][2] = 1
            elif mode == 2:
                if pixelNum == 0:
                    oSInput[0][0] = 1
                elif pixelNum == 2:
                    oSInput[0][1] = 1
                elif pixelNum == 5:
                    oSInput[0][3] = 1  
                                             
            # inputs to distances
            dSInput = np.zeros((sPairs[0], sizes[0]))
            if d < sizes[0]:
                dSInput[0,d] = 1
                 
            # inputs to heading
            if pixelNum == 1 and previousPixel == 0:
                hSInput = np.array([[0,1]])
            elif pixelNum == numPixels-2 and previousPixel == numPixels-1:
                hSInput = np.array([[1,0]])
        
            d += 1
                
            hInput = [dHInput, hHInput, oHInput, pHInput]
            sInput = [dSInput, hSInput, oSInput, pSInput]

            run(150, hInput, sInput, enabled, remember, learn)
            e = enabled
            if np.sum(SN[2][-1][1]) > 0.2:
                # choose a place neuron to activate
                if np.sum(H[3][-1]) > 0.1:
                    chosen = np.argmax(H[3][-1])
                else:
                    sumWeights = np.zeros(sizes[3])
                    for sPair in range(places.sPairs):    
                        sumWeights += np.sum(places.shouldW[sPair], (0,2))
                    sumWeights += np.random.uniform(0, 0.1, sizes[3])
                    chosen = np.argmin(sumWeights)
                    chosen = np.argmax(H[2][-1])
                    
                pHInput[chosen] = 3
                hInput = [dHInput, hHInput, oHInput, pHInput]
                if firstPlace:
                    e = disabled
                    firstPlace = False
                else:
                    e = enabled
                run(150, hInput, sInput, e, remember, learn)
                if mode == 0:
                    while np.sum(SN[3][-1]) > 0.1:
                        run(1, hInput, sInput, e, remember, learn)
                    
            pHInput = np.zeros((sizes[3]))
            hInput = [dHInput, hHInput, oHInput, pHInput]
            run(229, hInput, sInput, e, remember, learn)
            if np.sum(H[3][-1]) > 0.1:
                run(1, hInput, sInput, e, resetPast, learn)
                d = 0
            else:
                run(1, hInput, sInput, enabled, remember, learn)
            
            previousPixel = pixelNum
            
    if mode == 0:
        file = open('/home/eloy/KheperaIV/SNeurons/Places/weights.pkl', 'wb')
        pickle.dump((places.shouldW, odours.shouldW), file)
        file.close()
        
elif mode in (3,4):
    allEnabled = [np.array([1,1]), np.array([1,1]), np.array([1,1]), np.array([1,1,1])]
    pDisabled = [np.array([1,1]), np.array([1,1]), np.array([1,1]), np.array([0,1,1])]
    dhDisabled = [np.array([1,0]), np.array([1,0]), np.array([1,1]), np.array([1,1,1])]
    
    tag = [np.array([[0,0],[1,0]]), np.array([[0,0],[1,0]]), np.array([[0,0],[1,1]]), np.array([[1,1],[1,1]])]
    capture = [np.array([[0,0],[0,1]]), np.array([[0,0],[0,1]]), np.array([[0,0],[1,1]]), np.array([[1,1],[1,1]])]
                
    file = open('/home/eloy/KheperaIV/SNeurons/Places/weights.pkl', 'rb')
    places.shouldW, odours.shouldW = pickle.load(file)
    file.close()
    
    if mode == 4:
        file = open('/home/eloy/KheperaIV/SNeurons/Places/weights_dh.pkl', 'rb')
        distances.shouldW, heading.shouldW = pickle.load(file)
        file.close()
        passes = 1
    else:
        passes = 6
    
    placeOrder = []
    for o in range(sizes[2]-1):
        placeOrder.append(np.argmax(odours.shouldW[1][:,o,0]))
        
    d = 0

    previousPixel = 0
    places.hPast[placeOrder[0]] = 1
    
    pSInput = np.zeros((sPairs[3], sizes[3]))
    pSInput[2][placeOrder[1]] = 1

    dHInput = np.zeros((sizes[0]))
    hHInput = np.zeros((sizes[1]))
    oHInput = np.zeros((sizes[2]))
    pHInput = np.zeros((sizes[3]))
    
    for passNum in range(passes):
        for pixelNum in pixels:
                
            print("pass: ", passNum, " pixelNum: ", pixelNum)
            
            #input to places
            pHInput = np.zeros((sizes[3]))
            
            # inputs to odours
            oSInput = np.zeros((sPairs[2], sizes[2]))
            if pixelNum == 0:
                oSInput[0][0] = 1
            elif pixelNum == 2:
                oSInput[0][1] = 1
            elif pixelNum == 5:
                oSInput[0][2] = 1
                                             
            # inputs to distances
            dSInput = np.zeros((sPairs[0], sizes[0]))
            if d < sizes[0]:
                dSInput[0,d] = 1
                 
            # inputs to heading
            if pixelNum == 1 and previousPixel == 0:
                hSInput = np.zeros((sPairs[1], sizes[1]))
                hSInput[0][1] = 1 
            elif pixelNum == numPixels-2 and previousPixel == numPixels-1:
                hSInput = np.zeros((sPairs[1], sizes[1]))
                hSInput[0][0] = 1
        
            d += 1
                
            hInput = [dHInput, hHInput, oHInput, pHInput]
            sInput = [dSInput, hSInput, oSInput, pSInput]
            
            if pixelNum in (0,2,5):
                while np.sum(H[3][-1]) < 0.5:
                    run(1, hInput, sInput, allEnabled, remember, tag)
                    
                run(800, hInput, sInput, dhDisabled, remember, capture)
                run(1, hInput, sInput, dhDisabled, resetPast, capture)
                pSInput = np.zeros((sPairs[3], sizes[3]))
                if pixelNum == 0:
                    pSInput[2][placeOrder[1]] = 1
                elif pixelNum == 2 and previousPixel == 1:
                    pSInput[2][placeOrder[2]] = 1
                elif pixelNum == 2 and previousPixel == 3:
                    pSInput[2][placeOrder[0]] = 1
                elif pixelNum == 5:
                    pSInput[2][placeOrder[1]] = 1
                sInput = [dSInput, hSInput, oSInput, pSInput]
                run(100, hInput, sInput, pDisabled, remember, capture)
                d = 0
            else:
                run(300, hInput, sInput, allEnabled, remember, tag)

            previousPixel = pixelNum
    
    if mode == 3:        
        file = open('/home/eloy/KheperaIV/SNeurons/Places/weights_dh.pkl', 'wb')
        pickle.dump((distances.shouldW, heading.shouldW), file)
        file.close()


# # PLOT ACTIVITIES
 
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
 
def figsize(scale, ratio):
    fig_width_pt = 437.46112                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio        
    fig_size = [fig_width,fig_height]
    return fig_size
 
plt.rc('figure', figsize=figsize(1,1))

# define colormaps
orangeCM = LinearSegmentedColormap.from_list('orangeCM', [colors.to_rgba('C1',0), colors.to_rgba('C1',1)], N=100)
greenCM = LinearSegmentedColormap.from_list('greenCM', [colors.to_rgba('C2',0), colors.to_rgba('C2',1)], N=100)
blueCM = LinearSegmentedColormap.from_list('blueCM', [colors.to_rgba('C0',0), colors.to_rgba('C0',1)], N=100)
beigeCM = LinearSegmentedColormap.from_list('beigeCM', [colors.to_rgba('xkcd:beige',0), 
                                                        colors.to_rgba('xkcd:beige',1)], N=100)
 
fig, ax = plt.subplots(numGroups, sPairsMax, sharex='col', sharey='row', gridspec_kw = {'height_ratios':sizes})
ylabels = ["DISTANCE", "DIRECTION", "ODOUR", "PLACE"]
if mode in (3,4):
    titles = [["S-Pair 1 receiving\n unmodelled\n low-level input", 
               "S-Pair 2 receiving\n input from\n  past- and should-places"], 
              ["unmodelled\n low-level input", "input from\n past- and should-places"],
              ["unmodelled\n low-level input", "input from places"],
              ["input from distances, \ndirections and past-places", "input from odours", 
               "S-Pair 3 receiving\n unmodelled\n motivation input"]]
else:
    titles = [["S-Pair 1 receiving\n unmodelled low-level input", "input from past- and should-places"], 
              ["unmodelled low-level input", "input from past- and should-places"],
              ["unmodelled low-level input", "S-Pair 2 receiving\n input from places"],
              ["input from distances, \ndirections and past-places", "input from odours", "unmodelled\n motivation"]]
 
for groupNum in range(numGroups):
    for sPair in range(sPairs[groupNum]):  
        im0 = ax[groupNum,sPair].imshow(np.array(HPast[groupNum]).T, cmap=beigeCM, origin='lower', aspect='auto', 
                                        vmin=0, vmax=1, interpolation="none")
        im1 = ax[groupNum,sPair].imshow(np.array(H[groupNum]).T, cmap=blueCM, origin='lower', aspect='auto', 
                                        vmin=0, vmax=1, interpolation="none")
        im2 = ax[groupNum,sPair].imshow(np.array(S[groupNum])[:,sPair].T, cmap=greenCM, origin='lower', aspect='auto', 
                                        vmin=0, vmax=1, interpolation="none")
        im3 = ax[groupNum,sPair].imshow(np.array(SN[groupNum])[:,sPair].T, cmap=orangeCM, origin='lower', 
                                        aspect='auto', vmin=0, vmax=1)
 
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
             
plt.tight_layout(w_pad=1, h_pad=1)
     
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
 
cbar_ax_sn = fig.add_axes([0.87, pos_low.y0, 0.015, y_span/5])
prettyColorBar(cbar_ax_sn, orangeCM, "should not")
 
pos = pos_low.y0 + y_span/5 + y_span/15
cbar_ax_s = fig.add_axes([0.87, pos, 0.015, y_span/5])
prettyColorBar(cbar_ax_s, greenCM, "should")
 
pos += y_span/5 + y_span/15
cbar_ax_hp = fig.add_axes([0.87, pos, 0.015, y_span/5])
prettyColorBar(cbar_ax_hp, beigeCM, "past")
 
pos += y_span/5 + y_span/15
cbar_ax_h = fig.add_axes([0.87, pos, 0.015, y_span/5])
prettyColorBar(cbar_ax_h, blueCM, "head")
 
if savefigs:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/p'+str(mode)+'.pdf')


 
# PLOT WEIGHTS

numCols = 2

if mode in (3,4):
    plt.rc('figure', figsize=figsize(1,0.8))
    groups = [distances, heading]
    numRows = 2
else:
    plt.rc('figure', figsize=figsize(1,1.2))
    numRows = np.sum(np.sum(sPairHasWeights))
 
inputSizes = []
for groupNum, group in enumerate(groups):
    for sPair in range(sPairs[groupNum]):
        if sPairHasWeights[groupNum][sPair]:
            inputSizes.append(group.size*len(WS[groupNum][sPair][0]))
 

figW, axW = plt.subplots(numRows, numCols, sharex='col', sharey='row', gridspec_kw = {'height_ratios': inputSizes})

 
colNames = ['Connections onto dendrite {}'.format(dendriteNum+1) for dendriteNum in range(numCols)]
colGotName = [False for _ in range(numCols)]
rowNames = []
 
rowNum = 0
for groupNum, group in enumerate(groups):
    for sPairNum in range(sPairs[groupNum]):
        if sPairHasWeights[groupNum][sPairNum]:
            rowNames.append("Connections onto\n s-pair " + str(sPairNum+1) + " of " + group.name)
            for dendriteNum in range(numDendrites[groupNum]):
                size = sizes[groupNum]
                inputSize = group.shouldW[sPairNum].shape[0]
                wr = np.zeros((size*inputSize, len(WS[groupNum][sPairNum])))
                for n in range(size):
                    for i in range(inputSize):
                        wr[n*inputSize+i,:] = np.array(WS[groupNum][sPairNum])[:,i,n,dendriteNum]
                         
                if numRows == 1 and numCols == 1:
                    ax = axW
                elif numRows == 1:
                    ax = axW[dendriteNum]
                elif numCols == 1:
                    ax = axW[rowNum]
                else:
                    ax = axW[rowNum, dendriteNum]
                 
                if not colGotName[dendriteNum]:
                    ax.set_title(colNames[dendriteNum], size='medium')
                    colGotName[dendriteNum] = True
                if rowNum == numRows-1:
                    ax.set_xlabel("Simulation steps")
                 
                if dendriteNum == 0:
                    ax.set_ylabel(rowNames[rowNum])
                 
                cm = ax.pcolormesh(wr, vmin=0, vmax=1, rasterized=True)
                        
                axr = ax.twinx()
                axr.yaxis.set_ticks(np.arange(0.5, size*inputSize, 1))
                axr.yaxis.set_ticklabels(group.weightNames[sPairNum]*size)
                axr.yaxis.set_ticks(range(size*inputSize+1), minor=True)
                axr.yaxis.grid(which='minor')
                for tic in axr.yaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                            
                ax.yaxis.set_ticks(np.arange(0.5*inputSize,size*inputSize,inputSize))
                ax.yaxis.set_ticklabels(range(1,size+1))
                for tic in ax.yaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                ax.yaxis.set_ticks(range(0,size*inputSize,inputSize), minor=True)
                ax.yaxis.grid(which='minor', linewidth=3, color='w')
            else:
                dendriteNum += 1
                while dendriteNum < numCols:
                    if numRows == 1:
                        ax = axW[dendriteNum]
                    else:
                        ax = axW[rowNum, dendriteNum]
                    ax.axis('off')
                    dendriteNum += 1
                 
            rowNum += 1
 
plt.tight_layout(w_pad=1, h_pad=1)
 
if numCols == 1:
    ax = axW[-1]
else:
    ax = axW[-1,0]
 
pos_low = ax.get_position()
figW.subplots_adjust(right=0.82)
cbar_ax = figW.add_axes([0.90, pos_low.y0, 0.02, 0.4])
figW.colorbar(cm, cax=cbar_ax)
 
if savefigs and mode != 1:
    plt.savefig('/media/eloy/OS/Users/Eloy/OneDrive/NSC/THESIS/WRITING/TEX/Chapter3/Figs/pw'
                +str(mode)+'.pdf')
         
plt.show()




