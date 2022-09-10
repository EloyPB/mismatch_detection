import numpy as np
from copy import deepcopy

class SlowNoise:
    """Draws a uniform noise sample in the range (lower, upper) of a given size every 'period' steps"""
    
    def __init__(self, size, period, lower, upper):
        self.size = size
        self.period = period
        self.hpper = upper
        self.lower = lower
        self.step = 0
        self.current = np.random.uniform(lower, upper, size)
        
    def get(self):
        if self.step < self.period:
            self.step += 1
        else:
            self.step = 0
            self.current = np.random.uniform(self.lower, self.hpper, self.size)
        return self.current


class delay:
    """Delay line"""
    def __init__(self, size, delay):
        self.delay = delay-1
        self.queue = np.zeros((delay, size))
        self.inputPos = 0
        
    def pushpop(self, values):
        self.queue[self.inputPos] = values

        if self.inputPos == self.delay:
            self.inputPos = 0
        else:
            self.inputPos += 1
            
        return self.queue[self.inputPos]


class SNeurons:
    """Simulates an array of mismatch detection circuits. Reuses weights for should and shouldn't.
    After initialization, call 'enableConnections' and 'initializeWeights'"""
    
    def __init__(self, name, size, numDendrites, sPairs, sPairWeights, snPairWeights, tau, ageingTau, 
                 alpha, taggingRate, captureRate, noiseTau, transientBlock=True, dendriteThreshold=2/3):
        """Initialize SNeurons instance
        
        size: number of micro-circuits
        numDendrites: number of dendrites in every should and shouldnt neuron
        sPairs: number of should/shouldnt pairs in each micro-circuit
        sPairWeights: [weight should -> head, weight shouldn't -> head] (both positive)
        tau = time constant; ageingTau = ageing time constant; alpha = smoothing factor for tags, [0,1) 
        captureRate = set rate of capture; noiseTau = period for SlowNoise; transientBlock = True/False"""
        
        self.name = name
        self.size = size
        self.numDendrites = numDendrites
        self.sPairs = sPairs
        self.sPairWeights = sPairWeights
        self.snPairWeights = snPairWeights
        self.tau = tau
        self.ageingTau = ageingTau
        self.tauFast = tau/20
        
        self.alpha = alpha
        self.taggingRate = taggingRate
        self.captureRate = captureRate
        
        self.transientBlock = transientBlock
        
        self.h = np.zeros(size)
        self.hOut = np.zeros(size)
        self.hPast = np.zeros(size)
        
        self.ages = np.zeros(size)
                
        self.should = np.zeros((sPairs, size))
        self.shouldPrevious = np.zeros((sPairs, size))
        self.shouldOut = np.zeros((sPairs, size))
        self.shouldBlockCount = np.zeros((sPairs, size))
        self.shouldTagUp = np.zeros((sPairs, size))
        self.shouldTagDown = np.zeros((sPairs, size))
        self.shouldDendrites = np.zeros((sPairs, size, numDendrites))
        
        self.shouldntOut = np.zeros((sPairs, size))
        
        self.dendriteThreshold = dendriteThreshold
        self.dendriteM = 1/(1-dendriteThreshold)
        self.dendriteN = self.dendriteM*dendriteThreshold

        self.toSPairs = []
        self.fromGroups = [None]*sPairs
        self.fromNeurons = [None]*sPairs
        self.fromSPairs = [None]*sPairs
        
        self.dendritePriorityNoise = SlowNoise((size, numDendrites), noiseTau, 0.5, 1.5)
        
        self.threshold_u = np.vectorize(self.threshold_u, otypes=[np.float])
        self.threshold_dendrite = np.vectorize(self.threshold_dendrite, otypes=[np.float])
        self.saturation = np.vectorize(self.saturation, otypes=[np.float])
        self.bpS = np.vectorize(self.bpS, otypes=[np.float])
        

    def enableConnections(self, fromGroups, fromNeurons, fromSPairs, toSPair):
        """Enable connections from SNeurons instances in 'fromGroups' to s-pair number
        'toSPair' of this SNeurons instance. Call once per s-pair! 
        
        fromNeurons' specifies whether the connections are made from the head (0), past (1), 
        should (2) or shouldnt (3) neurons
        
        'fromSPairs' specifies from which should/shouldnt pair the connections are made (in cases 0 and 1 it 
        is meaningless but some value must be provided)"""
        self.toSPairs.append(toSPair)
        self.fromGroups[toSPair] = fromGroups
        self.fromNeurons[toSPair] = fromNeurons
        self.fromSPairs[toSPair] = fromSPairs
    
        
    def initializeWeights(self):
        """Creates weight arrays and other structures"""
        
        selfAt = -1*np.ones(self.sPairs)
        inputSizes = np.zeros(self.sPairs)
        self.weightNames = [[] for _ in range(self.sPairs)]
        for sPair in self.toSPairs:
            for groupNum, inputGroup in enumerate(self.fromGroups[sPair]):
                if inputGroup is self:
                    selfAt[sPair] = inputSizes[sPair]
                inputSizes[sPair] += inputGroup.size
                for circuitNum in range(inputGroup.size):
                    if self.fromNeurons[sPair][groupNum] == 1:
                        specialNeurons = "p"
                    elif self.fromNeurons[sPair][groupNum] == 2:
                        specialNeurons = "s"
                    elif self.fromNeurons[sPair][groupNum] == 3:
                        specialNeurons = "sn"
                    else:
                        specialNeurons = ""
                    self.weightNames[sPair].append(inputGroup.name[0]+str(circuitNum+1)+specialNeurons)
        
        weights0 = [np.zeros((int(inputSizes[sPair]),self.size,self.numDendrites)) for sPair in range(self.sPairs)]
        weights1 = [np.ones((int(inputSizes[sPair]),self.size,self.numDendrites)) for sPair in range(self.sPairs)]
        self.shouldW = deepcopy(weights0)
        self.shouldWTagUp = deepcopy(weights0)
        self.shouldWTagDown = deepcopy(weights0)
        self.shouldWFitness = deepcopy(weights1)
        self.shouldntW = deepcopy(weights0)
                
        self.taggingMask = [np.ones((int(inputSizes[sPair]),self.size)) for sPair in range(self.sPairs)]
        for sPair in range(self.sPairs):
            if selfAt[sPair] >= 0:
                self.taggingMask[sPair][int(selfAt[sPair]):int(selfAt[sPair])+self.size,:] = 1-np.eye(self.size)
                                                                                          

               
    def weightUpdate(self, weights, weightsFitness, dendriteValues, desiredUp, desiredDown, 
                     somaTagUp, somaTagDown):
        
        updatesUp = np.zeros(weights.shape)
        updatesDown = np.zeros(weights.shape)
        
        availableUp = 0.5*self.captureRate*somaTagUp
        availableDown = 0.5*self.captureRate*somaTagDown

        dendritePrioritiesUp = np.sum((weights+0.05)*desiredUp, 0)*self.dendritePriorityNoise.get()
        dendritePrioritiesDown = np.sum(desiredDown/(weights+0.05), 0)*self.dendritePriorityNoise.get()

        synapsePrioritiesUp = np.exp(-3*weights)
        synapsePrioritiesDown = np.exp(3*weights)
         
        
        for circuitNum in range(self.size):
            #distribute available to branches
            dendritesGetUp = self.distributeToDendrites(availableUp[circuitNum],
                                                        np.sum(desiredUp[:,circuitNum,:], 0),
                                                        dendritePrioritiesUp[circuitNum])
            dendritesGetDown = self.distributeToDendrites(availableDown[circuitNum],
                                                          np.sum(desiredDown[:,circuitNum,:], 0),
                                                          dendritePrioritiesDown[circuitNum])
         
            #distribute available to synapses
            updatesUp[:,circuitNum,:] = self.distributeToSynapses(dendritesGetUp, desiredUp[:,circuitNum,:],
                                                                  synapsePrioritiesUp[:,circuitNum,:], 
                                                                  weights[:,circuitNum,:])

            updatesDown[:,circuitNum,:] = self.distributeToSynapses(dendritesGetDown, desiredDown[:,circuitNum,:],
                                                                    synapsePrioritiesDown[:,circuitNum,:], 
                                                                    weights[:,circuitNum,:])
            
            weights[:,circuitNum,:] = weights[:,circuitNum,:] + updatesUp[:,circuitNum,:] - updatesDown[:,circuitNum,:]
            
            # competitive normalization
            for dendriteNum in range(self.numDendrites):
                sumWeights = np.sum(weights[:,circuitNum,dendriteNum])
                while sumWeights > 1.02:
                    divisors = np.array([np.max(weightsFitness[:,circuitNum,dendriteNum])/wf if w > 0\
                                         else 0 for wf,w in zip(weightsFitness[:,circuitNum,dendriteNum],\
                                                                weights[:,circuitNum,dendriteNum])])
                    sumDivisors = np.sum(divisors)
                    weights[:,circuitNum,dendriteNum] -= np.minimum((sumWeights-1.018)*divisors/sumDivisors,
                                                                    weights[:,circuitNum,dendriteNum])
                    sumWeights = np.sum(weights[:,circuitNum,dendriteNum])

        return weights, somaTagUp-np.sum(updatesUp), somaTagDown-np.sum(updatesDown)


    def distributeToDendrites(self, remainingAvailable, remainingDesired, dendritePriorities):
        dendritesGet = np.zeros(self.numDendrites)
        while(remainingAvailable > 1e-5 and (remainingDesired > 1e-5).any()):
            dendritePriorities = np.where(remainingDesired < 1e-5, 0, dendritePriorities)
            dendritesCanTake = (remainingAvailable*dendritePriorities/np.sum(dendritePriorities))
            dendritesTake = np.minimum(dendritesCanTake, remainingDesired)
            dendritesGet += dendritesTake
            remainingDesired -= dendritesTake
            remainingAvailable -= np.sum(dendritesTake)
            
        return dendritesGet
    
    
    def distributeToSynapses(self, dendritesGet, remainingDesired, synapsePriorities, weights):
        updates = np.zeros((weights.shape))
        for dendriteNum in range(self.numDendrites):
            while(dendritesGet[dendriteNum] > 1e-5 and (remainingDesired[:,dendriteNum] > 1e-5).any()):
                synapsePriorities[:,dendriteNum] = np.where(remainingDesired[:,dendriteNum] < 1e-5, 0, 
                                                            synapsePriorities[:,dendriteNum])
                synapsesCanTake = (dendritesGet[dendriteNum]*synapsePriorities[:,dendriteNum]
                                   /np.sum(synapsePriorities[:,dendriteNum]))
                synapsesTake = np.minimum(synapsesCanTake, remainingDesired[:,dendriteNum])
                updates[:,dendriteNum] += synapsesTake
                remainingDesired[:,dendriteNum] -= synapsesTake
                dendritesGet[dendriteNum] -= np.sum(synapsesTake)
                                     
        return updates

        
    def step(self, uInput, sInput, sPairsEnabled, resetPast, learn):
        """Simulates one time step of the micro-circuits
        
        uInput = direct input to the head neurons (use negative values to switch them off)
        sInput = direct input to should/shouldnt neurons (array of dims: [sPairs, size])"""
        
        self.h += (1/self.tau)*(-self.h + self.hOut + np.dot(self.sPairWeights, self.shouldOut)
                                -np.dot(self.snPairWeights, self.shouldntOut) + uInput)
        self.hOut = self.threshold_u(self.h)
        
        if resetPast:
            self.hPast = deepcopy(self.hOut)
        
        self.ages += self.hOut

        # sum inputs to should and shouldnt
        inputValues = [None]*self.sPairs
        for sPair in self.toSPairs:
            inputValuesToSPair = []
            for groupNum, inputGroup in enumerate(self.fromGroups[sPair]):
                inputValuesToSPair = np.append(inputValuesToSPair,
                                               self.get_values(inputGroup, self.fromNeurons[sPair][groupNum],
                                                              self.fromSPairs[sPair][groupNum]))
            inputValues[sPair] = inputValuesToSPair
            
            self.shouldDendrites[sPair] = np.tensordot(inputValues[sPair], self.shouldW[sPair], 1)
            inputValues[sPair] = np.transpose(inputValues[sPair][np.newaxis])
                    
        self.shouldDendritesOut = self.threshold_dendrite(self.shouldDendrites)
        
        shouldInput = np.sum(self.shouldDendritesOut, 2) + sInput

        self.should += (1/self.tauFast*(-self.should + (- self.hOut + shouldInput)
                                        *sPairsEnabled.reshape(self.sPairs,1)))
        self.shouldOut = self.saturation(self.should)

        self.shouldntOut = self.saturation(-self.should)


        # LEARN                 
        if (learn[:,0] > 0).any():
            
            flexibilities = 10*np.exp(-self.ages/self.ageingTau)+1
            learningRate = (self.taggingRate*self.captureRate*flexibilities).reshape(1,self.size,1)

            if self.transientBlock:
                thres = 0.02
                self.shouldBlockCount = np.where(np.abs(self.should-self.shouldPrevious) > thres, 
                                                 150, np.maximum(self.shouldBlockCount-1, 0))
                shouldOpen = np.where(self.shouldBlockCount == 0, 1, 0)
                self.shouldPrevious = deepcopy(self.should)
            else:
                shouldOpen = np.ones((self.sPairs, self.size))
                
            instantShould = ((-self.hOut*np.random.uniform(0.5,1.5,(self.sPairs,self.size)) + shouldInput)
                             *sPairsEnabled.reshape(self.sPairs,1))
            shouldTagDrive = -self.taggingRate*flexibilities*instantShould*shouldOpen
            self.shouldTagUp = np.where(shouldTagDrive > 0, self.alpha*self.shouldTagUp + 
                                        (1-self.alpha)*shouldTagDrive, self.alpha*self.shouldTagUp)
            self.shouldTagDown = np.where(shouldTagDrive < 0, self.alpha*self.shouldTagDown -
                                          (1-self.alpha)*shouldTagDrive, self.alpha*self.shouldTagDown)
        

                
        
        for sPair in self.toSPairs:
            if learn[sPair][0]:           
                # update weights' fitness
                self.shouldWFitness[sPair] += (learningRate*inputValues[sPair].reshape(len(inputValues[sPair]),1,1)
                                               *self.shouldDendritesOut[sPair])
                   
                # tagging
                rate = self.taggingMask[sPair]*self.taggingRate*flexibilities      
                rateS = rate*shouldOpen[sPair] if self.transientBlock else rate       
                shouldWTagDrive = np.dstack((rateS
                                             *(np.dot(inputValues[sPair],
                                                      -self.bpS(instantShould[sPair],
                                                                            self.shouldDendrites\
                                                                            [sPair,:,dendriteNum])[np.newaxis])
                                               +(inputValues[sPair]-1)*self.shouldDendritesOut[sPair,:,dendriteNum]))
                                            for dendriteNum in range(self.numDendrites))

                                
                self.shouldWTagUp[sPair] = np.where(shouldWTagDrive > 0, self.alpha*self.shouldWTagUp[sPair] 
                                                    + (1-self.alpha)*shouldWTagDrive, 
                                                    self.alpha*self.shouldWTagUp[sPair])
                self.shouldWTagDown[sPair] = np.where(shouldWTagDrive < 0, self.alpha*self.shouldWTagDown[sPair] 
                                                      - (1-self.alpha)*shouldWTagDrive, 
                                                      self.alpha*self.shouldWTagDown[sPair])
                self.shouldWTagDown[sPair] = np.where(self.shouldW[sPair]-self.shouldWTagDown[sPair] < 0, 
                                                      self.shouldW[sPair], self.shouldWTagDown[sPair])


            # capture/consolidation
            if learn[sPair][1]:
                                                       
                (self.shouldW[sPair], self.shouldTagUp[sPair], 
                 self.shouldTagDown[sPair]) = self.weightUpdate(self.shouldW[sPair], self.shouldWFitness[sPair],
                                                                self.shouldDendrites[sPair],
                                                                self.shouldWTagUp[sPair], 
                                                                self.shouldWTagDown[sPair], 
                                                                self.shouldTagUp[sPair],
                                                                self.shouldTagDown[sPair])
                
        return self.hOut, self.hPast, self.shouldOut, self.shouldntOut
#         return self.hOut, self.hPast, self.shouldOut, self.shouldntOut, self.should+0


    def get_values(self, fromGroup, fromNeuron, fromSPair):
        if fromNeuron == 0:
            return fromGroup.hOut
        elif fromNeuron == 1:
            return fromGroup.hPast
        elif fromNeuron == 2:
            return fromGroup.shouldOut[fromSPair]


    def threshold_u(self, value):
        """Activation function for the head neurons"""
        
        if value <= 0.5:
            return 0
        elif value <= 0.75:
            return 4*value-2
        else:
            return 1


    def threshold_dendrite(self, value):
        """Activation function for the dendrites"""
        
        if value < self.dendriteThreshold:
            return 0
        else:
            return self.dendriteM*value-self.dendriteN


    def saturation(self, value):
        """Saturates 'value' between 0 and 1"""
        
        if value < 0:
            return 0
        elif value > 1:
            return 1
        else:
            return value
        

    def bpS(self, should, dendrite):
        """Blocks the backpropagation of the "should" signal to the dendrites when the signal can be
        due to noise and the dendrite is below threshold"""
        
        if dendrite < self.dendriteThreshold and should > -0.55:
            return 0
        else:
            return should
    