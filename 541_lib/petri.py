import numpy as np
from mc_aux import getIdxForState, knownState, numStates, getStateFromIdx, dictToSparse 
from mc_aux import addToDict, dictToFull, getIdxToState
from mc_solvers import steadyStateCTMC
import sys
import copy
import json
import pdb

# class Transition represents immediate transitions, and is base class for 
# timed transitions


class Transition():
    def __init__(self, name, inputs, input_weights, outputs, output_weights):
        self.name    = name

        # inputs is a vector of integer indices into a marking vector, identifying
        # the places that direct input arcs into this transition
        self.inputs  = inputs

        # input_weights is a vector of integer counts, with the count in index
        # position i giving the weight of the input arc from the place identified
        # in the i^{th} position of self.inputs

        self.input_weights = input_weights

        # outputs like inputs, except referring to output places for the transition

        self.outputs = outputs

        # output_weights like input_weights, except referring to output places for the transition
        self.output_weights = output_weights

    # enabled returns a boolean value indicating whether this transition is enabled to fire,
    # meaning that every input places has at least as many tokens in it as the input arc weight

    def enabled(self, s):
        for idx in range(len(self.inputs)):
            if s[ self.inputs[idx] ] < self.input_weights[idx]:
                return False
        return True


    # fire creates and returns a new marking that reflects the firing of this transition
    def fire(self, s):

        if not self.enabled(s):
            raise ValueError(f"attempting to fire transition {self.name} when it is not logically enabled to fire")

        # make a list copy of s, list type so it can be modified
        t = list(s)

        # subtract tokens from input places 
        for idx in range(len(self.inputs)):
            t[ self.inputs[ idx ] ] -= self.input_weights[ idx ]

        for idx in range(len(self.outputs)):
            t[ self.outputs[ idx ] ] += self.output_weights[ idx ]

        # return the modification, transformed into a tuple
        return tuple(t)

# class TimeTransition is a Transition with a firing rate attached.

class TimedTransition(Transition):
    def __init__(self, name, inputs, input_weights, \
                outputs, output_weights, rate):
        super().__init__(name, inputs, input_weights, \
            outputs, output_weights)

        self.rate = rate 


# validateTopo ensures the the references to places and 
# transitions are complete and consistent.
#   It does not attempt any analysis of the topology itself, e.g., connectedness

def validateTopo(topoDict):
    try:
        places       = topoDict['places']
        transitions  = topoDict['transitions']
        marking      = topoDict['marking']
        rates        = topoDict['transition_rates']

    except:        
        raise ValueError(f"Petri net description file requires attributes for lists 'places', 'transitions', 'marking', and 'transition_rates'")

    missing_names = []
    for name, mark in marking.items():
        if not name in places:
            missing_names.append(name)

    if len(missing_names) > 0:
        raise ValueError(f"Names {','.join(missing_names)} are given initial markings but are absent from the list of place names")

    trans_names = []
    problem_trans = []

    for trans in transitions:
        if 'name' not in trans:
            raise ValueError(f"Transition description is missing a name")
        else:
            trans_names.append(trans['name'])

        test_attrb = ('trans_type', 'input_places', 'input_weights', 
            'output_places', 'output_weights')

        for attrb in test_attrb:
            if attrb not in trans: 
                problem_trans.append(f"Transition {trans['name']} lacks attribute '{attrb}'")

    if len(problem_trans) > 0:  
        raiseError(",".join(problem_trans))

    if trans['trans_type'] not in ('immediate', 'timed'):
        problem_trans.append(f"Transition {trans['name']} attribute 'trans_type' not 'immediate' or 'timed'")

    for name in trans['input_places']:
        if not name in places:
            problem_trans.append(f"Transition {trans['name']} attribute includes undeclared input place name '{name}'")

    for weight in trans['input_weights']:
        if not isinstance(weight, int):
            problem_trans.append(f"Transition {trans['name']} input place weight '{weight}' is not an integer")
        elif weight < 1:
            problem_trans.append(f"Transition {trans['name']} input place weight '{weight}' is not a positive integer")

    if len(trans['input_places']) != len(trans['input_weights']):
            problem_trans.append(f"Transition {trans['name']} input places not 1-1 with input place weights")

    for name in trans['output_places']:
        if not name in places:
            problem_trans.append(f"Transition {trans['name']} attribute includes undeclared output place name '{name}'")

    for weight in trans['output_weights']:
        if not isinstance(weight, int):
            problem_trans.append(f"Transition {trans['name']} output place weight '{weight}' is not an integer")

        elif weight < 1:
            problem_trans.append(f"Transition {trans['name']} output place weight '{weight}' is not a positive integer")

    if len(trans['output_places']) != len(trans['output_weights']):
        problem_trans.append(f"Transition {trans['name']} output places not 1-1 with output place weights")

    if len(problem_trans) > 0:
        raise ValueError(".".join(problem_trans))

    problem_rate = []
    for name, rate in rates.items(): 
        if not name in trans_names:
            problem_rate.append(f"transition name '{name}' in 'transition_rates' not recognized")
        if not isinstance(rate, float):
            problem_rate.append(f"transition rate '{rate}' assigned to transition '{name}' in 'transition_rates' is not a floating point number'")

        elif not rate > 0.0:
            problem_rate.append(f"transition rate '{rate}' assigned to transition '{name}' in 'transition_rates' is not a positive floating point number'")

    if len(problem_rate) > 0:
        raise ValueError(".".join(problem_rate))

# readTopo opens the file offered, assumes it is a json file and attempts to load it into a dictionary.
# If successful it calls validateTopo, and if that is successful, returns the topology dictionary.
     
def readTopo(topoFile):
    with open(topoFile, "r") as rf:
        try:
            topoDict = json.load(rf)
        except:
            print(f"Error parsing input file {topoFile}")
            exit(1)

    validateTopo(topoDict)
    return topoDict

# fromImmediate fires every immediate transition that is enabled
# from input marking s, and returns the markings that result
# from those firings as a list

def fromImmediate(s, immTrans):
    fim = []
    for imt in immTrans:
        if imt.enabled(s):
            ns = imt.fire(s)
            fim.append(ns)

    return fim

# stableDerivatives analyzes the input marking s.  If s does not enable any
# immediate transitions, s is returned with the reaching probability of 1.0.
# If alternatively there are enabled immediate transitions, stableDerivatives
# works through and discovers all stable markings (i.e. those that do not enable
# immediate transitions) that are reachable only through firings of immediate transitions,
# and returns this list.   For markings that enable multiple immediate transitions we 
# assume that one that is chosen to fire is non-deterministically selected uniformly at
# random.   So, for a stable marking m that is reached through firing of immediate transitions
# t_1, t_2, ..., t_k, for each t_i we associate a probability equal to one over the number of
# immediate transitions enabled in the parent marking from which t_i is fired, and
# return with m the product of the probabilities associated with each fired immediate transition.
# In essence this is the probability of reaching m though that sequence of firings. 
# We use these probabilities to 'split' the transition rate out of s's parent to s among
# the stable markings that are reached by that transition.

def stableDerivatives(src, imt):
    reachableDict = {}

    reachable = []

    # see if src is itself stable    
    descs = fromImmediate(src, imt)
    if len(descs) == 0:
        # yes
        return [(src,1.0)]

    equalProb = 1.0/len(descs)
    imnbrs = []

    # put descendents of s that result from firing immediate
    # transitions into list imnbrs, and remember the probability
    # for each of being chosen to fire

    for desc in descs:
        imnbrs.append((desc, equalProb))     
  
    # imnbrs will hold markings that have not yet been
    # determined to be stable or not.  Test each, and
    # if stable add to the a dictionary of stable derivatives, along with
    # chance of the chain of transitions leading to it having
    # been chosen. N.B. it is in principle possible for different
    # sequences of immediate transitions leading to the same stable
    # marking, which means that reachableDict is structured to accumulate
    # these and on exit transform the accumulating dictionary into
    # a list
   
    while len(imnbrs) > 0:
        (s,pr) = imnbrs.pop() 

        # skip a feedback to the source, CTMCs don't have
        # transitions back to self
        #
        if s==src:
            continue

        children = fromImmediate(s, imt)
        if len(children) == 0:
            if s not in reachableDict: 
                reachableDict[s] = 0.0
            reachableDict[s] += pr
        else:
            equalProb = 1.0/len(nxtBrs)
            for child in children:
                imnbrs.append((child, pr*equalProb))

    for key,pr in reachableDict.items():
        reachable.append((key,pr))

    return reachable
   

# createTopo transforms a validated topology dictionary
# into data structures used by the analysis code.
# what is returned is a list of place names in the order they appear
# in the markings associated with states, a dictionary whose keys are
# the CTMC states and values are a description (in terms of indices into
# the list of place names) of the states associated with each,
# and the equilibrium occupancy probability vector resulting from the CTMC
# analysis

def createTopo(topoFile):
    topoDict = readTopo(topoFile)
    pDict = {}
    immTrans = []
    timedTrans = []

    rates    = topoDict['transition_rates']
    marking  = topoDict['marking']
    places   = topoDict['places']

    initialMarking  = []
     
    for idx, name in enumerate(topoDict['places']):
        pDict[name] = idx
        initialMarking.append(marking[name])

    for tDict in topoDict['transitions']:
        inputs  = []
        input_weights = []
        outputs = []
        output_weights = []

        for in_plc in tDict['input_places']:
            inputs.append(pDict[in_plc])

        for in_wgt in tDict['input_weights']:
            input_weights.append(in_wgt)

        for out_plc in tDict['output_places']:
            outputs.append(pDict[out_plc])

        for out_wgt in tDict['output_weights']:
            output_weights.append(out_wgt)

        if tDict['trans_type'] == 'immediate':
            immTrans.append( Transition(tDict['name'], inputs, input_weights, outputs, output_weights))
        else:
            rate = rates[tDict['name']]
            timedTrans.append( TimedTransition(tDict['name'], inputs, input_weights, outputs, output_weights, rate))

    return places, immTrans, timedTrans, initialMarking

def solvePI(topoFile):
    markSeq, immTrans, timedTrans, initMark = createTopo(topoFile)
 
    s = tuple(initMark)
    getIdxForState(s)

    Qdict = {}
    observed = {}

    # explore is a list of markings from which no immediate transitions are enabled.
    # Try to fire the timed transitions to create new states, and for each new state
    # expand into as-yet-undiscovered states that have no immeidate transitins enabled
    explore = [s]
    while len(explore) > 0:
        s = explore.pop()

        if s in observed:
            continue

        observed[s] = True
        srcIdx = getIdxForState(s)

        for trans in timedTrans:
            if not trans.enabled(s):
                continue

            ns = trans.fire(s)
            rate = trans.rate

            # expand ns into set of reachable stable descendents (if ns is not itself stable)
            dsts   = stableDerivatives(ns, immTrans)
            for (dst, pr) in dsts:
                dstIdx = getIdxForState(dst)
                addToDict(Qdict, srcIdx, dstIdx, pr*rate)
                explore.append(dst)

    # create diagonal element for Qdict
    for row in Qdict:
        row_sum = 0.0
        for col in Qdict[row]:
            row_sum += Qdict[row][col]
        Qdict[row][row] = -row_sum

    # create sparse representation of Q
    N = numStates()

    # transform dictionary expression of Q to sparse representation
    spQ, _ = dictToSparse(Qdict, rows=N )

    print(f"Number of states {N}")

    # solve for steady state probabilities
    pi = steadyStateCTMC(spQ)

    return markSeq, getIdxToState(), pi

if __name__ == "__main__":
    pi = solvePI(sys.argv[1])


