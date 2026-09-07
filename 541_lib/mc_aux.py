import pdb
import numpy as np
import networkx as nx
from scipy import sparse 
from scipy.sparse.csgraph import connected_components

# mc_aux has data structures and methods that support construction and analysis
# of markov chain models.   These provide a transparent way to assign
# integer values to states, and to flip back and forth between dictionary-based
# expressions of sparse matrices and scipy representations.  The representation for a
# state is user defined, but needs to be able to be used as an index into a dictionary
# (technically called 'immutable')
#
# dictionaries to remember state to index mapping
state2Idx = {}
idx2State = {}
nxtState = 0

# getNxtStateIdx is called to assign an integer to a new state, but should not
# be called directly by a user.  getIdxForState will do that and allow for the possibility
# that the state has already been given an index
def getNxtStateIdx():
    global nxtState
    n = nxtState
    nxtState += 1
    return n

def knownState(s):
    return (s in state2Idx)

# don't require knowledge of the data structure to ask how many states have been
# assigned integers so far
def numStates():
    return nxtState

# each successive call for a state not yet seen
# assigns the next integer in the state index sequence.  But
# if the state has already been given
def getIdxForState(s):
    # return one that has already been created
    if knownState(s):
        return state2Idx[s]

    # save the assignment in both directions
    stateIdx = getNxtStateIdx()
    state2Idx[s] = stateIdx
    idx2State[stateIdx] = s 

    return stateIdx

# recover state from index it is assigned
def getStateFromIdx(x):
    return idx2State[x]

def sparseToDict(sA, diag):
    dictRep = {}
    n, _ = sA.shape

    rowIdx, colIdx, V = sparse.find(sA)

    for idx in range(0, len(V)):
        i = rowIdx[idx]; j = colIdx[idx]; pr = V[idx]

        if i not in dictRep:
            dictRep[i] = {} 

        # ignore diagonal for now, will fill in later
        if i==j:
            continue
        
        dictRep[ i ][ j ] = V[idx]

    # put in the diagonal values
    for row in range(0, n):
        dictRep[ row ][ row ] = diag[row]

    return dictRep


# convert dictionary expression of sparse square matrix to scipy sparse
# with a separately called out diagonal.   Diagonals found in the dictionary
# are included in the input to the sparse array construction, but if zero may not
# appear in it.   Some Markov chain applications require a transpose, so 
# this is provided as an option for the output

def dictToSparse(Bdict, rows=None, transpose=False):
    # convert the dictionary to list representation

    if rows is None:
        # dictionary has a key for every state
        n = len(Bdict.keys()) 
    else:
        n = rows

    # initialize the diagonal information to return
    diag = np.zeros(n)

    # arrays needed for sparse construction
    I = []; J=[]; V=[]

    for row in Bdict:
        for col in Bdict[row]:
            # in case of diagonal, explicitly remember it
            if row==col and row in Bdict and col in Bdict[row]:
                diag[row] = Bdict[row][col]
            else:
                diag[row] = 0.0

            # every element in dictionary included in the sparse construction input
            I.append(row); J.append(col); V.append(Bdict[row][col])


    # to do a transpose just make the columns rows and rows columns
    if transpose:
        A = sparse.coo_array((V, (J, I)), shape=(n,n)).tocsc() 
    else:
        A = sparse.coo_array((V, (I, J)), shape=(n,n)).tocsc() 

    return A, diag


# convert dictionary expression of square matrix to scipy matrix

def dictToFull(Bdict, transpose=False):
    # size of matrix is one plus largest key
    n = np.max(list(Bdict.keys())) + 1

    # start with zeros
    A = np.zeros((n,n))

    for i in Bdict:
        for j in Bdict[i]:
            A[i,j] = Bdict[i][j]

    if transpose:
        A = A.T

    return A


# addToDict is a helper function to add an element to a dictionary of dictionaries
# to avoid visual clutter of determining whether a new element of the dictionary needs
# to be created

def addToDict(data, i, j, v):
    if i not in data:
        data[i] = {}
    data[i][j] = v

def isIrreducible(P):
    n_components = connected_components(csgraph=P, directed=True, return_labels=False)
    return (n_components == 1)

def isAperiodic(P):
    """
    Determines if a Markov chain transition matrix P is aperiodic.
    
    Parameters:
    P (numpy.ndarray or list of lists): The transition probability matrix.
    
    Returns:
    bool: True if the Markov chain is aperiodic, False otherwise.
    """
    # 1. Build a directed graph where edges represent positive transition probabilities
    G = nx.DiGraph()
    n, _ = P.shape
 
    for i in range(n):
        for j in range(n):
            if P[i,j] > 0:
                G.add_edge(i, j)
                
    # 2. Extract strongly connected components (SCCs)
    sccs = list(nx.strongly_connected_components(G))
    
    # 3. Check each component for period issues
    for scc in sccs:
        # A single node with no self-loop has no cycles (transient state); skip it
        if len(scc) == 1:
            node = list(scc)[0]
            if not G.has_edge(node, node):
                continue
        
        # Extract the subgraph for the current component
        subgraph = G.subgraph(scc)
        
        # NetworkX calculates the GCD of all cycle lengths in the SCC
        if not nx.is_aperiodic(subgraph):
            return False  # Found a periodic component
            
    return True

def getIdxToState():
    idxMap = [0]*len(idx2State)
    for (idx, state) in idx2State.items():
        idxMap[idx] = state

    return idxMap

