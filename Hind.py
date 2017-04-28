#!/usr/bin/env python
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
import quotes as q
import mdtraj as mdt

# Identify which H correspond to which O
def O_neighbors(indO,
                fn_top = "water2.pdb",
                ):

    # => Load Info <= #
    top = mdt.load(fn_top).topology
    traj = mdt.load(fn_top, top=top)
    
    
    # => Indices of O/H <= #
    
    atoms = [atom for atom in top.atoms]
    nwaters = (len(atoms) - 1)/3 #Assumes only one proton defect! 
    
    indsO = [ind for ind, val in enumerate(atoms) if val.name == 'O']
    indsH = [ind for ind, val in enumerate(atoms) if val.name == 'H']
    n_O = len(indsO)
    n_H = len(indsH)
    pairs = []
    for n in indsH:
        pairs.append((n,indO))
    dists = mdt.compute_distances(traj,pairs)
    indH1 = indsH[np.argsort(dists[0])[0]]+1
    indH2 = indsH[np.argsort(dists[0])[1]]+1

    return indH1, indH2
