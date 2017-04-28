import os, sys, re, math
import numpy as np
from numpy import linalg as LA
import com 
from operator import itemgetter
import copy as c
import man as MN
import mdtraj as mdt

atomIDs = {
	1 : 'H',
	2 : 'O',
	}
revatomIDs = {
	'H' : 1,
	'O' : 2,
	}
boxAngle = 90.00

# Cuts plane out of bulk
def plane(fn_pdb,           # input file
          gap,              # thickness of layer 
          ax_perp,          # axis perpendicular to plane
          center=[],              # center from which to make cut (if set to zero, center will be set to the center of mass)
          cutoff = 0.105,   # approximate max bond len (nm)
          periodic = False, # whether simulation will be periodic (how to calculate distances and cut molecules)
          ):

        # => Loading topology and coordinates from pdb file <= #

        frame = mdt.load(fn_pdb)
	top = mdt.load(fn_pdb).topology

        atoms = [atom for atom in top.atoms]

        # => Determing O and H indices <= #

        indsO = [ind for ind,val in enumerate(atoms) if val.name == 'O']
        indsH = [ind for ind,val in enumerate(atoms) if val.name == 'H']
        n_O = len(indsO)
        n_H = len(indsH)

        # => Calculating pair-wise distances between all O and H <= #
    
	pairs = []
	for indH in indsH:
	    for indO in indsO:
	        pairs.append((indH,indO))

        distances = mdt.compute_distances(frame, pairs,opt=True,periodic=periodic).reshape((-1,n_H,n_O))

        # => Determining which Oxygens fall within the layer <= #

        if len(center) == 0:
            symbols = [val.name for val in atoms]
            center = com.com(frame.xyz[0,:],symbols)

	newatoms = []	
        indices = []
        local_indices = []

	for n, O in enumerate(indsO):
            atom = frame.xyz[0,O]
            r = math.sqrt((atom[ax_perp]-center[ax_perp])**2)
            if r < gap:
                newatoms.append(atom)
                indices.append(O)
                local_indices.append(n)

        indices = np.array(indices)
        indsO = np.array(indsO)
        indsH = np.array(indsH)

        # => Assigning Hydrogens to selected Oxygens to ensure whole water molecules <= #

        if periodic == False:
            final_indices = []
            for n,O in enumerate(local_indices):
                if np.sort(distances[0,:,O])[0] < cutoff and np.sort(distances[0,:,O])[1] < cutoff:
                    final_indices.append(indsO[O])
                    final_indices.append(indsH[np.argsort(distances[0,:,O])[0]])
                    final_indices.append(indsH[np.argsort(distances[0,:,O])[1]])
        else:
            new_indsH = indsH[np.argsort(distances[0,:,local_indices])[:,:2]].reshape(-1)
            final_indices = np.hstack((new_indsH,indices))
                
        final_indices = np.array(final_indices) 
        final_atoms = frame.xyz[0,final_indices]

	return final_atoms,final_indices
