#!/usr/bin/env python
import os, sys, re, math
import numpy as np
from numpy import linalg as LA
import com as com
from operator import itemgetter
import mdtraj as mdt
import time
from datetime import date
import copy as c

atomIDs = {
	1 : 'H',
	2 : 'O',
	}
revatomIDs = {
	'H' : 1,
	'O' : 2,
	}
boxAngle = 90.00

# => Displace atoms in the x,y,z directions <= #

def displace(atoms,
             disp=[0.0,0.0,0.0]   # must match units of atom coordinates
             ):

        new_atoms = np.copy(atoms)
	new_atoms[:,0] += disp[0]
	new_atoms[:,1] += disp[1]
	new_atoms[:,2] += disp[2]
	
	return new_atoms

# => Uniformly expand the size of the system by a factor along the given axis <= #

def expand(atoms,       # coordinates
           axis,        # axes/axis to expand
           times,       # expansion factor
           gap = 0.2,   # gap between segments (nm)
           ):
    
    natoms = len(atoms)

    final_atoms = np.copy(atoms)
    for ind,n in enumerate(axis):
        disp = [0.0,0.0,0.0]
        new_atoms = np.copy(final_atoms)
        for tind in range(times[ind]):
            disp[n] = (tind)*(np.max(new_atoms[:,n]) + gap) 
            new_set = displace(new_atoms,disp)
            if disp[n] != 0.0:
                final_atoms = np.vstack((final_atoms,new_set))
   
    return final_atoms
    

def scale_density(fn_pdb,
                  axis,                 # axis/axes  to rescale
                  target_density=0.0,       # units appropriate for dimensionality
                  axis_scale=0.0,           # instead of target density
                  ):

	# => Load Info <= #
	
	top = mdt.load(fn_pdb).topology
        atoms = [atom for atom in top.atoms]
        positions = mdt.load(fn_pdb)

        natoms = len(atoms)

        indsO = [ind for ind, val in enumerate(atoms) if val.name == 'O'] 
        indsH = [ind for ind, val in enumerate(atoms) if val.name == 'H'] 
	n_O = len(indsO)
	n_H = len(indsH)

	pairs = []
	for indH in indsH:
	    for indO in indsO:
	        pairs.append((indH,indO))

        distances = mdt.compute_distances(positions, pairs,opt=True,periodic=False).reshape((-1,n_H,n_O))

        # => Developing scaling factors <= #

        if target_density != 0.0:
            len_axis = []
            for ind,ax in enumerate(axis):
                length = np.max(positions.xyz[0,:,ax]) -  np.min(positions.xyz[0,:,ax])
                len_axis.append(length)
            
            space = np.prod(np.array(len_axis))
            init_density = space/n_O

            scale = target_density/init_density
            axis_scale = scale**(1.0/float(len(len_axis)))

        new_positions = np.copy(positions.xyz)
        for ax in axis:
            base = 0.0
            for n in range(n_O):

                O = indsO[n]
                H1 = indsH[np.argsort(distances[0,:,n])[0]]
                H2 = indsH[np.argsort(distances[0,:,n])[1]] 

                diff_O  = positions.xyz[0,O, ax] - base
                disp_O  = axis_scale*diff_O 

                diff_H = (disp_O + base)-positions.xyz[0,O,ax]
                new_positions[0,O,ax]  =  disp_O + base
                new_positions[0,H1,ax] += diff_H 
                new_positions[0,H2,ax] += diff_H 

        new_positions = new_positions[0,:,:]

        return new_positions

# => Calculates pair-wise distances between all atoms <= #

def distances(atoms):

        xi, xj = np.meshgrid(atoms[:,0],atoms[:,0],indexing='ij')
        yi, yj = np.meshgrid(atoms[:,1],atoms[:,1],indexing='ij')
        zi, zj = np.meshgrid(atoms[:,2],atoms[:,2],indexing='ij')
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dr = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        
        return dr

# => Cuts Micelle Sphere out of Bulk based on Number of Water molecules <= #
def n_sphere(fn_pdb, nwaters, centerX=0.0, centerY=0.0, centerZ=0.0):

        frame = mdt.load(fn_pdb)
	top = mdt.load(fn_pdb).topology
        print "Sphere: Loaded Topology"

        atoms = [atom for atom in top.atoms]

	natoms = len(atoms)

        indsO = [ind for ind,val in enumerate(atoms) if val.name == 'O']
        indsH = [ind for ind,val in enumerate(atoms) if val.name == 'H']
        n_O = len(indsO)
        n_H = len(indsH)
   
        print "Sphere: Assembling O,H pairs" 
        print "     O,H pairs: Meshgrid O,H pairs"
        O1,H1 = np.meshgrid(indsO,indsH)
        print "     O,H pairs: Reshaping pairs"
        pairs = np.dstack((O1,H1)).reshape(-1,2)

        print "Sphere: Computing O-H  Distances"
        distances = mdt.compute_distances(frame, pairs,opt=True,periodic=False).reshape((-1,n_H,n_O))

        print "Sphere: Calculating Radii"
        print "     Radii: dx"
        Ox = frame.xyz[0,indsO,0]
        cx = np.ones_like(indsO)*centerX
        print "     Radii: dx**2"
        radii = (Ox-cx)**2

        print "     Radii: dy"
        Oy = frame.xyz[0,indsO,1]
        cy = np.ones_like(indsO)*centerY
        print "     Radii: dy**2"
        radii += (Oy-cy)**2

        print "     Radii: dz"
        Oz = frame.xyz[0,indsO,2]
        cz = np.ones_like(indsO)*centerZ
        print "     Radii: dz**2"
        radii += (Oz-cz)**2
    
        # Skipping square root operation since only relative distance matters
        print "Sphere: Radii Complete"

        #radii = np.array(radii)
        print "Sphere: Converting O,H indices to array"
        indsO = np.array(indsO)
        indsH = np.array(indsH)

        print "Sphere: Culling water molecules"
        print "     Cull: Sorting Radii"
        indsR = np.argpartition(radii,nwaters)[:nwaters]
        print "     Cull: Assigning new water indices"
        new_indsO = indsO[indsR]
        new_indsH = indsH[np.argsort(distances[0,:,indsR])[:,:2]].reshape(-1)

        print "Sphere: Assmbling new atoms" 
        H_atoms = frame.xyz[0,new_indsH]
        O_atoms = frame.xyz[0,new_indsO]
        final_atoms = np.vstack((H_atoms,O_atoms))
        indices = np.hstack((new_indsH,new_indsO))  

        if len(H_atoms)/len(O_atoms) != 2:
            print 'Mismatch in Number of H to O!'

        print "Sphere: Complete!"

	return final_atoms,indices

def proton(fn_pdb, ind_Ostar):

        # => Loading Data <= #

        frame = mdt.load(fn_pdb)
	top = mdt.load(fn_pdb).topology

        atoms = [atom for atom in top.atoms]
	natoms = len(atoms)

        indsO = [ind for ind,val in enumerate(atoms) if val.name == 'O']
        indsH = [ind for ind,val in enumerate(atoms) if val.name == 'H']
        n_O = len(indsO)
        n_H = len(indsH)
    
        O1,H1 = np.meshgrid(indsO,indsH)
        pairs = np.dstack((O1,H1)).reshape(-1,2)

        # => Determining Nearest H to Ostar <= #

        distances = mdt.compute_distances(frame, pairs,opt=True,periodic=False).reshape((-1,n_H,n_O))

        indH1 = indsH[np.argsort(distances[0,:,ind_Ostar])[0]]
        indH2 = indsH[np.argsort(distances[0,:,ind_Ostar])[1]]
        indO = indsO[ind_Ostar]

        # => Determining Coordinates for new H <= #
	natoms = len(atoms)
	newatoms = np.zeros((natoms+1,3))

        indO = indsO[ind_Ostar]

        # Declaring variables from coordinates
	Ox = frame.xyz[0,indO,0] 
	Oy = frame.xyz[0,indO,1]
	Oz = frame.xyz[0,indO,2]

	Hx1 = frame.xyz[0,indH1, 0] 
	Hy1 = frame.xyz[0,indH1, 1]
	Hz1 = frame.xyz[0,indH1, 2]
	vH1 = np.array([Hx1-Ox, Hy1-Oy, Hz1-Oz])
	nH1 = LA.norm(vH1)

	Hx2 = frame.xyz[0,indH2, 0]
	Hy2 = frame.xyz[0,indH2, 1]
	Hz2 = frame.xyz[0,indH2, 2]
	vH2 = np.array([Hx2-Ox, Hy2-Oy, Hz2-Oz])
	nH2 = LA.norm(vH2)
	
	HHd = math.sqrt((Hx1-Hx2)**2 + (Hy1-Hy2)**2 + (Hz1-Hz2)**2)
        b = np.array(vH1+vH2)
	b[:] = [x/2 for x in b]	
	c = vH2 - vH1
	
	m = [vH1, vH2, c]
	v = np.diag(m)

	c1 = ((HHd**2) - 2*nH1)/(-2)
	c2 = ((HHd**2) - 2*nH2)/(-2)
	c3 = np.dot(b,c)
		
	Hx3 = c1/v[0]
	Hy3 = c2/v[1]
	Hz3 = c3/v[2]

	H3 = np.array([Hx3, Hy3, Hz3])
	nH3 = LA.norm(H3)
	length = ((nH1+nH2)/2)/nH3
	H3[:] = [x*length for x in H3]

	H3[0] += Ox
	H3[1] += Oy
	H3[2] += Oz

	newatoms[natoms, 0] = H3[0]
	newatoms[natoms, 1] = H3[1]
	newatoms[natoms, 2] = H3[2]
        newatoms[:natoms,:] = frame.xyz[0,:,:]

	return newatoms

def cull_incomplete(fn_pdb,
                    cutoff=0.105,
                    ):

    
        # => Loading Data <= #

        frame = mdt.load(fn_pdb)
	top = mdt.load(fn_pdb).topology

        atoms = [atom for atom in top.atoms]
	natoms = len(atoms)

        indsO = [ind for ind,val in enumerate(atoms) if val.name == 'O']
        indsH = [ind for ind,val in enumerate(atoms) if val.name == 'H']
        n_O = len(indsO)
        n_H = len(indsH)
    
	pairs = []
	for indH in indsH:
	    for indO in indsO:
	        pairs.append((indH,indO))

        # => Determining Nearest H to Ostar <= #

        distances = mdt.compute_distances(frame, pairs,opt=True,periodic=False).reshape((-1,n_H,n_O))

        local_indices = np.array(np.arange(n_O),dtype=int)
        indsO = np.array(indsO)
        indsH = np.array(indsH)
        indices = indsO[local_indices]

        final_indices = []
        for n,O in enumerate(local_indices):
            if np.sort(distances[0,:,O])[0] < cutoff and np.sort(distances[0,:,O])[1] < cutoff:
                final_indices.append(indsO[O])
                final_indices.append(indsH[np.argsort(distances[0,:,O])[0]])
                final_indices.append(indsH[np.argsort(distances[0,:,O])[1]])
                
        final_indices = np.array(final_indices) 
        final_atoms = frame.xyz[0,final_indices]

	return final_atoms,final_indices

