import os, sys, re, math
import numpy as np
from numpy import linalg as LA
import com as CM
from operator import itemgetter
import mdtraj as mdt
import time
from datetime import date
import copy as c

# = > Rotate Molecules <= #       

#TODO: Redo this to account for new formatting 
# also, write function to separate atoms into recognizable molecules

def rotate(atoms, theta, atomsMol=3, interval=2, nMols=0):

	natoms = len(atoms)
        
	v = []
	ID = [0]*natoms
	for n in range(0,atomsN,atomsMol):
		for i in range(n, n+atomsMol,):
			ID[i]=(atomIDs[atoms[i,2]])
			v.append([
				ID[i],
				atoms[i,4],
				atoms[i,5],
				atoms[i,6]
				])
	R = np.matrix([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
	for k in range(0,atomsN,atomsMol*interval):
		molecule = v[k:k+atomsMol]
		comZ = CM.comZ(molecule)
		comY = CM.comY(molecule)

		for i in range(atomsMol):
			molecule[i][3] -= comZ
			molecule[i][2] -= comY	
			zy = np.matrix(molecule[i][2:4])
			newZY = np.array(zy*R)
			atoms[i+k,6] = newZY[0,0] + comZ
			atoms[i+k,5] = newZY[0,1] + comY
	return atoms
# => Add proton defect <= #

def proton(atoms, wateratomID, h_up=True):

	atomsN = len(atoms)
	newatoms = np.zeros((atomsN+1,10))

	if h_up==True:
		Ox = atoms[wateratomID-1, 4] #subtract one since python indexing starts at 0, lammps starts at 1
		Oy = atoms[wateratomID-1, 5]
		Oz = atoms[wateratomID-1, 6]

		Hx1 = atoms[wateratomID-2, 4] #the h index precedes the O index
		Hy1 = atoms[wateratomID-2, 5]
		Hz1 = atoms[wateratomID-2, 6]
		vH1 = np.array([Hx1-Ox, Hy1-Oy, Hz1-Oz])
		nH1 = LA.norm(vH1)
	
		Hx2 = atoms[wateratomID-3, 4]
		Hy2 = atoms[wateratomID-3, 5]
		Hz2 = atoms[wateratomID-3, 6]
		vH2 = np.array([Hx2-Ox, Hy2-Oy, Hz2-Oz])
		nH2 = LA.norm(vH2)
	
	if h_up==False:
		Ox = atoms[wateratomID-1, 4]
		Oy = atoms[wateratomID-1, 5]
		Oz = atoms[wateratomID-1, 6]

		Hx1 = atoms[wateratomID, 4]
		Hy1 = atoms[wateratomID, 5]
		Hz1 = atoms[wateratomID, 6]
		vH1 = np.array([Hx1-Ox, Hy1-Oy, Hz1-Oz])
		nH1 = LA.norm(vH1)
	
		Hx2 = atoms[wateratomID+1,4]
		Hy2 = atoms[wateratomID+1,5]
		Hz2 = atoms[wateratomID+1,6]
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

	for i in range(atomsN):
		newatoms[i,:] += atoms[i,:]
		if atoms[i,2] == 1:
			newatoms[atomsN,:] = atoms[i,:]
	newatoms[atomsN, 4] = H3[0]
	newatoms[atomsN, 5] = H3[1]
	newatoms[atomsN, 6] = H3[2]
	
	newatoms = index(newatoms)

	return newatoms
# => Cuts Micelle Sphere out of Bulk based on Radius <= #
def r_sphere(atoms, radius, centerX=0.0, centerY=0.0, centerZ=0.0, atomsMol=3):
	atomsN = len(atoms)
	v = reformat(atoms)

	if centerX == 0.0:
		centerX += CM.comX(v)
	if centerY == 0.0:
		centerY += CM.comY(v)
	if centerZ == 0.0:
		centerZ += CM.comZ(v)

	newatoms = []	
	for atom in atoms:
            r = math.sqrt((atom[4]-centerX)**2 + (atom[5]-centerY)**2 + (atom[6]-centerZ)**2) 
            if r < radius:
                newatoms.append(atom)

        newatoms = np.array(newatoms)
        dr = distances(newatoms)
        cutoff = 1.05
        new2atoms = []
        diag_test = np.eye(len(dr))*cutoff*2.0
        dr += diag_test

        for k in range(len(newatoms)):
            if newatoms[k,2] == 2.0:
                if sum(float(num) < cutoff for num in dr[:,k]) < 2:
                    pass
                else:
                    new2atoms.append(newatoms[k])
            else:
                new2atoms.append(newatoms[k])

        new2atoms = np.array(new2atoms)
        finalAtoms = []
        dr2 = distances(new2atoms)
        diag_test2 = np.eye(len(dr2))*cutoff*2.0
        dr2 += diag_test2
        for k in range(len(new2atoms)):
            if sum(float(num) < cutoff for num in dr2[:,k]) < 1:
                pass
            else:
                finalAtoms.append(new2atoms[k])
        
	finalAtoms = index(np.array(finalAtoms))
        if len(finalAtoms) == 0.0:
            print "WARNING: No atoms survived! Modify cutoff distance!"
        elif len(finalAtoms)% 3 != 0.0:
            print "WARNING: Incomplete molecule present: %d atoms total!" % (len(finalAtoms))
       
        nOs = (sum(float(num) == 1.0 for num in finalAtoms[:,2]))
        nHs = (sum(float(num) == 2.0 for num in finalAtoms[:,2]))
        if nHs*2 != nOs:
            print "MISMATCH in hydrogens and oxygens!!!!"
            print "Number of hydrogens: %d" % (nOs)
            print "Number of oxygens: %d" % (nHs)

	return finalAtoms
