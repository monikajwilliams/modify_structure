#!/usr/bin/env python
import numpy as np

# Calculates the center of mass
# for a system of atoms

mass_data_ = {
    'H'  :  1.00782503207,
    'He' :  4.00260325415,
    'Li' :  7.016004548,
    'Be' :  9.012182201,
    'B'  :  11.009305406,
    'C'  :  12.0000000,
    'N'  :  14.00307400478,
    'O'  :  15.99491461956,
    'F'  :  18.998403224,
    'Ne' :  19.99244017542,
    'Na' :  22.98976928087,
    'Mg' :  23.985041699,
    'Al' :  26.981538627,
    'Si' :  27.97692653246,
    'P'  :  30.973761629,
    'S'  :  31.972070999,
    'Cl' :  34.968852682,
    'Ar' :  39.96238312251,
    } 


def com(atoms,symbols):

    natoms = len(atoms)
    tot_mass = 0.0
    x_mass = 0.0
    y_mass = 0.0
    z_mass = 0.0
    
    for ind,atom in enumerate(atoms):
        m = mass_data_[symbols[ind]]
        tot_mass += m
        x_mass += m*atom[0] 
        y_mass += m*atom[1]        
        z_mass += m*atom[2]
    
    x_mass /= tot_mass
    y_mass /= tot_mass
    z_mass /= tot_mass

    com = [x_mass,y_mass,z_mass]

    return com

