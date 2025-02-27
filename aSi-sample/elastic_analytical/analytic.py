from sw_derivative.sw import *
from ase.io import read
import numpy as np
atoms = read('../final.data',format='lammps-data',style='atomic')

SW_SI_DICT={
'epsilon':2.1683,  
'sigma':2.0951,  
'a':1.80,  
'lamb':21.0,  
'gamma':1.20,  
'cos0':-0.3333333333333333,
'A':7.049556277,  
'B':0.6022245584,  
'p':4.0,  
'q':0.0,
}

sw_pot=sw(**SW_SI_DICT,dump=True)
elastic=sw_pot.get_elastic_constant(atoms)
np.save('elastic',elastic)