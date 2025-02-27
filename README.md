# strain and coordinate derivative for Stillinger-Weber potential

This is a python package to calculate the strain and coordinate derivative for the Stillinger-Weber potential. 

The script is based on the LAMMPS implementation of the Stillinger-Weber potential. 
The purpose of this package is to calculate elastic constant in silicon-like materials based on the analytical formula.

 (A. Lemaître, C. Maloney, Sum Rules for the Quasi-Static and Visco-Elastic Response of Disordered Solids at Zero Temperature. J. Stat. Phys. 123, 415–453 (2006).)


## contents
The package contains two scripts:

sw.py: main script to calculate the strain and coordinate derivative for the Stillinger-Weber potential. The derivative of the potential energy is explicitly written down in the script.

sw_deriv.py: a script to calculate the strain and coordinate derivative for the Stillinger-Weber potential. The derivative of the potential energy is calculated by using autograd module in pytorch. Very heavy and slow.

## Usage

Please see the example in `aSi-sample` directory.



This package is used to evaluate the elastic constants in the following paper:  

(1) https://arxiv.org/abs/2407.17707

