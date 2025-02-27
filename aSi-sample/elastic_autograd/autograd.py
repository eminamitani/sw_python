from ase.io import read 
atoms=read('../final.data',format='lammps-data',style='atomic')
atoms.symbols='Si512'

from sw_derivative.sw_deriv import *
epsilon=2.1683  
sigma=2.0951  
a=1.80  
lamb=21.0  
gamma=1.20  
costheta=-0.333333333333
bigA=7.049556277  
B=0.6022245584  
p=4.0  
q=0.0

potential=sw(epsilon=epsilon,sigma=sigma, a=a, bigA=bigA,B=B,lamb=lamb,gamma=gamma,costheta=costheta,p=p,q=q)

print('potential_energy {0}'.format(potential.get_potential_energy(atoms)))

first, second=potential.get_strain_deriv(atoms)
np.save('second_deriv',second.numpy())

first, mixed=potential.get_mixed_deriv(atoms)
np.save('mixed',mixed.numpy())

hessian=potential.get_hessian(atoms)
np.save('hessian',hessian.numpy())

#non affine-corr
natoms=512
reduce=511

hessian_sub=hessian[1:,1:,:,:].numpy()
hessian_sub_flat=hessian_sub.transpose(0,2,1,3).reshape(reduce*3,reduce*3)

hessian_sub_inv=np.linalg.inv(hessian_sub_flat)
hessian_sub_inv_tensor=hessian_sub_inv.reshape(reduce,3,reduce,3)

affine_force=mixed[:,1:,:]
mid=np.einsum('iajb,ljb->lia',hessian_sub_inv_tensor,affine_force)

nonaffine=np.einsum('mia,lia->ml',affine_force,mid)

C=second-nonaffine

C_GPa=C/atoms.get_volume()*1.602176634e6/10000

np.save('elastic', C_GPa)