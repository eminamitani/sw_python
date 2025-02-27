import torch

from ase.neighborlist import *
import numpy as np
from tqdm import tqdm

'''
pytorch autograd version of sw potential
'''

class sw:
    def __init__(self,epsilon, sigma,a, bigA,B,lamb,gamma,p, q,costheta):
        self.epsilon=torch.tensor(epsilon,dtype=torch.float)
        self.sigma=torch.tensor(sigma,dtype=torch.float)
        self.a=torch.tensor(a,dtype=torch.float)
        self.bigA=torch.tensor(bigA,dtype=torch.float)
        self.B=torch.tensor(B,dtype=torch.float) 
        self.lamb=torch.tensor(lamb,dtype=torch.float) 
        self.gamma=torch.tensor(gamma,dtype=torch.float) 
        self.p=torch.tensor(p,dtype=torch.float) 
        self.q=torch.tensor(q,dtype=torch.float) 
        self.costheta=torch.tensor(costheta,dtype=torch.float)
        #cutoff for sw potential
        self.cutoff=(self.a*self.sigma).item()

    #two-body term
    def potential_ij(self, rij):
        #simple version
        #rij: target bond vector (single vector)
        #rik: possibles bond to form angle (single vector)

        Rij=torch.linalg.norm(rij)

        #two-body
        twobody_exp=torch.exp(self.sigma/(Rij-self.a*self.sigma))
        twobody_power=self.B*torch.pow(self.sigma/Rij,self.p)-torch.pow(self.sigma/Rij,self.q)
        #print(twobody_power, twobody_exp, self.parameters.bigA, self.parameters.epsilon)
        #fix double counting (\sum_i \sum_j \neq i)
        twobody=self.bigA*self.epsilon*twobody_power*twobody_exp*0.5

        return twobody
        
    def potential_ijk(self,rij, rik):

        #simple version
        #rij: target bond vector (single vector)
        #rik: possibles bond to form angle (single vector)

        Rij=torch.linalg.norm(rij)
        Rik=torch.linalg.norm(rik)
        costheta=torch.dot(rij, rik)/Rij/Rik
        
        #threebody
        threebody_exp=torch.exp((self.gamma*self.sigma/(Rij-self.a*self.sigma)) \
            + (self.gamma*self.sigma/(Rik-self.a*self.sigma)))
        threebody_power=torch.pow((costheta-self.costheta),2)
        #fix double counting
        threebody=0.5*self.lamb*self.epsilon*threebody_power*threebody_exp

        return threebody
    
    def get_pairs(self,atoms):
        i_index,j_index,rij,Sij=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        pairs=np.vstack((i_index,j_index)).T

        ij_info=[]
        for i, p in enumerate(pairs):
            ijtmp={'i_index':p[0],'j_index':p[1],'Sij':Sij[i]}
            ij_info.append(ijtmp)

        ijk_info=[]

        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            kidx=pairs[subgroup,1].squeeze(0)
            shiftik=Sij[subgroup]
            for k, shift in zip(kidx, shiftik):
                ijktmp={'i_index':pairs[i,0], 'j_index':pairs[i,1], 'k_index':k, 'Sij':Sij[i], 'Sik':shift}
                ijk_info.append(ijktmp)
        
        return ij_info, ijk_info
    
    def get_potential_energy(self,atoms):
        ij_info, ijk_info=self.get_pairs(atoms)
        pot=torch.tensor(0.0,dtype=torch.float) 
        for triplet in ijk_info:
            ri=atoms.positions[triplet['i_index']]
            rj=atoms.positions[triplet['j_index']]+np.matmul(triplet['Sij'],atoms.cell)
            rk=atoms.positions[triplet['k_index']]+np.matmul(triplet['Sik'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float)
            tensor_rj=torch.tensor(rj,dtype=torch.float)
            tensor_rk=torch.tensor(rk,dtype=torch.float)
            rij=tensor_rj-tensor_ri
            rik=tensor_rk-tensor_ri

            pot+=self.potential_ijk(rij,rik)
        
        for doublet in ij_info:
            ri=atoms.positions[doublet['i_index']]
            rj=atoms.positions[doublet['j_index']]+np.matmul(doublet['Sij'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float)
            tensor_rj=torch.tensor(rj,dtype=torch.float)
            rij=tensor_rj-tensor_ri

            pot+=self.potential_ij(rij)
        
        return pot


    
    def get_strain_deriv(self, atoms):

        ij_info, ijk_info=self.get_pairs(atoms)
        e1=torch.tensor(0.0).requires_grad_()
        e2=torch.tensor(0.0).requires_grad_()
        e3=torch.tensor(0.0).requires_grad_()
        e4=torch.tensor(0.0).requires_grad_()
        e5=torch.tensor(0.0).requires_grad_()
        e6=torch.tensor(0.0).requires_grad_()

        strain_matrix=torch.zeros((3,3))

        strain_matrix[0,0]=torch.tensor(1.0)+e1
        strain_matrix[0,1]=0.5*e6
        strain_matrix[0,2]=0.5*e5 
        strain_matrix[1,0]=0.5*e6
        strain_matrix[1,1]=torch.tensor(1.0)+e2
        strain_matrix[1,2]=0.5*e4 
        strain_matrix[2,0]=0.5*e5 
        strain_matrix[2,1]=0.5*e4 
        strain_matrix[2,2]=torch.tensor(1.0)+e3

        strain_derivative_1st=torch.zeros(6)
        strain_derivative_2nd=torch.zeros((6,6))


        #three-body term
        print('three-body part')
        for triplet in tqdm(ijk_info):
            ri=atoms.positions[triplet['i_index']]
            rj=atoms.positions[triplet['j_index']]+np.matmul(triplet['Sij'],atoms.cell)
            rk=atoms.positions[triplet['k_index']]+np.matmul(triplet['Sik'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float)
            tensor_rj=torch.tensor(rj,dtype=torch.float)
            tensor_rk=torch.tensor(rk,dtype=torch.float)
            ridash=torch.matmul(strain_matrix,tensor_ri)
            rjdash=torch.matmul(strain_matrix,tensor_rj)
            rkdash=torch.matmul(strain_matrix,tensor_rk)
            rij=rjdash-ridash 
            rik=rkdash-ridash
            pot_trip=self.potential_ijk(rij,rik)

            deriv_1st=[]
            deriv_second=np.zeros((6,6))
            strain_variable=[e1,e2,e3,e4,e5,e6]
            for i in range(6):
                deriv_1st.append(torch.autograd.grad(pot_trip,strain_variable[i], create_graph=True, retain_graph=True))

            for i,deriv in enumerate(deriv_1st):
                for j in range(6):
                    deriv_second[i,j]=torch.autograd.grad(deriv, strain_variable[j],retain_graph=True)[0].item()

            
            for i in range(6):
                strain_derivative_1st[i]+=deriv_1st[i][0].detach()
                for j in range(6):
                    strain_derivative_2nd[i,j]+=deriv_second[i,j]
            
            del deriv_1st
            del deriv_second
            del tensor_ri,tensor_rj,tensor_rk
            del ridash, rjdash,rkdash
            del pot_trip
        
        #two-body term
        print('two-body part')
        for doublet in tqdm(ij_info):
            ri=atoms.positions[doublet['i_index']]
            rj=atoms.positions[doublet['j_index']]+np.matmul(doublet['Sij'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float)
            tensor_rj=torch.tensor(rj,dtype=torch.float)
            ridash=torch.matmul(strain_matrix,tensor_ri)
            rjdash=torch.matmul(strain_matrix,tensor_rj)
            
            rij=rjdash-ridash 
            pot_pair=self.potential_ij(rij)

            deriv_1st=[]
            deriv_second=np.zeros((6,6))
            strain_variable=[e1,e2,e3,e4,e5,e6]
            
            for i in range(6):
                deriv_1st.append(torch.autograd.grad(pot_pair,strain_variable[i], create_graph=True, retain_graph=True))

            for i,deriv in enumerate(deriv_1st):
                for j in range(6):
                    deriv_second[i,j]=torch.autograd.grad(deriv, strain_variable[j],retain_graph=True)[0].item()

            
            for i in range(6):
                strain_derivative_1st[i]+=deriv_1st[i][0].detach()
                for j in range(6):
                    strain_derivative_2nd[i,j]+=deriv_second[i,j]
            
            del deriv_1st
            del deriv_second 
            del tensor_ri, tensor_rj
            del ridash, rjdash
            del pot_pair
        
        return strain_derivative_1st, strain_derivative_2nd
    
    def get_hessian(self, atoms):
        ij_info, ijk_info=self.get_pairs(atoms)
        natom=len(atoms.positions)
        hessian=torch.zeros((natom,natom,3,3))

        #three body term
        print('three-body part')
        for triplet in tqdm(ijk_info):
            ri=atoms.positions[triplet['i_index']]
            rj=atoms.positions[triplet['j_index']]+np.matmul(triplet['Sij'],atoms.cell)
            rk=atoms.positions[triplet['k_index']]+np.matmul(triplet['Sik'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float).requires_grad_()
            tensor_rj=torch.tensor(rj,dtype=torch.float).requires_grad_()
            tensor_rk=torch.tensor(rk,dtype=torch.float).requires_grad_()

            rij=tensor_rj-tensor_ri 
            rik=tensor_rk-tensor_ri
            pot_trip=self.potential_ijk(rij,rik)

            deriv_1st=[]
            deriv_1st_idx=[]
            position_variavle=[tensor_ri,tensor_rj,tensor_rk]
            position_idx=[triplet['i_index'],triplet['j_index'],triplet['k_index']]
            for i, (idx,pos)in enumerate(zip(position_idx,position_variavle)):
                deriv_1st.append(torch.autograd.grad(pot_trip,position_variavle[i], create_graph=True, retain_graph=True))
                deriv_1st_idx.append(idx)

            for i,deriv in zip(deriv_1st_idx,deriv_1st):
                for j,pos in zip(position_idx,position_variavle):
                    for dimi in range(3):
                        hessian[i,j,dimi,:]+=torch.autograd.grad(deriv[0][dimi], pos,retain_graph=True)[0].detach()
            
            
            del deriv_1st
            del tensor_ri, tensor_rj, tensor_rk
            del pot_trip
        print('two-body part')
        for doublet in tqdm(ij_info):
            ri=atoms.positions[doublet['i_index']]
            rj=atoms.positions[doublet['j_index']]+np.matmul(doublet['Sij'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float).requires_grad_()
            tensor_rj=torch.tensor(rj,dtype=torch.float).requires_grad_()

            rij=tensor_rj-tensor_ri 
            pot_pair=self.potential_ij(rij)

            deriv_1st=[]
            deriv_1st_idx=[]
            position_variavle=[tensor_ri,tensor_rj]
            position_idx=[doublet['i_index'],doublet['j_index']]
            for i, (idx,pos)in enumerate(zip(position_idx,position_variavle)):
                deriv_1st.append(torch.autograd.grad(pot_pair,position_variavle[i], create_graph=True, retain_graph=True))
                deriv_1st_idx.append(idx)


            for i,deriv in zip(deriv_1st_idx,deriv_1st):
                for j,pos in zip(position_idx,position_variavle):
                    for dimi in range(3):
                        hessian[i,j,dimi,:]+=torch.autograd.grad(deriv[0][dimi], pos,retain_graph=True)[0].detach()
            
            
            del deriv_1st
            del tensor_ri, tensor_rj
            del pot_pair
        
        return hessian
    
    def get_mixed_deriv(self,atoms):

        ij_info, ijk_info=self.get_pairs(atoms)
        e1=torch.tensor(0.0).requires_grad_()
        e2=torch.tensor(0.0).requires_grad_()
        e3=torch.tensor(0.0).requires_grad_()
        e4=torch.tensor(0.0).requires_grad_()
        e5=torch.tensor(0.0).requires_grad_()
        e6=torch.tensor(0.0).requires_grad_()

        strain_matrix=torch.zeros((3,3))

        strain_matrix[0,0]=torch.tensor(1.0)+e1
        strain_matrix[0,1]=0.5*e6
        strain_matrix[0,2]=0.5*e5 
        strain_matrix[1,0]=0.5*e6
        strain_matrix[1,1]=torch.tensor(1.0)+e2
        strain_matrix[1,2]=0.5*e4 
        strain_matrix[2,0]=0.5*e5 
        strain_matrix[2,1]=0.5*e4 
        strain_matrix[2,2]=torch.tensor(1.0)+e3

        strain_derivative_1st=torch.zeros(6)
        natom=len(atoms.positions)
        mixed_derivative=torch.zeros((6,natom,3))

        #three-body term
        print('three-body part')
        for triplet in tqdm(ijk_info):
            ri=atoms.positions[triplet['i_index']]
            rj=atoms.positions[triplet['j_index']]+np.matmul(triplet['Sij'],atoms.cell)
            rk=atoms.positions[triplet['k_index']]+np.matmul(triplet['Sik'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float).requires_grad_()
            tensor_rj=torch.tensor(rj,dtype=torch.float).requires_grad_()
            tensor_rk=torch.tensor(rk,dtype=torch.float).requires_grad_()
            ridash=torch.matmul(strain_matrix,tensor_ri)
            rjdash=torch.matmul(strain_matrix,tensor_rj)
            rkdash=torch.matmul(strain_matrix,tensor_rk)
            rij=rjdash-ridash 
            rik=rkdash-ridash
            pot_trip=self.potential_ijk(rij,rik)

            deriv_1st=[]
            strain_variable=[e1,e2,e3,e4,e5,e6]
            for i in range(6):
                deriv_1st.append(torch.autograd.grad(pot_trip,strain_variable[i], create_graph=True, retain_graph=True))

            position_variavle=[tensor_ri,tensor_rj,tensor_rk]
            position_idx=[triplet['i_index'],triplet['j_index'],triplet['k_index']]
            for i,deriv in enumerate(deriv_1st):
                for j,(pos,index) in enumerate(zip(position_variavle, position_idx)):
                    mixed_derivative[i,index,:]+=torch.autograd.grad(deriv, position_variavle[j],retain_graph=True)[0].detach()
            
            for i in range(6):
                strain_derivative_1st[i]+=deriv_1st[i][0].detach()
            
            del deriv_1st
            del tensor_ri, tensor_rj, tensor_rk
            del ridash,rjdash,rkdash

        #two-body term
        print('two-body part')
        for doublet in tqdm(ij_info):
            ri=atoms.positions[doublet['i_index']]
            rj=atoms.positions[doublet['j_index']]+np.matmul(doublet['Sij'],atoms.cell)
            tensor_ri=torch.tensor(ri,dtype=torch.float).requires_grad_()
            tensor_rj=torch.tensor(rj,dtype=torch.float).requires_grad_()
            ridash=torch.matmul(strain_matrix,tensor_ri)
            rjdash=torch.matmul(strain_matrix,tensor_rj)
            
            rij=rjdash-ridash 
            pot_pair=self.potential_ij(rij)

            deriv_1st=[]
            strain_variable=[e1,e2,e3,e4,e5,e6]
            
            for i in range(6):
                deriv_1st.append(torch.autograd.grad(pot_pair,strain_variable[i], create_graph=True, retain_graph=True))

            position_variavle=[tensor_ri,tensor_rj]
            position_idx=[doublet['i_index'],doublet['j_index']]
            for i,deriv in enumerate(deriv_1st):
                for j,(pos,index) in enumerate(zip(position_variavle, position_idx)):
                    mixed_derivative[i,index,:]+=torch.autograd.grad(deriv, position_variavle[j],retain_graph=True)[0].detach()
            
            for i in range(6):
                strain_derivative_1st[i]+=deriv_1st[i][0].detach()
            
            del deriv_1st
            del tensor_ri, tensor_rj
            del ridash, rjdash
        
        return strain_derivative_1st,mixed_derivative