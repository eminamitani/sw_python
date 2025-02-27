import numpy as np
from ase.neighborlist import *
from tqdm import tqdm
from scipy import sparse
from scipy.io import mmwrite, mmread


#sample of SW potential setup
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

class sw:

    def __init__(self,epsilon,sigma,lamb,gamma,cos0,A,B,a,p,q,dump) -> None:
        
        self.epsilon=epsilon
        self.sigma=sigma
        self.lamb=lamb 
        self.gamma=gamma
        self.cos0=cos0
        self.A=A
        self.B=B
        self.a=a 
        self.p=p 
        self.q=q

        #dump intermediate values or not
        self.dump=dump

        self.cutoff=self.a*self.sigma

        '''
        comment on double counting:
        In taking summation of pair and triplet, we should be care about remove double counting.
        In this code, the derivative module just return the derivative of bare two-body and three-body interaction,
        and add factor to double counting should be done in the post processing part.
        In general, both two-body and three-body contribution should be factored by 1/2.

        '''

        '''
        unit conversion. 
        SW potential assume 'LAMMPS metal' type unit
        length : Angstrom
        mass : g/mol
        energy: eV
        velocity: Angstrom/ps
        time : ps 
        based on these unit, convert factor in verlet alrogithm is given as follows
        '''
        #same as LAMMPS
        self.boltz=8.617343e-5
        self.mv2e= 1.03642693e-4
        self.ftm2v=1.0/1.03642693e-4
        #convert to bar
        self.pfactor=1.602176634e6
        """

        i,j,k,l: i,k-> derivative j,l-> vector
        i,j and k,l 
        C11=Cxxxx C12=Cxxyy C13=Cxxzz C14=Cxxyz C15=Cxxxz C16=Cxxxy
        C21=Cyyxx C22=Cyyyy C23=Cyyzz C24=Cyyyz C25=Cyyxz C26=Cyyxy
        C31=Czzxx C32=Czzyy C33=Czzzz C34=Czzyz C35=Czzxz C36=Czzxy
        C41=Cyzxx C42=Cyzyy C43=Cyzzz C44=Cyzyz C45=Cyzxz C46=Cyzxy
        C51=Cxzxx C42=Cxzyy C43=Cxzzz C44=Cxzyz C45=Cxzxz C46=Cxzxy
        C61=Cxyxx C42=Cxyyy C43=Cxyzz C44=Cxyyz C45=Cxyxz C46=Cxyxy
        
        """
        self.idx=[[0,0,0,0],[0,0,1,1],[0,0,2,2],[0,0,1,2],[0,0,0,2],[0,0,0,1],
             [1,1,0,0],[1,1,1,1],[1,1,2,2],[1,1,1,2],[1,1,0,2],[1,1,0,1],
             [2,2,0,0],[2,2,1,1],[2,2,2,2],[2,2,1,2],[2,2,0,2],[2,2,0,1],
             [1,2,0,0],[1,2,1,1],[1,2,2,2],[1,2,1,2],[1,2,0,2],[1,2,0,1],
             [0,2,0,0],[0,2,1,1],[0,2,2,2],[0,2,1,2],[0,2,0,2],[0,2,0,1],
             [0,1,0,0],[0,1,1,1],[0,1,2,2],[0,1,1,2],[0,1,0,2],[0,1,0,1]]
    

    def two_body_pot(self,rij):
        norm_ij=np.sqrt(np.dot(rij,rij))
        expterm=np.exp(self.sigma/(norm_ij-self.a*self.sigma))
        powerterm=self.B*np.power(self.sigma/norm_ij,self.p)-np.power(self.sigma/norm_ij,self.q)

        return self.A*self.epsilon*powerterm*expterm 
    
    def two_body_diff_ij(self,rij):
        norm_ij=np.sqrt(np.dot(rij,rij))
        expterm=np.exp(self.sigma/(norm_ij-self.a*self.sigma))
        term1=self.A*self.epsilon*(-self.p*self.B*np.power(self.sigma,self.p)/np.power(norm_ij,self.p+2)*rij + self.q*np.power(self.sigma,self.q)/np.power(norm_ij,self.q+2)*rij)*expterm
        term2=-self.A*self.epsilon*(self.B*np.power(self.sigma/norm_ij,self.p)-np.power(self.sigma/norm_ij,self.q))*self.sigma*rij/norm_ij/(norm_ij-self.a*self.sigma)**2*expterm

        return term1+term2
    
    def two_body_deriv_ij_ij(self,rij):
        norm_ij=np.sqrt(np.dot(rij,rij))
        tensor_I=np.eye(3)
        tensor_ijij=np.tensordot(rij,rij,axes=0)
        expterm=np.exp(self.sigma/(norm_ij-self.a*self.sigma))
        powterm=self.B*np.power(self.sigma/norm_ij,self.p)-np.power(self.sigma/norm_ij,self.q)
        derivpowterm=-self.B*self.p*np.power(self.sigma/norm_ij,self.p)/norm_ij**2+self.q*np.power(self.sigma/norm_ij,self.q)/norm_ij**2

        term1=self.A*self.epsilon*derivpowterm*expterm*tensor_I
        term2=self.A*self.epsilon*(self.p*(self.p+2)*self.B*np.power(self.sigma/norm_ij,self.p)/norm_ij**4 \
            -self.q*(self.q+2)*np.power(self.sigma/norm_ij,self.q)/norm_ij**4)*tensor_ijij*expterm
        term3=-2*self.A*self.epsilon*self.sigma*derivpowterm/norm_ij/(norm_ij-self.a*self.sigma)**2*tensor_ijij*expterm
        term4=-self.A*self.epsilon*self.sigma*powterm*(tensor_I/norm_ij/(norm_ij-self.a*self.sigma)**2 \
            -tensor_ijij/norm_ij**3/(norm_ij-self.a*self.sigma)**2-2*tensor_ijij/norm_ij**2/(norm_ij-self.a*self.sigma)**3)*expterm
        term5=self.A*self.epsilon*self.sigma**2*powterm*tensor_ijij/norm_ij**2/(norm_ij-self.a*self.sigma)**4*expterm

        return term1+term2+term3+term4+term5
    
    def three_body_pot(self,rij,rik):
        norm_ij=np.sqrt(np.dot(rij,rij))
        norm_ik=np.sqrt(np.dot(rik,rik))
        costheta=(np.dot(rij,rik))/norm_ij/norm_ik
        term1 =self.lamb*self.epsilon*(costheta-self.cos0)**2
        term2= np.exp(self.gamma*self.sigma/(norm_ij-self.a*self.sigma)+self.gamma*self.sigma/(norm_ik-self.a*self.sigma))

        return term1*term2
    
    def three_body_deriv_ij(self,rij,rik):
        norm_ij=np.sqrt(np.dot(rij,rij))
        norm_ik=np.sqrt(np.dot(rik,rik))
        costheta=(np.dot(rij,rik))/norm_ij/norm_ik   
        factor1=np.exp(self.gamma*self.sigma/(norm_ij-self.a*self.sigma))*np.exp(self.gamma*self.sigma/(norm_ik-self.a*self.sigma))   
        sub1=rik/norm_ij/norm_ik-costheta*rij/norm_ij**2
        term1=2*self.lamb*self.epsilon*sub1*(costheta-self.cos0)*factor1 

        sub2=self.gamma*self.sigma*rij/norm_ij/(norm_ij-self.a*self.sigma)**2
        term2=-self.lamb*self.epsilon*(costheta-self.cos0)**2*sub2*factor1

        return term1+term2 
    
    def three_body_deriv_ik(self,rij,rik):
        norm_ij=np.sqrt(np.dot(rij,rij))
        norm_ik=np.sqrt(np.dot(rik,rik))
        costheta=(np.dot(rij,rik))/norm_ij/norm_ik   
        factor1=np.exp(self.gamma*self.sigma/(norm_ij-self.a*self.sigma))*np.exp(self.gamma*self.sigma/(norm_ik-self.a*self.sigma))   
        sub1=rij/norm_ij/norm_ik-costheta*rik/norm_ik**2
        term1=2*self.lamb*self.epsilon*sub1*(costheta-self.cos0)*factor1 

        sub2=self.gamma*self.sigma*rik/norm_ik/(norm_ik-self.a*self.sigma)**2
        term2=-self.lamb*self.epsilon*(costheta-self.cos0)**2*sub2*factor1

        return term1+term2 


    def three_body_deriv_ij_ij(self,rij,rik):
        tensor_ijij=np.tensordot(rij,rij,axes=0)
        tensor_ijik=np.tensordot(rij,rik,axes=0)
        tensor_ikij=np.tensordot(rik,rij,axes=0)
        tensor_ikik=np.tensordot(rik,rik,axes=0)
        tensor_I=np.eye(3)

        norm_ij=np.sqrt(np.dot(rij,rij))
        norm_ik=np.sqrt(np.dot(rik,rik))
        costheta=(np.dot(rij,rik))/norm_ij/norm_ik

        factor1=np.exp(self.gamma*self.sigma/(norm_ij-self.a*self.sigma))*np.exp(self.gamma*self.sigma/(norm_ik-self.a*self.sigma))
        factor2=(costheta-self.cos0)
        factor3=factor2**2


        coef1=2*self.lamb*self.epsilon
        coef2=2*self.lamb*self.epsilon*self.gamma*self.sigma
        coef3=self.lamb*self.epsilon*self.gamma*self.sigma
        coef4=self.lamb*self.epsilon*self.gamma**2*self.sigma**2
        
        #deriv term1
        sub1=-1/(norm_ij**3*norm_ik)*(tensor_ijik+tensor_ikij)+3*costheta/norm_ij**4*tensor_ijij-costheta/norm_ij**2*tensor_I
        term1=coef1*sub1*factor1*factor2

        #deriv term2
        sub2=1/norm_ij**2/norm_ik**2*tensor_ikik-costheta/norm_ij**3/norm_ik*(tensor_ijik+tensor_ikij)+costheta**2/norm_ij**4*tensor_ijij
        term2=coef1*sub2*factor1

        #deriv term3+term4
        sub3=-1/(norm_ij**2*norm_ik*(norm_ij-self.a*self.sigma)**2)*(tensor_ijik+tensor_ikij)+2*costheta/norm_ij**3/(norm_ij-self.a*self.sigma)**2*tensor_ijij
        term34=coef2*sub3*factor1*factor2

        #deriv term5 
        sub5=-tensor_I/norm_ij/(norm_ij-self.a*self.sigma)**2+1/norm_ij**3/(norm_ij-self.a*self.sigma)**2*tensor_ijij+2/norm_ij**2/(norm_ij-self.a*self.sigma)**3*tensor_ijij
        term5=coef3*sub5*factor3*factor1

        #deriv term6
        sub6=1/norm_ij**2/(norm_ij-self.a*self.sigma)**4*tensor_ijij
        term6=coef4*sub6*factor3*factor1

        return term1+term2+term34+term5+term6

    def three_body_deriv_ik_ij(self,rij,rik):
        tensor_ijij=np.tensordot(rij,rij,axes=0)
        tensor_ijik=np.tensordot(rij,rik,axes=0)
        tensor_ikij=np.tensordot(rik,rij,axes=0)
        tensor_ikik=np.tensordot(rik,rik,axes=0)
        tensor_I=np.eye(3)

        norm_ij=np.sqrt(np.dot(rij,rij))
        norm_ik=np.sqrt(np.dot(rik,rik))
        costheta=(np.dot(rij,rik))/norm_ij/norm_ik

        factor1=np.exp(self.gamma*self.sigma/(norm_ij-self.a*self.sigma))*np.exp(self.gamma*self.sigma/(norm_ik-self.a*self.sigma))
        factor2=(costheta-self.cos0)
        factor3=factor2**2


        coef1=2*self.lamb*self.epsilon
        coef2=2*self.lamb*self.epsilon*self.gamma*self.sigma
        coef3=self.lamb*self.epsilon*self.gamma*self.sigma
        coef4=self.lamb*self.epsilon*self.gamma**2*self.sigma**2
        
        #deriv term1
        sub1=tensor_I/norm_ij/norm_ik-1/(norm_ij*norm_ik**3)*tensor_ikik-1/(norm_ij**3*norm_ik)*tensor_ijij+costheta/norm_ij**2/norm_ik**2*tensor_ikij
        term1=coef1*sub1*factor1*factor2

        #deriv term2
        sub2=1/norm_ij**2/norm_ik**2*tensor_ijik-costheta/norm_ij**3/norm_ik*tensor_ijij-costheta/norm_ij/norm_ik**3*tensor_ikik+costheta**2/norm_ij**2/norm_ik**2*tensor_ikij
        term2=coef1*sub2*factor1

        #deriv term3
        sub3=-1/(norm_ij*norm_ik**2*(norm_ik-self.a*self.sigma)**2)*tensor_ikik+costheta/norm_ij**2/norm_ik/(norm_ik-self.a*self.sigma)**2*tensor_ikij
        term3=coef2*sub3*factor1*factor2

        #devir term4
        sub4=-1/(norm_ij**2*norm_ik*(norm_ij-self.a*self.sigma)**2)*tensor_ijij+costheta/norm_ij/norm_ik**2/(norm_ij-self.a*self.sigma)**2*tensor_ikij
        term4=coef2*sub4*factor1*factor2 

        #deriv term6
        sub6=1/norm_ij/norm_ik/(norm_ij-self.a*self.sigma)**2/(norm_ik-self.a*self.sigma)**2*tensor_ikij
        term6=coef4*sub6*factor3*factor1

        return term1+term2+term3+term4+term6

    def three_body_deriv_ik_ik(self,rij,rik):
        tensor_ijij=np.tensordot(rij,rij,axes=0)
        tensor_ijik=np.tensordot(rij,rik,axes=0)
        tensor_ikij=np.tensordot(rik,rij,axes=0)
        tensor_ikik=np.tensordot(rik,rik,axes=0)
        tensor_I=np.eye(3)

        norm_ij=np.sqrt(np.dot(rij,rij))
        norm_ik=np.sqrt(np.dot(rik,rik))
        costheta=(np.dot(rij,rik))/norm_ij/norm_ik

        factor1=np.exp(self.gamma*self.sigma/(norm_ij-self.a*self.sigma))*np.exp(self.gamma*self.sigma/(norm_ik-self.a*self.sigma))
        factor2=(costheta-self.cos0)
        factor3=factor2**2


        coef1=2*self.lamb*self.epsilon
        coef2=2*self.lamb*self.epsilon*self.gamma*self.sigma
        coef3=self.lamb*self.epsilon*self.gamma*self.sigma
        coef4=self.lamb*self.epsilon*self.gamma**2*self.sigma**2
        
        #deriv term1
        sub1=-1/(norm_ik**3*norm_ij)*(tensor_ijik+tensor_ikij)+3*costheta/norm_ik**4*tensor_ikik-costheta/norm_ik**2*tensor_I
        term1=coef1*sub1*factor1*factor2

        #deriv term2
        sub2=1/norm_ij**2/norm_ik**2*tensor_ijij-costheta/norm_ik**3/norm_ij*(tensor_ijik+tensor_ikij)+costheta**2/norm_ik**4*tensor_ikik
        term2=coef1*sub2*factor1

        #deriv term3+term4
        sub3=-1/(norm_ik**2*norm_ij*(norm_ik-self.a*self.sigma)**2)*(tensor_ikij+tensor_ijik)+2*costheta/norm_ik**3/(norm_ik-self.a*self.sigma)**2*tensor_ikik
        term34=coef2*sub3*factor1*factor2

        #deriv term5 
        sub5=-tensor_I/norm_ik/(norm_ik-self.a*self.sigma)**2+1/norm_ik**3/(norm_ik-self.a*self.sigma)**2*tensor_ikik+2/norm_ik**2/(norm_ik-self.a*self.sigma)**3*tensor_ikik
        term5=coef3*sub5*factor3*factor1

        #deriv term6
        sub6=1/norm_ik**2/(norm_ik-self.a*self.sigma)**4*tensor_ikik
        term6=coef4*sub6*factor3*factor1

        return term1+term2+term34+term5+term6
    

    def get_pairs(self, atoms):
        i_index,j_index,rijs,S=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        pairs=np.vstack((i_index,j_index)).T
        triplets=[]
        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            triplets.append(rijs[subgroup])
        
        return rijs, triplets

    def get_potential_energy(self,atoms):

        pairs, triplets=self.get_pairs(atoms)
        twobody=0.0
        three_body=0.0

        for rij, riks in zip(pairs,triplets):
            twobody=twobody+self.two_body_pot(rij)*0.5

            for rik in riks:
                three_body=three_body+self.three_body_pot(rij,rik)*0.5
    
        return twobody+three_body, twobody, three_body


    def get_pair_force(self,atoms):
        """
        evaluate pair force defined by
        .. math::
        F_{ij}=\frac{\partial U_i}{\partial \bf{r_{ij}}}-\frac{\partial U_j}{\partial \bf{r_{ji}}}
        
        Args:
            atoms(ase(Atoms)): Atoms type object to evaluate forces acting on each atom.

        """
        

        natoms=len(atoms.positions)
        cutoff=self.a*self.sigma
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=cutoff)
        pairs=np.vstack((i_index,j_index)).T

        two_body_grad=np.zeros((natoms,natoms,3))
        three_body_grad=np.zeros((natoms,natoms,3))

        #two-body term

        """
        Here we use the definition of
        ..math
        F_{ij}=\left{ \frac{\partial U_i}{\partial r_{ij} -\frac{\partial U_j}{\partial r_{ji}}\right}

        Here, 
        ..math 
        U_{i}=\frac{1}{2} \sum_{j\neq i} \psi^{(2)} (r_{ij})+\frac{1}{2} \sum_{j\neq i} \sum_{k \neq i,j}\psi^{(3)} (r_{ij, r_ik})

        For two body term,
        F^{(2)}_{ij}=\frac{1}{2}\left{ \frac{\partial \psi^{(2)}(r_{ij})}{\partial r_{ij} 
        -\frac{\partial \psi^{(2)}(r_{ij})}{\partial r_{ji}}\right}

        In two body term, $\psi^{(2)}(r_{ij})=\psi^{(2)}(r_{ij})$, thus,
        F^{(2)}_{ij}=\frac{\partial \psi^{(2)}(r_{ij})}{\partial r_{ij} 
        """

        counter=0
        for i,j  in zip (i_index,j_index):
            vector_ij=rij[counter]
            grad=self.two_body_diff_ij(vector_ij)
            two_body_grad[i,j]=grad
            counter=counter+1
        
        #three-body term
        print('pair force:')
        for ij,p in enumerate(tqdm(pairs)):
            subgroup_ij=np.where((pairs[:,0]==p[0]))[0]
            subgroup_ji=np.where((pairs[:,0]==p[1]))[0]

            ij_triplets=[]
            ji_triplets=[]

            ji_idx=np.where((pairs[:,0]==p[1]) & (pairs[:,1]==p[0]))[0]
            rji=rij[ji_idx][0]
            #print(rij[ij],rji)

            for ik_idx,sub_ik in zip(subgroup_ij,rij[subgroup_ij]):
                if (pairs[ik_idx][1] !=p[1]):
                    ij_triplets.append([rij[ij],sub_ik])
            
            for ik_idx,sub_ik in zip(subgroup_ji,rij[subgroup_ji]):
                if (pairs[ik_idx][1] !=p[0]):
                    ji_triplets.append([rji,sub_ik])
                
            
            for tri in ij_triplets:
                three_body_grad[p[0],p[1],:]=three_body_grad[p[0],p[1],:]+self.three_body_deriv_ij(tri[0],tri[1])/2
                three_body_grad[p[0],p[1],:]=three_body_grad[p[0],p[1],:]+self.three_body_deriv_ik(tri[1],tri[0])/2
            
            for tri in ji_triplets:
                three_body_grad[p[0],p[1],:]=three_body_grad[p[0],p[1],:]-self.three_body_deriv_ij(tri[0],tri[1])/2
                three_body_grad[p[0],p[1],:]=three_body_grad[p[0],p[1],:]-self.three_body_deriv_ik(tri[1],tri[0])/2
    
        return two_body_grad+three_body_grad
    
    def get_virial_pressure(self,atoms):
        natoms=len(atoms.positions)
        cutoff=self.a*self.sigma
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=cutoff)
        pairs=np.vstack((i_index,j_index)).T

        pair_force=self.get_pair_force(atoms)

        virial=0.0
        for ij,pair in enumerate(pairs):
            virial=virial+np.dot(rij[ij],pair_force[pair[0],pair[1]])
        

        return -1/2*virial*self.pfactor/3/atoms.get_volume()

    def get_virial_tensor(self,atoms):
        natoms=len(atoms.positions)
        cutoff=self.a*self.sigma
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=cutoff)
        pairs=np.vstack((i_index,j_index)).T

        pair_force=self.get_pair_force(atoms)

        virial=np.zeros((3,3))
        for ij,pair in enumerate(pairs):
            virial=virial+np.tensordot(rij[ij],pair_force[pair[0],pair[1]],axes=0)
        

        return -1/2*virial*self.pfactor/atoms.get_volume()
    
    def get_reference_pressure(self,atoms):
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        pairs=np.vstack((i_index,j_index)).T
        refT_twobody=np.zeros((3,3))
        counter=0
        for i,j  in zip (i_index,j_index):
            vector_ij=rij[counter]
            grad=self.two_body_diff_ij(vector_ij)
            refT_twobody=refT_twobody+np.tensordot(grad,vector_ij,axes=0)+np.tensordot(vector_ij,grad,axes=0)
            counter=counter+1

        refT_threebody=np.zeros((3,3))
        rik=[]
        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            rik.append(rij[subgroup])

        for ij,iks in zip(rij,rik):
            for ik in iks:
                term_ij=np.tensordot(self.three_body_deriv_ij(ij,ik),ij,axes=0)+np.tensordot(ij,self.three_body_deriv_ij(ij,ik),axes=0)
                term_ik=np.tensordot(self.three_body_deriv_ik(ij,ik),ik,axes=0)+np.tensordot(ik,self.three_body_deriv_ik(ij,ik),axes=0)

                refT_threebody=refT_threebody+term_ij+term_ik

        refT=refT_twobody+refT_threebody
        if (self.dump):
            np.save('reference_stress', refT)

        return refT
    
    def get_reference_pressure_peratom(self,atoms):
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        pairs=np.vstack((i_index,j_index)).T
        natoms=len(atoms.positions)
        refT_twobody_peratom=np.zeros((natoms,3,3))
        counter=0
        for i,j  in zip (i_index,j_index):
            vector_ij=rij[counter]
            grad=self.two_body_diff_ij(vector_ij)
            refT_twobody_peratom[i,:,:]=refT_twobody_peratom[i,:,:]+np.tensordot(grad,vector_ij,axes=0)+np.tensordot(vector_ij,grad,axes=0)
            counter=counter+1

        refT_threebody_peratom=np.zeros((natoms,3,3))
        rik=[]
        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            rik.append(rij[subgroup])

        for i,j,ij,iks in zip(i_index,j_index,rij,rik):
            for ik in iks:
                term_ij=np.tensordot(self.three_body_deriv_ij(ij,ik),ij,axes=0)+np.tensordot(ij,self.three_body_deriv_ij(ij,ik),axes=0)
                term_ik=np.tensordot(self.three_body_deriv_ik(ij,ik),ik,axes=0)+np.tensordot(ik,self.three_body_deriv_ik(ij,ik),axes=0)

                refT_threebody_peratom[i,:,:]=refT_threebody_peratom[i,:,:]+term_ij+term_ik

        refT=refT_twobody_peratom+refT_threebody_peratom
        if (self.dump):
            np.save('reference_stress_peratom', refT)

        return refT

    
    def get_second_deriv_tensor(self, atoms):
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        pairs=np.vstack((i_index,j_index)).T


        #two-body term

        counter=0
        two_body_tensor=np.zeros((3,3,3,3))
        print('second derivative two-body')
        for i,j  in zip (tqdm(i_index),j_index):
            vector_ij=rij[counter]
            grad=self.two_body_deriv_ij_ij(vector_ij)
            two_body_tensor=two_body_tensor+np.tensordot(np.tensordot(grad,vector_ij,axes=0),vector_ij,axes=0)
            counter=counter+1
        
        #three-body term
        rik=[]
        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            rik.append(rij[subgroup])
        
        three_body_tensor=np.zeros((3,3,3,3))
        print('second derivative three-body')
        for ij,iks in zip(tqdm(rij),rik):
            for ik in iks:
                term_ijij=np.tensordot(np.tensordot(self.three_body_deriv_ij_ij(ij,ik),ij,axes=0),ij,axes=0)
                term_ikij=np.tensordot(np.tensordot(self.three_body_deriv_ik_ij(ij,ik),ik,axes=0),ij,axes=0)
                term_ijik=np.tensordot(np.tensordot(self.three_body_deriv_ik_ij(ij,ik).T,ij,axes=0),ik,axes=0)
                term_ikik=np.tensordot(np.tensordot(self.three_body_deriv_ik_ik(ij,ik),ik,axes=0),ik,axes=0)

                three_body_tensor=three_body_tensor+term_ijij+term_ijik+term_ikij+term_ikik
        
        return two_body_tensor+three_body_tensor


    def get_second_deriv_tensor_peratom(self, atoms):
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        pairs=np.vstack((i_index,j_index)).T


        natoms=len(atoms.positions)
        #two-body term

        counter=0
        two_body_tensor_peratom=np.zeros((natoms,3,3,3,3))
        print('second derivative two-body')
        for i,j  in zip (tqdm(i_index),j_index):
            vector_ij=rij[counter]
            grad=self.two_body_deriv_ij_ij(vector_ij)
            two_body_tensor_peratom[i,:,:,:,:]=two_body_tensor_peratom[i,:,:,:,:]+np.tensordot(np.tensordot(grad,vector_ij,axes=0),vector_ij,axes=0)
            counter=counter+1
        
        #three-body term
        rik=[]
        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            rik.append(rij[subgroup])
        
        three_body_tensor_peratom=np.zeros((natoms,3,3,3,3))
        print('second derivative three-body')
        for i,j, ij,iks in zip(i_index,j_index, tqdm(rij),rik):
            for ik in iks:
                term_ijij=np.tensordot(np.tensordot(self.three_body_deriv_ij_ij(ij,ik),ij,axes=0),ij,axes=0)
                term_ikij=np.tensordot(np.tensordot(self.three_body_deriv_ik_ij(ij,ik),ik,axes=0),ij,axes=0)
                term_ijik=np.tensordot(np.tensordot(self.three_body_deriv_ik_ij(ij,ik).T,ij,axes=0),ik,axes=0)
                term_ikik=np.tensordot(np.tensordot(self.three_body_deriv_ik_ik(ij,ik),ik,axes=0),ik,axes=0)

                three_body_tensor_peratom[i,:,:,:,:]=three_body_tensor_peratom[i,:,:,:,:]+term_ijij+term_ijik+term_ikij+term_ikik
        
        return two_body_tensor_peratom+three_body_tensor_peratom

    #make tensor to elastic constant form

    def get_C(self,tensorC,i,j,k,l):

        term1=tensorC[i,k,j,l]
        term2=tensorC[i,l,j,k]
        term3=tensorC[j,k,i,l]
        term4=tensorC[j,l,i,k]

        return term1+term2+term3+term4

    #evaluate reference pressure effect
    def residual(self,T,i,j,k,l):
        residual=0.0
        if(i==k):
            residual=residual+T[j,l]
        if(i==l):
            residual=residual+T[j,k]
        if(j==k):
            residual=residual+T[i,l]
        if(j==l):
            residual=residual+T[i,k]

        return residual
    

    def get_born_term(self,atoms):

        deriv_term=self.get_second_deriv_tensor(atoms)
        reference_tensor=self.get_reference_pressure(atoms)

        born_term=np.zeros(36)
        born_term_without_press=np.zeros(36)
        residual_term=np.zeros(36)

        #scale GPa
        deriv_factor=1.0/8.0/atoms.get_volume()*self.pfactor/10000
        residual_factor=1.0/16.0/atoms.get_volume()*self.pfactor/10000

        for i, id in enumerate(self.idx):
            term1=self.get_C(deriv_term,id[0],id[1],id[2],id[3])
            res=self.residual(reference_tensor,id[0],id[1],id[2],id[3])

            born_term[i]=term1*deriv_factor-res*residual_factor 
        
        if (self.dump):
            np.save('born_term', born_term)
            np.save('born_term_without_press',born_term_without_press)
            np.save('residual_term',residual_term)
        
        return born_term

    def get_born_term_peratom(self,atoms):

        deriv_term=self.get_second_deriv_tensor_peratom(atoms)
        reference_tensor=self.get_reference_pressure_peratom(atoms)

        natoms=len(atoms.positions)
        born_term_peratom=np.zeros((natoms,36))
        born_term_without_press_peratom=np.zeros((natoms,36))
        residual_term_peratom=np.zeros((natoms,36))

        #scale GPa
        deriv_factor=1.0/8.0/atoms.get_volume()*self.pfactor/10000
        residual_factor=1.0/16.0/atoms.get_volume()*self.pfactor/10000

        for iatom in range(natoms):
            for i, id in enumerate(self.idx):
                term1=self.get_C(deriv_term[iatom],id[0],id[1],id[2],id[3])
                res=self.residual(reference_tensor[iatom],id[0],id[1],id[2],id[3])

                born_term_peratom[iatom,i]=term1*deriv_factor-res*residual_factor 
        
        if (self.dump):
            np.save('born_term_peratom', born_term_peratom)
            np.save('born_term_without_press',born_term_without_press_peratom)
            np.save('residual_term_peratom',residual_term_peratom)
        
        return born_term_peratom
    
    def get_hessian(self,atoms):
        """
        obtain hessian using the derivative implemented above.
        Args:
            atoms(ase(Atoms)): Atoms type object to evaluate forces acting on each atom.
        Returns:
            numpy.array(natoms,3,natoms,3). natoms--> number of atoms, 3--> xyz direction.
            By .reshape((natoms*3, natoms*3)), we get similar array format of LAMMPS dynmat file.
            (LAMMPS dynmat is scaled by mass, so, compare with dynmat, we need to rescale by mass)
        """
        
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        natoms=len(atoms.positions)
        hessian=np.zeros((natoms,3,natoms,3))

        #two-body term
        counter=0
        print('hessian two-body')
        for i,j  in zip (tqdm(i_index),j_index):
            vector_ij=rij[counter]
            deriv_klkl=self.two_body_deriv_ij_ij(vector_ij)
            idxk=i
            idxl=j

            hessian[idxk,:,idxk,:]+=0.5*(deriv_klkl)
            hessian[idxk,:,idxl,:]+=-0.5*(deriv_klkl)
            hessian[idxl,:,idxk,:]+=-0.5*(deriv_klkl)
            hessian[idxl,:,idxl,:]+=0.5*(deriv_klkl)
            counter=counter+1
        
        #three-body term
        triplets_vectors=[]
        triplets_idx=[]
        pairs=np.vstack((i_index,j_index)).T

        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            for rik, ikpair in zip(rij[subgroup],pairs[subgroup]):
                triplets_vectors.append([rij[i],rik])
                triplets_idx.append([pairs[i,0],pairs[i,1],ikpair[1]])

        print('hessian three-body')
        for tri, vector in zip(tqdm(triplets_idx), triplets_vectors):
            idxk=tri[0]
            idxl=tri[1]
            idxm=tri[2]
            deriv_klkl=self.three_body_deriv_ij_ij(vector[0],vector[1])
            deriv_klkm=self.three_body_deriv_ik_ij(vector[0],vector[1]).T
            deriv_kmkl=self.three_body_deriv_ik_ij(vector[0],vector[1])
            deriv_kmkm=self.three_body_deriv_ik_ik(vector[0],vector[1])

            hessian[idxk,:,idxk,:]+=0.5*(deriv_klkl+deriv_klkm+deriv_kmkl+deriv_kmkm)
            hessian[idxl,:,idxl,:]+=0.5*(deriv_klkl)
            hessian[idxm,:,idxm,:]+=0.5*(deriv_kmkm)
            hessian[idxk,:,idxl,:]+=-0.5*(deriv_klkl+deriv_kmkl)
            hessian[idxl,:,idxk,:]+=-0.5*(deriv_klkl+deriv_klkm)
            hessian[idxl,:,idxm,:]+=0.5*(deriv_klkm)
            hessian[idxm,:,idxl,:]+=0.5*(deriv_kmkl)
            hessian[idxk,:,idxm,:]+=-0.5*(deriv_klkm+deriv_kmkm)
            hessian[idxm,:,idxk,:]+=-0.5*(deriv_kmkl+deriv_kmkm)

        return hessian


    def get_affine_force(self,atoms):
        """
        evaluate \Xi_{i,alpha,kappa,chi} Lamaitre&Maloney J.Stat.Phys. 123, 415 (2006) (based on Eq A.25)
        Notice: In the original paper,

        ..math
        \Xi_{i,alpha,kappa,chi}=\sum_{j}\Xi_{i,j,alpha,kappa,chi}

        But, this simple summation is not appropreate in the case of many-body potential.
        To convert the derivative of r_{ij}(vector between two atoms) to r_{i} (atomic posirion),
        more complicated treatment is required as implemented in this function.

        Returns:
            np.array((Natom,3,3,3)): 3 after Natom is the direction of atom displacement, last two (3,3) is the direction of strain, 
        """
        i_index,j_index,rij,S=neighbor_list('ijDS',atoms,cutoff=self.cutoff)
        pairs=np.vstack((i_index,j_index)).T
        #two-body term
        natoms=len(atoms.positions)

        two_body_tensor=np.zeros((natoms,3,3,3))

        counter=0
        print('affine force two-body')
        for i,j  in zip(tqdm(i_index),j_index):
            vector_ij=rij[counter]

            term_klkl=np.tensordot(self.two_body_deriv_ij_ij(vector_ij),vector_ij,axes=0)

            two_body_tensor[i,:,:,:]+=-term_klkl*0.5
            two_body_tensor[j,:,:,:]+=term_klkl*0.5
            counter=counter+1


        #three-body term
        triplets_vectors=[]
        triplets_idx=[]
        pairs=np.vstack((i_index,j_index)).T

        
        for i,p in enumerate(pairs):
            subgroup=np.where((pairs[:,0]==p[0]) & (pairs[:,1]!=p[1]))
            for rik, ikpair in zip(rij[subgroup],pairs[subgroup]):
                triplets_vectors.append([rij[i],rik])
                triplets_idx.append([pairs[i,0],pairs[i,1],ikpair[1]])

        three_body_tensor=np.zeros((natoms,3,3,3))

        print('affine force three-body')
        for tri, vector in zip(tqdm(triplets_idx), triplets_vectors):

            term_klkl=np.tensordot(self.three_body_deriv_ij_ij(vector[0],vector[1]),vector[0],axes=0)
            term_klkm=np.tensordot(self.three_body_deriv_ik_ij(vector[0],vector[1]).T,vector[1],axes=0)
            term_kmkl=np.tensordot(self.three_body_deriv_ik_ij(vector[0],vector[1]),vector[0],axes=0)
            term_kmkm=np.tensordot(self.three_body_deriv_ik_ik(vector[0],vector[1]),vector[1],axes=0)

            three_body_tensor[tri[0],:,:,:]+=(-term_klkl-term_klkm-term_kmkl-term_kmkm)*0.5
            three_body_tensor[tri[1],:,:,:]+=(term_klkl+term_klkm)*0.5
            three_body_tensor[tri[2],:,:,:]+=(term_kmkl+term_kmkm)*0.5
        
        
        deriv=two_body_tensor+three_body_tensor
        xi=np.zeros((len(deriv),3,3,3))
        for ip in range(len(deriv)):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        xi[ip,i,j,k]=-0.5*(deriv[ip,i,j,k]+deriv[ip,i,k,j])

        
        return xi

    def get_nonaffine(self,atoms):
        affine_force=self.get_affine_force(atoms)
        hessian=self.get_hessian(atoms)

        reduce=len(atoms.positions)-1 

        #fix one atom to remove effect of translation (zero mode)
        hessian_sub_flat=hessian[1:,:,1:,:].reshape(reduce*3,reduce*3)
        inv=np.linalg.inv(hessian_sub_flat)
        inv_tensor=inv.reshape(reduce,3,reduce,3)
        nonaffine_disp=np.einsum("iajb,jbkx->iakx",inv_tensor,affine_force[1:,:,:])
        nonaffine_term=np.einsum('ialm,iakx->lmkx',affine_force[1:,:,:],nonaffine_disp)

        nonaffine_corr=np.zeros(36)

        for i, id in enumerate(self.idx):
            nonaffine_corr[i]=nonaffine_term[id[0],id[1],id[2],id[3]]
        
        nonaffine_GPa=nonaffine_corr/atoms.get_volume()*self.pfactor/10000

        if(self.dump):
            #hessian is huge, reduce size by using sparce_matrix
            #sparce_matrix need to be 2-dim, make flat form
            natom=len(atoms.positions)
            hessian_flat=np.reshape(hessian,(natom*3,natom*3))
            data_csc = sparse.csc_matrix(hessian_flat)
            mmwrite('hessian_sparse',data_csc)
            np.save('affine_force',affine_force)
            np.save('nonaffine_term',nonaffine_term)
            np.save('nonaffine_disp',nonaffine_disp)
            np.save('nonaffine_GPa', nonaffine_GPa)
        
        return nonaffine_GPa

    
    def get_elastic_constant(self,atoms):
        born_term=self.get_born_term(atoms)
        nonaffine_corr=self.get_nonaffine(atoms)
        
        return born_term-nonaffine_corr
    











        





        






