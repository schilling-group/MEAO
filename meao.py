import numpy as np
from scipy.stats import ortho_group
from pyscf import gto
from pyscf.lo.iao import iao
from pyscf.lo import orth
from functools import reduce
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from MEAO.tools import *


class MEAO:
    
    def __init__(self, mol, mf, lo_coeff, norbs_in_atoms, dm_order=1):
        self.mol = mol
        self.mf = mf
        self.lo_coeff = lo_coeff
        self.norbs_in_atoms = norbs_in_atoms
        self.orbs_atomic_index = [i for i, n in enumerate(norbs_in_atoms) for _ in range(n)]
        self.dm_order = dm_order
        self.dm1_meao = None
        self.mo_meao = None
        
    def meao(self):
        orb_alive = [1]*sum(self.norbs_in_atoms)
        
        mo_iao = self.lo_coeff
        nmo = mo_iao.shape[1]
        
        mo_hf = self.mf.mo_coeff

        dm1_hf = np.zeros((len(mo_hf),len(mo_hf)))
        dm1_hf[self.mf.mo_occ>0,self.mf.mo_occ>0] = 2

        s = self.mol.intor('int1e_ovlp')
        U = reduce(np.dot, (mo_iao.T,s,mo_hf))
        dm1_iao = U @ dm1_hf @ U.T

        if self.dm_order == 1:
            U = max_coh(dm1_iao,self.orbs_atomic_index,self.norbs_in_atoms,orb_alive)
        elif self.dm_order == 2:
            dm2_hf = make_rdm2_mean_field(dm1_hf)
            dm2_iao = np.einsum('pi,qj,ijkl,rk,sl->pqrs', U, U, dm2_hf, U, U)
            U = max_coh_2rdm(dm1_iao,dm2_iao,self.orbs_atomic_index,self.norbs_in_atoms,orb_alive)
        else:
            raise NotImplementedError("Only 1-RDM and 2-RDM are implemented.")
        
        self.dm1_meao = U @ dm1_iao @ U.T
        self.mo_meao = mo_iao @ U.T

    def MI_mean_field(self):
        if self.dm1_meao is None:
            raise ValueError("Please run the meao() method first to compute the MEAO orbitals and density matrix.")
        MI = MI_mean_field(self.dm1_meao)
        return MI
    
    def get_bonds(self, threshold=0.1):
        if self.dm1_meao is None:
            raise ValueError("Please run the meao() method first to compute the MEAO orbitals and density matrix.")
        bonds, largest_cluster = get_cluster_index(self.dm1_meao, threshold)
        self.bonds = bonds
        self.mcb = largest_cluster
        return bonds
    
    def meao_mcb(self,mcb=None):
        if mcb is None:
            if not hasattr(self, 'mcb'):
                raise ValueError("Please run the get_bonds() method first to identify the largest cluster of bonded orbitals, or provide mcb manually. If get_bonds() has been run, and mcb is None, then all bonds are two-center.")
            mcb = self.mcb
        
        if self.mo_meao is None:
            raise ValueError("Please run the meao() method first to compute the MEAO orbitals.")
        else:
            orbs_nonmcb_index = [i for i in range(self.mo_meao.shape[1]) if i not in mcb]
            dm1_nonmcb = self.dm1_meao[orbs_nonmcb_index][:,orbs_nonmcb_index]
            orbs_nonmcb = self.mo_meao[:,orbs_nonmcb_index]
            w,v = np.linalg.eigh(dm1_nonmcb)
            n_closed = sum(w>=1)
            n_virtual = sum(w<1)
            n_active_elec = self.mol.nelectron - 2*n_closed
            # transform orbs_nonmcb to natural orbitals
            orbs_nonmcb = orbs_nonmcb @ v
            orbs_active = self.mo_meao[:,mcb]
            # select the first n_virtual orbitals in orbs_nonmcb as virtual orbitals
            orbs_virtual = orbs_nonmcb[:,:n_virtual]
            # select the last n_closed orbitals in orbs_nonmcb as closed orbitals
            orbs_closed = orbs_nonmcb[:,-n_closed:]
            # combine orbs_closed, orbs_active, orbs_virtual as the final orbitals
            self.mo_meao_mcb = np.hstack((orbs_closed, orbs_active, orbs_virtual))
            print('This molecule has a %d-center %d-electron bond.' % (len(mcb), int(n_active_elec)))
            return self.mo_meao_mcb, (len(mcb), int(n_active_elec))
        







