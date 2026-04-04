import numpy as np
from math import pi
from scipy.stats import ortho_group
from pyscf import gto, scf
from pyscf.lo.iao import iao
from pyscf.lo import orth
from functools import reduce
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from ..tools import *
from ..meao import MEAO   

rcc = 1.3970 
rch = 1.0840

coords = []
for n in range(6):
    coords.append([rcc*np.sin(pi/3*n),rcc*np.cos(pi/3*n),0])
for n in range(6):
    coords.append([rcc*np.sin(pi/3*n)+rch*np.sin(pi/3*n),rcc*np.cos(pi/3*n)+rch*np.cos(pi/3*n),0])
        
atoms = ['C']*6 + ['H']*6
xyz = ''
for n in range(len(atoms)):
    xyz = xyz + (atoms[n]+' '+str(coords[n][0])+' '+str(coords[n][1])+' '+str(coords[n][2])+';')

mol = gto.M(atom=xyz,
    spin=0, verbose=4,basis='ccpvdz',unit = 'A',
    max_memory=1000,symmetry = False) # mem in MB

pmol = gto.M(atom=xyz,
    spin=0, verbose=4,basis='minao',unit = 'A',
    max_memory=1000,symmetry = False) # mem in MB

aoslices = pmol.aoslice_by_atom()
norbs_in_atoms = []
for ia in range(pmol.natm):
    norbs_in_atoms.append(aoslices[ia][3]-aoslices[ia][2])


# Run RHF calculation and construct IAOs
mf = scf.RHF(mol)
mf.kernel()
orbocc = mf.mo_coeff[:,mf.mo_occ>0]
c = iao(mol, orbocc)
s = mol.intor('int1e_ovlp')
mo_iao = np.dot(c, orth.lowdin(reduce(np.dot, (c.T,s,c))))

# Construct MEAOs and separate them into bonds
my_meao = MEAO(mol, mf, mo_iao, norbs_in_atoms)
my_meao.meao()
bonds = my_meao.get_bonds()
print('Bonds:', bonds)

# Obtain multicenter bonding clusters
meao_mcb, active_space = my_meao.meao_mcb()
print('Active space:', active_space)

# prepare multicenter active space
mf.mo_coeff = meao_mcb

# Run CASCI with DMRG
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
    ncore=(mol.nelectron - active_space[1])//2, ncas=active_space[0], g2e_symm=8)
print('n_elec',n_elec)
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=active_space[0], n_elec=active_space[1], spin=0)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
ket = driver.get_random_mps(tag="GS", bond_dim=200, nroots=1)
bond_dims = list(range(100,801,100))
nstage = len(bond_dims)
energy = driver.dmrg(mpo, ket, n_sweeps=nstage, bond_dims=bond_dims, noises=[1e-10]*nstage,
    thrds=[1e-6]*nstage, iprint=1)

# Compute mutual information between MEAOs
odm1 = driver.get_orbital_entropies(ket, orb_type=1)
odm2 = driver.get_orbital_entropies(ket, orb_type=2)
minfo = (odm1[:, None] + odm1[None, :] - odm2) * (1 - np.identity(len(odm1)))
minfo = minfo / np.log(16)
print('MI for the active space:',minfo*(minfo>0.1))

bip_ent = driver.get_bipartite_entanglement()

# Compute genuine multipartite entanglement in the active space using minfo and bip_ent
print('GME for the active space:',min([odm1.min(),bip_ent.min()]/np.log(4)))

