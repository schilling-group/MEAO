import numpy as np
from scipy.stats import ortho_group
from pyscf import gto, scf
from pyscf.lo.iao import iao
from pyscf.lo import orth
from functools import reduce
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from ..tools import *
from ..meao import MEAO

# Build the molecule of interest
mol = gto.M(atom='N 0 0 0; N 0 0 1.098',
    spin=0, verbose=0,basis='ccpvdz',unit='A',
    max_memory=1000,symmetry = False) # mem in MB

# Build the reference MINAO molecule
pmol = gto.M(atom='N 0 0 0; N 0 0 1.098',
    spin=0, verbose=0,basis='minao',unit='A',
    max_memory=1000,symmetry = False) # mem in MB

aoslices = pmol.aoslice_by_atom()
norbs_in_atoms = []
for ia in range(pmol.natm):
    norbs_in_atoms.append(aoslices[ia][3]-aoslices[ia][2])

# Run RHF calculation
mf = scf.RHF(mol)
mf.kernel()

# Construct IAOs
orbocc = mf.mo_coeff[:,mf.mo_occ>0]
c = iao(mol, orbocc)
s = mol.intor('int1e_ovlp')
mo_iao = np.dot(c, orth.lowdin(reduce(np.dot, (c.T,s,c))))

# Construct 1RDM in HF basis
mo_hf = mf.mo_coeff
dm1_hf = np.zeros((len(mo_hf),len(mo_hf)))
dm1_hf[mf.mo_occ>0,mf.mo_occ>0] = 2

# Construct 1RDM in IAO basis
U = reduce(np.dot, (mo_iao.T,s,mo_hf))
dm1_iao = U @ dm1_hf @ U.T

# Perform MEAO analysis and obtain bonding MEAOs
my_meao = MEAO(mol, mf, mo_iao, norbs_in_atoms)
my_meao.meao()
bonds = my_meao.get_bonds()
print('Bonds:', bonds)

# Run DMRG calculation in MEAO basis
mf.mo_coeff = my_meao.mo_meao
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
    ncore=0, ncas=10, g2e_symm=8)
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=10, n_elec=14, spin=0)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
ket = driver.get_random_mps(tag="GS", bond_dim=200, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=15, bond_dims=[200]*15, noises=[1e-5]*15,
    thrds=[1e-8]*15, iprint=1)
print('DMRG energy = %20.15f' % energy)

# Compute mutual information between MEAOs
odm1 = driver.get_orbital_entropies(ket, orb_type=1)
odm2 = driver.get_orbital_entropies(ket, orb_type=2)
minfo = (odm1[:, None] + odm1[None, :] - odm2) * (1 - np.identity(len(odm1)))
minfo = minfo / np.log(16)
print('MI for MEAOs:',minfo*(minfo>0.1))

# Print MI for each bond
for bond in bonds:
    print('Bond:', bond, minfo[bond[0], bond[1]])

