import os,sys,copy,math
import numpy as np
from math import ceil, floor
from scipy.linalg import expm
from itertools import combinations

def one_orb_rdm(ket,driver,i):
    print([i])
    rho = np.zeros((4,4))

    op = driver.expr_builder()
    op.add_term("CD", [i,i], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu = np.real(driver.expectation(ket, n_mpo, ket))
    
    op = driver.expr_builder()
    op.add_term("cd", [i,i], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nd = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDcd", [i,i,i,i], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn = np.real(driver.expectation(ket, n_mpo, ket))

    rho[0,0] = 1 - nu - nd + nn
    rho[1,1] = nu - nn
    rho[2,2] = nd - nn
    rho[3,3] = nn

    return rho

def s1_mean_field(gamma):
    s1 = np.zeros(gamma.shape[0])
    for i in range(gamma.shape[0]):
        spec = np.zeros(4)
        spec[0] = 1 - gamma[i,i]/2 - gamma[i,i]/2 + gamma[i,i]*gamma[i,i]/4
        spec[1] = gamma[i,i]/2 - gamma[i,i]*gamma[i,i]/4
        spec[2] = gamma[i,i]/2 - gamma[i,i]*gamma[i,i]/4
        spec[3] = gamma[i,i]*gamma[i,i]/4
        s = 0
        for p in spec:
            if p > 1e-6:
                s -= p * np.log(p)
        s1[i] = s
    return s1

def two_orb_rdm(ket,driver,i,j):
    rho = np.zeros((16,16))

    print([i,j])

    op = driver.expr_builder()
    op.add_term("CD", [i,i], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_id = np.real(driver.expectation(ket, n_mpo, ket))
    nd_id = np.real(nu_id)
    op = driver.expr_builder()
    op.add_term("CDcd", [i,i,i,i], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn_id = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CD", [j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    id_nu = np.real(driver.expectation(ket, n_mpo, ket))
    id_nd = np.real(id_nu)
    op = driver.expr_builder()
    op.add_term("CDcd", [j,j,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    id_nn = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDCD", [i,i,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_nu = np.real(driver.expectation(ket, n_mpo, ket))
    op = driver.expr_builder()
    op.add_term("CDcd", [i,i,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_nd = np.real(driver.expectation(ket, n_mpo, ket))
    op = driver.expr_builder()
    op.add_term("CDCDcd", [i,i,j,j,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_nn = np.real(driver.expectation(ket, n_mpo, ket))

    nd_nu = np.real(nu_nd)
    nd_nd = np.real(nu_nu)
    nd_nn = np.real(nu_nn)

    op = driver.expr_builder()
    op.add_term("CDcdCD", [i,i,i,i,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn_nu = np.real(driver.expectation(ket, n_mpo, ket))
    nn_nd = np.real(nn_nu)
    op = driver.expr_builder()
    op.add_term("CDcdCDcd", [i,i,i,i,j,j,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn_nn = np.real(driver.expectation(ket, n_mpo, ket))
    
    n0_nu = id_nu-nu_nu-nd_nu+nn_nu
    n0_nd = id_nd-nu_nd-nd_nd+nn_nd
    n0_nn = id_nn-nu_nn-nd_nn+nn_nn
    nu_n0 = nu_id-nu_nu-nu_nd+nu_nn
    nd_n0 = nd_id-nd_nu-nd_nd+nd_nn
    nn_n0 = nn_id-nn_nu-nn_nd+nn_nn
    n0_n0 = (1 - id_nu - id_nd + id_nn) - (nu_id - nu_nu - nu_nd + nu_nn) - (nd_id - nd_nu - nd_nd + nd_nn) + (nn_id - nn_nu - nn_nd + nn_nn)

    rho[0,0] = n0_n0
    rho[1,1] = n0_nu - n0_nn
    rho[2,2] = n0_nd - n0_nn
    rho[3,3] = n0_nn
    rho[4,4] = nu_n0 - nn_n0
    rho[5,5] = nu_nu - nn_nu - nu_nn + nn_nn
    rho[6,6] = nu_nd - nn_nd - nu_nn + nn_nn
    rho[7,7] = nu_nn - nn_nn
    rho[8,8] = nd_n0 - nn_n0
    rho[9,9] = nd_nu - nn_nu - nd_nn + nn_nn
    rho[10,10] = nd_nd - nn_nd - nd_nn + nn_nn
    rho[11,11] = nd_nn - nn_nn
    rho[12,12] = nn_n0 
    rho[13,13] = nn_nu - nn_nn
    rho[14,14] = nn_nd - nn_nn
    rho[15,15] = nn_nn

    # 1 pt
    op = driver.expr_builder()
    op.add_term("CD", [i,j], 1)
    op.add_term("CDcd", [i,j,i,i], -1)
    op.add_term("CDcd", [i,j,j,j], -1)
    op.add_term("CDcdcd", [i,j,i,i,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[1,4] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[4,1] = rho[1,4]
    rho[2,8] = rho[1,4]
    rho[8,2] = rho[1,4]

    # 2 pt # CD for spin up
    op = driver.expr_builder()
    op.add_term("CDcd", [i,j,j,j], 1)
    op.add_term("CDcdcd", [i,j,i,i,j,j], -1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[3,6] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[6,3] = rho[3,6]
    rho[3,9] = -rho[3,6]
    rho[9,3] = -rho[3,6]
    op = driver.expr_builder()
    op.add_term("CcdD", [i,i,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[3,12] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[12,3] = rho[3,12]
    op = driver.expr_builder()
    op.add_term("cCdD", [i,j,j,i], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[6,9] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[9,6] = rho[6,9]
    op = driver.expr_builder()
    op.add_term("cdCD", [i,j,i,i], 1)
    op.add_term("cdCDCD", [i,j,i,i,j,j], -1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[6,12] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[12,6] = rho[6,12]
    rho[9,12] = -rho[6,12]
    rho[12,9] = -rho[6,12]

    # 3 pt 
    op = driver.expr_builder()
    op.add_term("DCcdcd", [i,j,i,i,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[7,13] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[13,7] = rho[7,13]
    rho[11,14] = rho[7,13]
    rho[14,11] = rho[7,13]

    return rho


def two_orb_rdm_no_sym(ket,driver,i,j):
    rho = np.zeros((16,16))

    print([i,j])

    op = driver.expr_builder()
    op.add_term("CD", [i,i], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_id = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("cd", [i,i], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nd_id = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDcd", [i,i,i,i], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn_id = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CD", [j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    id_nu = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("cd", [j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    id_nd = np.real(driver.expectation(ket, n_mpo, ket))
    
    op = driver.expr_builder()
    op.add_term("CDcd", [j,j,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    id_nn = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDCD", [i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_nu = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("cdcd", [i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nd_nd = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDcd", [i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_nd = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("cdCD", [i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nd_nu = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDCDcd", [i,i,j,j,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nu_nn = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("cdCDcd", [i,i,j,j,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nd_nn = np.real(driver.expectation(ket, n_mpo, ket))


    op = driver.expr_builder()
    op.add_term("CDcdCD", [i,i,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn_nu = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDcdcd", [i,i,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn_nd = np.real(driver.expectation(ket, n_mpo, ket))

    op = driver.expr_builder()
    op.add_term("CDcdCDcd", [i,i,i,i,j,j,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    nn_nn = np.real(driver.expectation(ket, n_mpo, ket))
    
    n0_nu = id_nu-nu_nu-nd_nu+nn_nu
    n0_nd = id_nd-nu_nd-nd_nd+nn_nd
    n0_nn = id_nn-nu_nn-nd_nn+nn_nn
    nu_n0 = nu_id-nu_nu-nu_nd+nu_nn
    nd_n0 = nd_id-nd_nu-nd_nd+nd_nn
    nn_n0 = nn_id-nn_nu-nn_nd+nn_nn
    n0_n0 = (1 - id_nu - id_nd + id_nn) - (nu_id - nu_nu - nu_nd + nu_nn) - (nd_id - nd_nu - nd_nd + nd_nn) + (nn_id - nn_nu - nn_nd + nn_nn)

    # 0 0 
    rho[0,0] = n0_n0
    # 0 u
    rho[1,1] = n0_nu - n0_nn
    # 0 d
    rho[2,2] = n0_nd - n0_nn
    # 0 2
    rho[3,3] = n0_nn
    # u 0
    rho[4,4] = nu_n0 - nn_n0
    # u u
    rho[5,5] = nu_nu - nn_nu - nu_nn + nn_nn
    # u d
    rho[6,6] = nu_nd - nn_nd - nu_nn + nn_nn
    # u 2
    rho[7,7] = nu_nn - nn_nn
    # d 0
    rho[8,8] = nd_n0 - nn_n0
    # d u
    rho[9,9] = nd_nu - nn_nu - nd_nn + nn_nn
    # d d
    rho[10,10] = nd_nd - nn_nd - nd_nn + nn_nn
    # d 2
    rho[11,11] = nd_nn - nn_nn
    # 2 0
    rho[12,12] = nn_n0 
    # 2 u
    rho[13,13] = nn_nu - nn_nn
    # 2 d
    rho[14,14] = nn_nd - nn_nn
    # 2 2
    rho[15,15] = nn_nn

    # 1 pt
    # |0u><u0| = C_j * D_i * (1-cd)_i * (1-cd)_j
    # |u0><0u| = C_i * D_j * (1-cd)_i * (1-cd)_j < this operator is used
    op = driver.expr_builder()
    op.add_term("CD", [i,j], 1.0)
    op.add_term("CDcd", [i,j,i,i], -1.0)
    op.add_term("CDcd", [i,j,j,j], -1.0)
    op.add_term("CDcdcd", [i,j,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[1,4] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[4,1] = rho[1,4]

    # |0d><d0| = c_j * d_i * (1-CD)_i * (1-CD)_j
    # |d0><0d| = c_i * d_j * (1-CD)_i * (1-CD)_j < this operator is used
    op = driver.expr_builder()
    op.add_term("cd", [i,j], 1.0)
    op.add_term("cdCD", [i,j,i,i], -1.0)
    op.add_term("cdCD", [i,j,j,j], -1.0)
    op.add_term("cdCDCD", [i,j,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[2,8] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[8,2] = rho[2,8]

    # 2 pt # CD for spin up
    # |02><ud| = C_j * D_i * (1-cd)_i * (cd)_j
    # |ud><02| = C_i * D_j * (1-cd)_i * (cd)_j < this operator is used
    op = driver.expr_builder()
    op.add_term("CDcd", [i,j,j,j], 1.0)
    op.add_term("CDcdcd", [i,j,i,i,j,j], -1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[3,6] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[6,3] = rho[3,6]

    # |02><du| = - c_j * d_i * (1-CD)_i * CD_j
    # |du><02| = - c_i * d_j * (1-CD)_i * CD_j < this operator is used
    op = driver.expr_builder()
    op.add_term("cdCD", [i,j,j,j], -1.0)
    op.add_term("cdCDCD", [i,j,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[3,9] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[9,3] = rho[3,9]

    # |02><20| = C_j * c_j * d_i * D_i
    # |20><02| = C_i * c_i * d_j * D_j < this operator is used
    op = driver.expr_builder()
    op.add_term("CcdD", [i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[3,12] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[12,3] = rho[3,12]

    # |ud><du| = C_i * c_j * D_j * d_i
    # |du><ud| = c_i * C_j * d_j * D_i < this operator is used
    op = driver.expr_builder()
    op.add_term("cCdD", [i,j,j,i], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[6,9] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[9,6] = rho[6,9]

    # |ud><20| = c_j * d_i * (1-CD)_j * (CD)_i
    # |20><ud| = c_i * d_j * (1-CD)_j * (CD)_i < this operator is used
    op = driver.expr_builder()
    op.add_term("cdCD", [i,j,i,i], 1.0)
    op.add_term("cdCDCD", [i,j,i,i,j,j], -1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[6,12] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[12,6] = rho[6,12]

    # |du><20| = - C_j * D_i * (1-cd)_j * (cd)_i
    # |20><du| = - C_i * D_j * (1-cd)_j * (cd)_i < this operator is used
    op = driver.expr_builder()
    op.add_term("cdCD", [i,j,i,i], -1.0)
    op.add_term("cdCDCD", [i,j,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[9,12] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[12,9] = rho[9,12]

    # 3 pt 
    # |u2><2u| = d_i * c_j * (CD)_j * (CD)_i < this operator is used
    # |2u><u2| = d_j * c_i * (CD)_j * (CD)_i 
    op = driver.expr_builder()
    op.add_term("dcCDCD", [i,j,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[7,13] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[13,7] = rho[7,13]

    # |d2><2d| = D_i * C_j * (cd)_j * (cd)_i < this operator is used
    # |2d><d2| = D_j * C_i * (cd)_j * (cd)_i 
    op = driver.expr_builder()
    op.add_term("DCcdcd", [i,j,i,i,j,j], 1.0)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    rho[11,14] = np.real(driver.expectation(ket, n_mpo, ket))
    rho[14,11] = rho[11,14]

    return rho







def one_orb_rdm_mean_field(gamma,i):
    gamma = np.kron(gamma/2,np.eye(2))
    rho = np.zeros((4,4))

    rho[0,0] = (1 - gamma[2*i,2*i])*(1 - gamma[2*i+1,2*i+1])
    rho[1,1] = (1 - gamma[2*i,2*i])*(gamma[2*i+1,2*i+1])
    rho[2,2] = (gamma[2*i,2*i])*(1 - gamma[2*i+1,2*i+1])
    rho[3,3] = (gamma[2*i,2*i])*(gamma[2*i+1,2*i+1])
    return rho

def two_orb_rdm_mean_field(gamma,i,j):

    gamma = np.kron(gamma/2,np.eye(2))

    rho = np.zeros((16,16))

    # C_i D_i
    id_nu = gamma[2*j,2*j]
    # C_i D_i C_j D_j
    nu_nu = gamma[2*i,2*i]*gamma[2*j,2*j] - gamma[2*i,2*j]*gamma[2*j,2*i]
    nd_nu = gamma[2*i+1,2*i+1]*gamma[2*j,2*j]
    nn_nu = gamma[2*i+1,2*i+1]*nu_nu

    id_nd = gamma[2*j+1,2*j+1]
    nu_nd = gamma[2*i,2*i]*gamma[2*j+1,2*j+1]
    nd_nd = gamma[2*i+1,2*i+1]*gamma[2*j+1,2*j+1]-gamma[2*i+1,2*j+1]*gamma[2*j+1,2*i+1]
    nn_nd = gamma[2*i,2*i]*nd_nd

    id_nn = gamma[2*j,2*j]*gamma[2*j+1,2*j+1]
    nu_nn = gamma[2*j+1,2*j+1]*nu_nu
    nd_nn = gamma[2*j,2*j]*nd_nd
    nn_nn = nu_nu*nd_nd

    nu_id = gamma[2*i,2*i]
    nd_id = gamma[2*i+1,2*i+1]
    nn_id = gamma[2*i,2*i]*gamma[2*i+1,2*i+1]

    n0_nu = id_nu-nu_nu-nd_nu+nn_nu
    n0_nd = id_nd-nu_nd-nd_nd+nn_nd
    n0_nn = id_nn-nu_nn-nd_nn+nn_nn
    nu_n0 = nu_id-nu_nu-nu_nd+nu_nn
    nd_n0 = nd_id-nd_nu-nd_nd+nd_nn
    nn_n0 = nn_id-nn_nu-nn_nd+nn_nn
    n0_n0 = (1 - id_nu - id_nd + id_nn) - (nu_id - nu_nu - nu_nd + nu_nn) - (nd_id - nd_nu - nd_nd + nd_nn) + (nn_id - nn_nu - nn_nd + nn_nn)

    rho[0,0] = n0_n0
    rho[1,1] = n0_nu - n0_nn
    rho[2,2] = n0_nd - n0_nn
    rho[3,3] = n0_nn
    rho[4,4] = nu_n0 - nn_n0
    rho[5,5] = nu_nu - nn_nu - nu_nn + nn_nn
    rho[6,6] = nu_nd - nn_nd - nu_nn + nn_nn
    rho[7,7] = nu_nn - nn_nn
    rho[8,8] = nd_n0 - nn_n0
    rho[9,9] = nd_nu - nn_nu - nd_nn + nn_nn
    rho[10,10] = nd_nd - nn_nd - nd_nn + nn_nn
    rho[11,11] = nd_nn - nn_nn
    rho[12,12] = nn_n0 
    rho[13,13] = nn_nu - nn_nn
    rho[14,14] = nn_nd - nn_nn
    rho[15,15] = nn_nn

    
    # 1 pt
    rho[1,4] = gamma[2*i,2*j] * (1 - nd_id - id_nd + nd_nd)
    rho[4,1] = rho[1,4]
    rho[2,8] = gamma[2*i+1,2*j+1] * (1 - nu_id - id_nu + nu_nu)
    rho[8,2] = rho[2,8]

    
    # 2 pt # CD for spin up

    rho[3,6] = gamma[2*j,2*i]*(id_nd - nd_nd)
    rho[6,3] = rho[3,6]
    rho[3,9] = -gamma[2*j+1,2*i+1]*(id_nd - nd_nd)
    rho[9,3] = rho[3,9]

    rho[3,12] = gamma[2*i,2*j]*gamma[2*i+1,2*j+1]
    rho[12,3] = rho[3,12]

    rho[6,9] = -gamma[2*i+1,2*j+1]*gamma[2*i,2*j]
    rho[9,6] = rho[6,9]

    rho[6,12] = gamma[2*j+1,2*i+1]*(nu_id-nu_nu)
    rho[12,6] = rho[6,12]
    rho[9,12] = -gamma[2*j,2*i]*(nd_id-nd_nd)
    rho[12,9] = rho[9,12]
    

    # 3 pt 
    rho[7,13] = -gamma[2*i+1,2*j+1]*nu_nu
    rho[13,7] = rho[7,13]
    rho[11,14] = -gamma[2*i,2*j]*nd_nd
    rho[14,11] = rho[11,14]
    

    return rho

def entropy(rho):
    #rho = (rho + rho.T)/2
    
    w,v = np.linalg.eigh(rho)
    S = 0
    for i in range(len(w)):
        if abs(np.imag(w[i]))>1e-12:
            print('not real eigenvalue')
        else:
            w[i] = np.real(w[i])
            if w[i] > 1e-12:
                S -= w[i]*np.log(w[i])

    return S

def get_MI(ket,driver):
    ordm1 = driver.get_orbital_entropies(ket, orb_type=1)
    ordm2 = driver.get_orbital_entropies(ket, orb_type=2)
    dim = len(ordm1)
    MI = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                MI[i,j] = ordm1[i] + ordm1[j] - ordm2[i,j]
    return MI

def get_coh(ket,driver,i,j):
    op = driver.expr_builder()
    op.add_term("CcDd", [i,i,j,j], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    a = abs(np.real(driver.expectation(ket, n_mpo, ket)))
    op = driver.expr_builder()
    op.add_term("CcDd", [i,j,j,i], 1)
    n_mpo = driver.get_mpo(op.finalize(), iprint=0)
    b = abs(np.real(driver.expectation(ket, n_mpo, ket)))
    return a,b

def max_coh(dm1,orbs_atomic_index,norbs_in_atoms,orb_alive):
    print('Optimizing MEAO...')
    no = len(dm1)
    
    
    
    def cost_at_0(rdm1):
        c = 0
        for i in range(no):
            for j in range(i):
                if orbs_atomic_index[i]!=orbs_atomic_index[j]:
                    #i = pair[0]
                    #j = pair[1]
                    c -= 2*abs(rdm1[i,j]/2)**4
        return c
    
    def cost_num_gradient_at_0(rdm2__):
        grad = np.zeros((no,no))
        hess = np.zeros((no,no))
        c = cost_at_0(rdm2__)
        print(c)
        for k in range(no):
            for l in range(k):
                if orb_alive[k]*orb_alive[l]==1 and orbs_atomic_index[k]==orbs_atomic_index[l]:
                    t = 0.001
                    U = np.eye(no)
                    U[k,k] = np.cos(t)
                    U[k,l] = np.sin(t)
                    U[l,k] = -np.sin(t)
                    U[l,l] = np.cos(t)
                    rdm2_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,rdm2__,optimize='optimal')
                    c_ = cost_at_0(rdm2_)
                    U = np.eye(no)
                    U[k,k] = np.cos(t)
                    U[k,l] = -np.sin(t)
                    U[l,k] = np.sin(t)
                    U[l,l] = np.cos(t)
                    rdm2_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,rdm2__,optimize='optimal')
                    c__ = cost_at_0(rdm2_)
                    grad[k,l] = (c_-c__)/(2*t)
                    hess[k,l] = ((c_-c)/t-(c-c__)/t)/t
        return grad-grad.T, hess + hess.T

    def cost_gradient_at_0_(rdm2):
        grad = np.zeros((no,no))
        hess = np.zeros((no,no))
        def Gamma_grad(rdm2,i,j):
            g_iijj = np.zeros((no,no))
            g_ijij = np.zeros((no,no))
            for k in range(no):
                for l in range(k):
                    if orb_alive[k]*orb_alive[l]==1 and orbs_atomic_index[k]==orbs_atomic_index[l] and bool(set([k,l]) & set([i,j])):
                        g_iijj[k,l] = (i==k)*(rdm2[l,i,j,j]+rdm2[i,l,j,j]) - (i==l)*(rdm2[k,i,j,j]+rdm2[i,k,j,j]) + (j==k)*(rdm2[i,i,l,j]+rdm2[i,i,j,l]) - (j==l)*(rdm2[i,i,k,j]+rdm2[i,i,j,k]) 
                        g_ijij[k,l] = (i==k)*(rdm2[l,j,i,j]+rdm2[i,j,l,j]) - (i==l)*(rdm2[k,j,i,j]+rdm2[i,j,k,j]) + (j==k)*(rdm2[i,l,i,j]+rdm2[i,j,i,l]) - (j==l)*(rdm2[i,k,i,j]+rdm2[i,j,i,k]) 

            return g_iijj - g_iijj.T, g_ijij - g_ijij.T
        
        def Gamma_hess(rdm2,i,j):
            h_iijj = np.zeros((no,no))
            h_ijij = np.zeros((no,no))
            for k in range(no):
                for l in range(k):
                    if orb_alive[k]*orb_alive[l]==1 and orbs_atomic_index[k]==orbs_atomic_index[l] and bool(set([k,l]) & set([i,j])):
                        h_iijj[k,l] = -2*(len(set([i,j]) & set([k,l])))*rdm2[i,i,j,j] + 2*((i==k)*rdm2[l,l,j,j] + (i==l)*rdm2[k,k,j,j] + (j==k)*rdm2[i,i,l,l] + (j==l)*rdm2[i,i,k,k]) - 2*(len(set([i,j]) & set([k,l]))==2)*(rdm2[i,j,i,j]+rdm2[i,j,j,i]+rdm2[j,i,i,j]+rdm2[j,i,j,i])
                        h_ijij[k,l] = -2*(len(set([i,j]) & set([k,l])))*rdm2[i,j,i,j] + 2*((i==k)*rdm2[l,j,l,j] + (i==l)*rdm2[k,j,k,j] + (j==k)*rdm2[i,l,i,l] + (j==l)*rdm2[i,k,i,k]) - 2*(len(set([i,j]) & set([k,l]))==2)*(rdm2[j,i,i,j]+rdm2[j,j,i,i]+rdm2[i,i,j,j]+rdm2[i,j,j,i])

            return h_iijj + h_iijj.T, h_ijij + h_ijij.T

        for i in range(no):
            for j in range(i):
                if orbs_atomic_index[i]!=orbs_atomic_index[j]:
                    g1,g2 = Gamma_grad(rdm2,i,j)
                    h1,h2 = Gamma_hess(rdm2,i,j)
                    grad += -2*(rdm2[i,i,j,j]**1) *g1 - 2*(rdm2[i,j,i,j]**1) *g2
                    hess += -2*(rdm2[i,i,j,j]**0) * (g1**2) - 2*(rdm2[i,i,j,j]**1) *h1 - 2*(rdm2[i,j,i,j]**0) * (g2**2) - 2*(rdm2[i,j,i,j]**1) *h2 

        return grad, hess

    def cost_gradient_at_0(rdm1):
        grad = np.zeros((no,no))
        hess = np.zeros((no,no))
        def Gamma_grad(rdm1,i,j):
            g_ = np.zeros((no,no))
            for k in range(no):
                for l in range(k):
                    if orb_alive[k]*orb_alive[l]==1 and orbs_atomic_index[k]==orbs_atomic_index[l] and bool(set([k,l]) & set([i,j])):
                        g_[k,l] = (i==k)*(rdm1[l,j]*rdm1[i,j]/2) - (i==l)*(rdm1[k,j]*rdm1[i,j]/2) + (j==k)*(rdm1[i,j]*rdm1[i,l]/2) - (j==l)*(rdm1[i,k]*rdm1[i,j]/2) 

            return g_ - g_.T
        
        def Gamma_hess(rdm1,i,j):
            h_ = np.zeros((no,no))
            for k in range(no):
                for l in range(k):
                    if orb_alive[k]*orb_alive[l]==1 and orbs_atomic_index[k]==orbs_atomic_index[l] and bool(set([k,l]) & set([i,j])):
                        h_[k,l] = -1/2*(len(set([i,j]) & set([k,l])))*rdm1[i,j]**2 + 1/2*((i==k)*rdm1[j,l]**2 + (i==l)*rdm1[j,k]**2 + (j==k)*rdm1[i,l]**2 + (j==l)*rdm1[i,k]**2) - (len(set([i,j]) & set([k,l]))==2)*(rdm1[i,i]*rdm1[j,j]+rdm1[i,j]**2)

            return h_ + h_.T

        for i in range(no):
            for j in range(i):
                if orbs_atomic_index[i]!=orbs_atomic_index[j]:
                    g = Gamma_grad(rdm1,i,j)
                    h = Gamma_hess(rdm1,i,j)
                    grad += -4*((rdm1[i,j]/2)**6)*g
                    hess += -12*((rdm1[i,j]/2)**4)*(g**2) - 4*((rdm1[i,j]/2)**6)*h
                    #grad += -(rdm1[i,j]**2) *g
                    #hess += -4 * (g**2) - (rdm1[i,j]**2) *h

        return grad, hess



    norbs_in_atoms = np.array(norbs_in_atoms)
    dim = int(sum(norbs_in_atoms**2))
    #dim = len(orb_pairs)
    x0 = np.random.rand(dim,1)-0.5
    x0 = np.zeros((dim,1))
    m=0
    X = np.zeros((no,no))
    for i in range(no):
        for j in range(i):
            if orb_alive[i]*orb_alive[j]==1 and orbs_atomic_index[i]==orbs_atomic_index[j]:
                X[i,j] = x0[m,0]*2*math.pi/10
                m += 1
    U0 = expm(X-X.T)
    dm10 = U0 @ dm1 @ U0.T

    U_tot = U0
    grad = np.ones((no,no))
    
    n=0
    cost_old = cost_at_0(dm10)
    print(cost_old)
    delta_cost = np.inf
    cost_min = np.inf
    level_shift = 1e-10
    thresh = 1e-10
    max_cycle = 200
    step_size = 0.2

    while np.abs(delta_cost) > thresh and n < max_cycle:
        n += 1
        grad,hess = cost_gradient_at_0(dm10)
        hess = hess * (abs(hess)>1e-10)
        #grad_,hess_ = cost_num_gradient_at_0(rdm20)
        #print(sum(sum(abs(grad-grad_))),sum(sum(abs(hess-hess_))))
        min_hess = hess.min()
        #print('min_hess',min_hess)
        level_shift = min_hess - 1e-10  #+ (n/max_cycle)*1e-3

        denom = hess - level_shift*np.ones((no,no))
        X = -np.divide(grad, denom, out=np.zeros_like(grad), where=np.abs(denom)>1e-12) #* step_size 
        
        U = expm(X)
        U_tot = np.matmul(U,U_tot)

        dm10_ = U @ dm10 @ U.T
        cost = cost_at_0(dm10_)
        print(cost)
        delta_cost = cost_old - cost

        if cost < cost_min:
            U_tot_final = U_tot
            cost_min = cost
        
        dm10 = dm10_.copy()

        cost_old = cost



    '''
    
    def cost(x,rdm2):
        X = np.zeros((no,no))
        m = 0
        for i in range(no):
            for j in range(i):
                if orb_alive[i]*orb_alive[j]==1 and orbs_atomic_index[i]==orbs_atomic_index[j]:
                    X[i,j] = x[m]
                    m += 1
        #for pair in orb_pairs:
        #    X[pair[0],pair[1]] = x[m]
        X = X - X.T
        U = expm(X)
        rdm2_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,rdm2,optimize='optimal')
        c = 0
        #for pair in orb_pairs:
        for i in range(no):
            for j in range(i):
                if orbs_atomic_index[i]!=orbs_atomic_index[j]:
                    #i = pair[0]
                    #j = pair[1]
                    c += abs(rdm2_[i,i,j,j])**4
                    c += abs(rdm2_[i,j,i,j])**4
        #print(c)
        return -c

    norbs_in_atoms = np.array(norbs_in_atoms)
    dim = int(sum(norbs_in_atoms*(norbs_in_atoms-1)/2))
    #dim = len(orb_pairs)
    x0 = np.random.rand(dim,1)-0.5

    result = minimize(cost,x0,rdm2,'Powell')
    print('MEAO optimized...')
    x = result.x
    X = np.zeros((no,no))
    m = 0
    for i in range(no):
        for j in range(i):
            if orb_alive[i]*orb_alive[j]==1 and orbs_atomic_index[i]==orbs_atomic_index[j]:
                X[i,j] = x[m]
                m += 1
    #for pair in orb_pairs:
    #    X[pair[0],pair[1]] = x[m]
    X = X - X.T
    U_tot = expm(X)
    

    #rdm2_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U_tot,U_tot,U_tot,U_tot,rdm2,optimize='optimal')
    print(cost(x,rdm2))
    #for pair in orb_pairs:
    #    i = pair[0]
    #    j = pair[1]
    #    print(abs(rdm2_[i,i,j,j]),abs(rdm2_[i,j,i,j]))
    '''

    return U_tot_final

def max_coh_2rdm(dm2,orbs_atomic_index,norbs_in_atoms,orb_alive):
    print('Optimizing MEAO...')
    no = len(dm2)
    
    
    
    def cost_at_0(rdm2):
        c = 0
        for i in range(no):
            for j in range(i):
                if orbs_atomic_index[i]!=orbs_atomic_index[j]:
                    #i = pair[0]
                    #j = pair[1]
                    c -= abs(rdm2[i,i,j,j])**2 + abs(rdm2[i,j,i,j])**2
        return c
    
    def cost_at_0_display(rdm2):
        c = 0
        for i in range(no):
            for j in range(i):
                if orbs_atomic_index[i]!=orbs_atomic_index[j]:
                    #i = pair[0]
                    #j = pair[1]
                    c -= abs(rdm2[i,i,j,j])**2 * abs(rdm2[i,j,i,j])**2
        return c
    

    def cost_gradient_at_0(rdm2):
        grad = np.zeros((no,no))
        hess = np.zeros((no,no))
        def Gamma_grad(rdm2,i,j):
            g_iijj = np.zeros((no,no))
            g_ijij = np.zeros((no,no))
            for k in range(no):
                for l in range(k):
                    if orb_alive[k]*orb_alive[l]==1 and orbs_atomic_index[k]==orbs_atomic_index[l] and bool(set([k,l]) & set([i,j])):
                        g_iijj[k,l] = (i==k)*(rdm2[l,i,j,j]+rdm2[i,l,j,j]) - (i==l)*(rdm2[k,i,j,j]+rdm2[i,k,j,j]) + (j==k)*(rdm2[i,i,l,j]+rdm2[i,i,j,l]) - (j==l)*(rdm2[i,i,k,j]+rdm2[i,i,j,k]) 
                        g_ijij[k,l] = (i==k)*(rdm2[l,j,i,j]+rdm2[i,j,l,j]) - (i==l)*(rdm2[k,j,i,j]+rdm2[i,j,k,j]) + (j==k)*(rdm2[i,l,i,j]+rdm2[i,j,i,l]) - (j==l)*(rdm2[i,k,i,j]+rdm2[i,j,i,k]) 

            return g_iijj - g_iijj.T, g_ijij - g_ijij.T
        
        def Gamma_hess(rdm2,i,j):
            h_iijj = np.zeros((no,no))
            h_ijij = np.zeros((no,no))
            for k in range(no):
                for l in range(k):
                    if orb_alive[k]*orb_alive[l]==1 and orbs_atomic_index[k]==orbs_atomic_index[l] and bool(set([k,l]) & set([i,j])):
                        h_iijj[k,l] = -2*(len(set([i,j]) & set([k,l])))*rdm2[i,i,j,j] + 2*((i==k)*rdm2[l,l,j,j] + (i==l)*rdm2[k,k,j,j] + (j==k)*rdm2[i,i,l,l] + (j==l)*rdm2[i,i,k,k]) - 2*(len(set([i,j]) & set([k,l]))==2)*(rdm2[i,j,i,j]+rdm2[i,j,j,i]+rdm2[j,i,i,j]+rdm2[j,i,j,i])
                        h_ijij[k,l] = -2*(len(set([i,j]) & set([k,l])))*rdm2[i,j,i,j] + 2*((i==k)*rdm2[l,j,l,j] + (i==l)*rdm2[k,j,k,j] + (j==k)*rdm2[i,l,i,l] + (j==l)*rdm2[i,k,i,k]) - 2*(len(set([i,j]) & set([k,l]))==2)*(rdm2[j,i,i,j]+rdm2[j,j,i,i]+rdm2[i,i,j,j]+rdm2[i,j,j,i])

            return h_iijj + h_iijj.T, h_ijij + h_ijij.T

        for i in range(no):
            for j in range(i):
                if orbs_atomic_index[i]!=orbs_atomic_index[j]:
                    g1,g2 = Gamma_grad(rdm2,i,j)
                    h1,h2 = Gamma_hess(rdm2,i,j)
                    #grad += -2*(rdm2[i,i,j,j]**1)*(rdm2[i,j,i,j]**2) *g1 - 2*(rdm2[i,j,i,j]**1)*(rdm2[i,i,j,j]**2) *g2
                    #hess += -2*((rdm2[i,i,j,j]*h1 + g1**2)*(rdm2[i,j,i,j]**2) + 2*rdm2[i,i,j,j]*g1*2*rdm2[i,j,i,j]*g2 + (rdm2[i,j,i,j]*h2 + g2**2)*(rdm2[i,i,j,j]**2) )
                    #grad += -4*(rdm2[i,i,j,j]**3)*(rdm2[i,j,i,j]**4) *g1 - 4*(rdm2[i,j,i,j]**3)*(rdm2[i,i,j,j]**4) *g2
                    #hess += -4*rdm2[i,i,j,j]**2*rdm2[i,j,i,j]**2*(4*rdm2[i,i,j,j]*g1*g2+rdm2[i,i,j,j]**2*h2 + 3*rdm2[i,i,j,j]**2*g2**2 + 4*rdm2[i,j,i,j]*g1*g2 + rdm2[i,j,i,j]**2*h1 + 3*rdm2[i,j,i,j]**2*g1**2)
                    grad += -2*(rdm2[i,i,j,j]**1) *g1 - 2*(rdm2[i,j,i,j]**1) *g2
                    hess += -2*(rdm2[i,i,j,j]**0) * (g1**2) - 2*(rdm2[i,i,j,j]**1) *h1 - 2*(rdm2[i,j,i,j]**0) * (g2**2) - 2*(rdm2[i,j,i,j]**1) *h2 
        return grad, hess

    



    norbs_in_atoms = np.array(norbs_in_atoms)
    dim = int(sum(norbs_in_atoms**2))
    #dim = len(orb_pairs)
    

    cost_list = []
    U_list = []

    for l in range(10):
        x0 = np.random.rand(dim,1)-0.5
        #x0 = np.zeros((dim,1))
        m=0
        X = np.zeros((no,no))
        for i in range(no):
            for j in range(i):
                if orb_alive[i]*orb_alive[j]==1 and orbs_atomic_index[i]==orbs_atomic_index[j]:
                    X[i,j] = x0[m,0]*2*math.pi/5
                    m += 1
        U0 = expm(X-X.T)
        #dm10 = U0 @ dm1 @ U0.T
        dm20 = np.einsum('ia,jb,kc,ld,abcd->ijkl',U0,U0,U0,U0,dm2,optimize='optimal')

        U_tot = U0
        grad = np.ones((no,no))
        
        n=0
        cost_old = cost_at_0(dm20)
        #print(cost_old)
        delta_cost = np.inf
        cost_min = np.inf
        level_shift = 1e-10
        thresh = 1e-10
        max_cycle = 500
        step_size = 0.8
        while np.abs(delta_cost) > thresh and n < max_cycle:
            n += 1
            grad,hess = cost_gradient_at_0(dm20)
            hess = hess * (abs(hess)>1e-10)
            #grad_,hess_ = cost_num_gradient_at_0(rdm20)
            #print(sum(sum(abs(grad-grad_))),sum(sum(abs(hess-hess_))))
            min_hess = hess.min()
            #print('min_hess',min_hess)
            level_shift = min_hess - 1e-10  #+ (n/max_cycle)*1e-3

            denom = hess - level_shift*np.ones((no,no))
            X = -np.divide(grad, denom, out=np.zeros_like(grad), where=np.abs(denom)>1e-12) * step_size 
            
            U = expm(X)
            U_tot = np.matmul(U,U_tot)

            #dm10_ = U @ dm10 @ U.T
            dm20_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,dm20,optimize='optimal')
            cost = cost_at_0(dm20_)
            #print(cost)
            delta_cost = cost_old - cost

            if cost < cost_min:
                U_tot_final = U_tot
                cost_min = cost
            
            dm20 = dm20_.copy()
            cost_print = cost_at_0_display(dm20)

            cost_old = cost

        print(cost_old,cost_print)
        cost_list.append(cost_old)
        U_list.append(U_tot_final)

    U_tot_final = U_list[np.argmin(np.array(cost_list))]

    return U_tot_final

def max_orb_ent(rdm1,rdm2,orbs_atomic_index,norbs_in_atoms,orb_alive):
    print('Optimizing MEAO...')
    no = len(rdm2)
    
    def cost(x):
        X = np.zeros((no,no))
        m = 0
        for i in range(no):
            for j in range(i):
                if orb_alive[i]*orb_alive[j]==1 and orbs_atomic_index[i]==orbs_atomic_index[j]:
                    X[i,j] = x[m]
                    m += 1
        #for pair in orb_pairs:
        #    X[pair[0],pair[1]] = x[m]
        X = X - X.T
        U = expm(X)
        rdm1_ = np.einsum('ia,jb,ab->ij',U,U,rdm1,optimize='optimal')
        rdm2_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,rdm2,optimize='optimal')
        c = 0
        #for pair in orb_pairs:
        for i in orb_alive:
            c += entropy(np.diag([1-rdm1_[i,i]+rdm2_[i,i,i,i],rdm1_[i,i]/2-rdm2_[i,i,i,i],rdm1_[i,i]/2-rdm2_[i,i,i,i],rdm2_[i,i,i,i]]))
        #print(c)
        return -c



    norbs_in_atoms = np.array(norbs_in_atoms)
    dim = int(sum(norbs_in_atoms*(norbs_in_atoms-1)/2))
    #dim = len(orb_pairs)
    x0 = np.random.rand(dim,1)-0.5

    result = minimize(cost,x0,'Powell')
    print('MEAO optimized...')
    x = result.x
    X = np.zeros((no,no))
    m = 0
    for i in range(no):
        for j in range(i):
            if orb_alive[i]*orb_alive[j]==1 and orbs_atomic_index[i]==orbs_atomic_index[j]:
                X[i,j] = x[m]
                m += 1
    #for pair in orb_pairs:
    #    X[pair[0],pair[1]] = x[m]
    X = X - X.T
    U = expm(X)
    
    #rdm2_ = np.einsum('ia,jb,kc,ld,abcd->ijkl',U,U,U,U,rdm2,optimize='optimal')

    #for pair in orb_pairs:
    #    i = pair[0]
    #    j = pair[1]
    #    print(abs(rdm2_[i,i,j,j]),abs(rdm2_[i,j,i,j]))

    return U

def make_rdm2_mean_field(gamma):

    no = len(gamma)

    #dm2 = np.kron(gamma/2,gamma/2).reshape((no,no,no,no))
    #Gamma = dm2.transpose((0,2,3,1)).copy()
    
    Gamma = np.zeros((no,no,no,no))
    for i in range(no):
        for j in range(no):
            for k in range(no):
                for l in range(no):
                    Gamma[i,j,l,k] = gamma[i,k]/2*gamma[j,l]/2


    
    #Gamma = dm2.copy()
    #Gamma = (2 * Gamma + Gamma.transpose((0, 1, 3, 2))) / 6.


    return Gamma

def prep_rdm2(dm2):
    '''
    Prepare the 1- and 2-RDM (splitting 1-RDM into spin parts and fix prefactor of 2-RDM)
    This only works for singlet states.
    For other spin states, one should run spin unrestricted DMRG and get the 
    spin 1- and 2-RDMs.

    Args:
        dm1 (ndarray): spatial-orbital 1RDM from pyscf
        dm2 (ndarray): spatial-orbital 2RDM from pyscf

    Returns:
        rdm1(ndarray): prepared 1RDM in spin-orbital indices
        rdm2(ndarray): prepared relevant part of the 2RDM in orbital indices and spin (up,down,down,up)
    '''
    rdm2 = dm2.transpose((0,2,3,1)).copy()
    rdm2 = (2 * rdm2 + rdm2.transpose((0, 1, 3, 2))) / 6.

    return rdm2

def orb_pairing(orbs_in_atoms,connectivity):
    partition=[]
    m = 0
    orbs_atomic_index = []
    atomic_index = 0
    for n in orbs_in_atoms:
        part = list(range(m,m+n))
        orbs_atomic_index = orbs_atomic_index + [atomic_index]*n
        atomic_index += 1
        partition.append(part)
        m = m + n
    print(partition,orbs_atomic_index)
    orb_pairs = []
    for link in connectivity:
        cluster_a = partition[link[0]]
        cluster_b = partition[link[1]]
        for i in cluster_a:
            for j in cluster_b:
                orb_pairs.append([i,j])
    print(orb_pairs)
    return orb_pairs, orbs_atomic_index

def get_ent(orb_rdms,n_sites):

    def ent_pssr(rho):
        ent = 0
        if rho[3,12]**2 > rho[0,0]*rho[15,15]:
            rho[3,12] = abs(rho[3,12])
            rho[3,3] = (rho[3,3]+rho[12,12])/2
            rho[12,12] = rho[3,3]
            s = rho[0,0]+rho[3,3]+rho[12,12]+rho[15,15]
            A = s**2 - (rho[0,0]-rho[15,15])**2
            B = 2*rho[3,12]*s
            C = (rho[0,0]+rho[15,15])**2 * (2*rho[3,12])**2 + 8*rho[0,0]*rho[15,15]*(2*rho[0,0]*rho[15,15]+(rho[0,0]+rho[15,15])*(rho[3,3]+rho[12,12]) +2*(rho[3,3]+rho[3,12])*(rho[3,3]-rho[3,12]))
            q8 = (A+B+np.sqrt(C))/(4*(s-rho[3,3]+rho[3,12]))
            q9 = (A-B-np.sqrt(C))/(4*(s-rho[3,3]-rho[3,12]))
            q10 = rho[0,0] + rho[3,3]-q8/2-q9/2
            q11 = rho[15,15] + rho[3,3]-q8/2-q9/2
            p = [rho[0,0],rho[3,3]+rho[3,12],rho[3,3]-rho[3,12],rho[15,15]]
            q = [q10,q8,q9,q11]
            for i in range(4):
                if p[i]*q[i] > 0:
                    ent += p[i]*np.log(p[i]/q[i])
        return ent
    
    def ent_nssr(rho):
        t = max(rho[6,6]+rho[6,9],rho[6,6]-rho[6,9])
        r = min(rho[6,6]+rho[6,9],rho[6,6]-rho[6,9]) + rho[5,5] + rho[10,10]
        ent = 0
        if r < t:
            ent = r*np.log(2*r/(r+t)) + t*np.log(2*t/(r+t))
        return ent

    epssr = np.zeros((n_sites,n_sites))
    enssr = np.zeros((n_sites,n_sites))

    for i in range(n_sites):
        irdms = orb_rdms[i]
        for j in range(i+1,n_sites):
            rdm = irdms[j-i-1]
            epssr[i,j] = ent_pssr(rdm)
            enssr[i,j] = ent_nssr(rdm)
    
    epssr = epssr + epssr.T
    enssr = enssr + enssr.T

    return epssr,enssr

def MI_mean_field(rdm1):
    
    norb = len(rdm1)

    MI = np.zeros((norb,norb))

    for i in range(norb):
        for j in range(i):
            MI[i,j] = entropy(one_orb_rdm_mean_field(rdm1,i)) + entropy(one_orb_rdm_mean_field(rdm1,j)) - entropy(two_orb_rdm_mean_field(rdm1,i,j))

    MI = (MI + MI.T)/np.log(16)

    return MI

def analyze_block_correlation(gamma,centers):

    def block_entropy(block):
        gamma_ = gamma.copy()[block][:,block]
        gamma_ = np.kron(np.eye(2),gamma_/2)

        no = len(gamma_)

        return entropy(gamma_) + entropy(np.eye(no)-gamma_)

    keys = list(centers.keys())
    for i in range(len(keys)):
        center1 = keys[i]
        for j in range(i+1,len(keys)):
            center2 = keys[j]
            Corr = (block_entropy(centers[center1])+block_entropy(centers[center2])-block_entropy(centers[center1]+centers[center2]))/np.log(16)
            print(center1+'-'+center2+' Correlation:', Corr)

def direct_MI(orb_rdms,n_sites):

    def partial_trace(Rho,n):
        rho = np.zeros((4,4))
        if n == 1:
            for i in range(4):
                for j in range(4):
                    for m in range(4):
                        rho[i,j] += Rho[4*i+m,4*j+m]
        if n == 2:
            for i in range(4):
                for j in range(4):
                    for m in range(4):
                        rho[i,j] += Rho[4*m+i,4*m+j]
        return rho
    
    def entropy(rho):
        w,v = np.linalg.eigh(rho)
        S = 0
        for i in range(len(w)):
            if w[i] > 0:
                S += -w[i]*np.log(w[i])
            else:
                print('Warning: negative eigenvalue in entropy calculation:',w[i])
        return S

    MI = np.zeros((n_sites,n_sites))
    for i in range(n_sites):
        irdms = orb_rdms[i]
        for j in range(i+1,n_sites):
            rdm = irdms[j-i-1]
            MI[i,j] = entropy(partial_trace(rdm,1)) + entropy(partial_trace(rdm,2)) - entropy(rdm)
    
    MI = MI + MI.T
    return MI

def write_2ordm_mean_field(file_name,gamma, indices):


    

    if isinstance(indices, list):

        print(indices)
        os.system('rm '+file_name)
        fd = open(file_name, 'x')
        for ind in indices:
            ordm = np.zeros((16,16))
            ordm = two_orb_rdm_mean_field(gamma,ind[0],ind[1])
            fd.write(str(ind[0])+'\t'+str(ind[1])+'\n')
            for m in range(16):
                for n in range(16):
                    fd.write('{:.16f}'.format(ordm[m,n])+'\t')
                fd.write('\n')

    fd.close()

def write_2ordm_mps(file_name, ket, driver, norb, reorder=None, usesym=True):

    os.system('rm '+file_name)
    
    if isinstance(norb, int):


        if reorder == None:
            reorder = list(range(norb))
    
        ordm = np.zeros((16,16,norb,norb))

        for i in range(norb):
            for j in range(i):
                i_ = max([reorder[i],reorder[j]])
                j_ = min([reorder[i],reorder[j]])
                print(i,j)
                if usesym:
                    ordm[:,:,i_,j_] = two_orb_rdm(ket,driver,i,j)
                else:
                    ordm[:,:,i_,j_] = two_orb_rdm_no_sym(ket,driver,i,j)
                
        fd = open(file_name, 'x')
        for i in range(norb):
            for j in range(i):
                for m in range(16):
                    for n in range(16):
                        fd.write('{:.16f}'.format(ordm[m,n,i,j])+'\t')
                    fd.write('\n')

        fd.close()
    
    if isinstance(norb, list):

        print(norb)
        

        fd = open(file_name, 'x')
        for ind in norb:
            ordm = np.zeros((16,16))
            ordm[:,:] = two_orb_rdm(ket,driver,ind[0],ind[1])
            for m in range(16):
                for n in range(16):
                    fd.write('{:.16f}'.format(ordm[m,n])+'\t')
                fd.write('\n')

        fd.close()


import networkx as nx

def find_disconnected_components(adj_matrix):
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Find connected components
    components = list(nx.connected_components(G))

    # Extract the vertices of each component
    components_vertices = [list(component) for component in components]

    return components_vertices

def get_cluster_index(dm1, threshold):
    MI = MI_mean_field(dm1)
    MI = (MI>threshold)

    clusters = find_disconnected_components(MI)

    # keep only clusters with more than 1 orbital
    clusters = [cluster for cluster in clusters if len(cluster) > 1]

    cluster_size = 3
    biggest_cluster = None

    for cluster in clusters:
        if len(cluster) >= cluster_size:
            cluster_size = len(cluster)
            biggest_cluster = cluster

    return clusters, biggest_cluster
    

    



def GME_mean_field(gamma):
    no = len(gamma)
    orbs = list(range(no))
    gamma = gamma/2
    gme = 100
    for L in range(1,ceil((no+1)/2)):
        gmeL = 100
        for subset in combinations(orbs,L):
            g = np.array([x[list(subset)] for x in gamma[list(subset)]])
            s = 2*(entropy(g)+entropy(np.eye(L)-g))/np.log(4)
            if s < gmeL:
                gmeL = s
            if s < gme:
                gme = s
                subset_opt = subset
        print(L,gmeL)

    g = np.array([x[list(subset)] for x in gamma[list(subset)]])
    print('Subset of smallest entropy:',list(subset_opt),'Entropy:',gme)

    return gme, subset_opt


def GME_mps(driver,ket):
    odm1 = driver.get_orbital_entropies(ket, orb_type=1)/np.log(4)
    odm2 = driver.get_orbital_entropies(ket, orb_type=2)/np.log(4)
    bip_ent = driver.get_bipartite_entanglement()/np.log(4)
    return min([odm1.min(),odm2.min(),bip_ent.min()])

def reorder_mps(Corr):
    norb = len(Corr)
    i=0
    reorder = []
    used_index = []
    while len(reorder) < norb:
        if i not in used_index:
            #print(reorder)
            reorder.append(i)
            used_index.append(i)
            corr = Corr[i,:]
            corr[used_index] = -1
            i = np.argmax(corr)

    return reorder