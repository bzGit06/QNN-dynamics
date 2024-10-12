import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from opt_einsum import contract
from itertools import combinations, product

def KmeanTheory_RPA(n, L, O):
    # analytical result of haar average \overline{K}
    d = 2**n
    Otr, O2tr = np.trace(O), np.trace(O @ O)
    Kmean = (d*O2tr-Otr**2)*L/(2*(d-1)*(d+1)**2)
    return np.real(Kmean)

def KstdTheoryInd_RPA(n, L, O):
    # analytical result of haar average \Delta K
    d = 2**n
    Otr, O2tr= np.trace(O), np.trace(O @ O)
    O3tr, O4tr = np.trace(O @ O @ O), np.trace(O @ O @ O @ O)
    Ksq_l1l2 = L*(L-1)*(d*O2tr-Otr**2)**2/(4*(d-1)**2*(d+1)**4)
    Ksq_l = L*3*((d**2+3*d+3)*O2tr**2 + d*(d+1)*O4tr + Otr**4 -2*d*Otr**2*O2tr
                 -4*(d+1)*Otr*O3tr)/(4*(d+3)**2*(d+1)**2*d*(d-1))
    return np.sqrt(Ksq_l+Ksq_l1l2 - KmeanTheory_RPA(n, L, O)**2)

def gl1gl2sq(n, O):
    # haar average of (\parital_{l1} \epsilon)^2(\parital_{l1} \epsilon)^2 with l1<l2
    # only U_{0l1}, U_{l1l2}, U_{l2L} are independent unitary
    d = 2**n
    Otr, O2tr= np.trace(O), np.trace(O @ O)
    O3tr, O4tr = np.trace(O @ O @ O), np.trace(O @ O @ O @ O)
    coeff = d/(4*(d-1)**2*(d+1)**3*(d+2)*(d+3))
    gl1gl2 = coeff*((d**2+3*d+3)*O2tr**2 + d*(d+1)*O4tr + Otr**4 - 2*d*O2tr*Otr**2
                    -4*(d+1)*O3tr*Otr)
    return gl1gl2

def glquad(n, O):
    # haar average of (\parital_l \epsilon)^4
    d = 2**n
    Otr, O2tr= np.trace(O), np.trace(O @ O)
    O3tr, O4tr = np.trace(O @ O @ O), np.trace(O @ O @ O @ O)
    coeff = 3/(4*d*(d-1)*(d+1)**2*(d+3)**2)
    glgl = coeff*((d**2+3*d+3)*O2tr**2 + d*(d+1)*O4tr + Otr**4 - 2*d*O2tr*Otr**2
                  -4*(d+1)*O3tr*Otr)
    return glgl

def KstdTheoryCor_RPA(n, L, O):
    # analytical result of haar average of std of K
    # assuming correlated Haar unitaries
    Ksq = gl1gl2sq(n, O)*L*(L-1)+glquad(n, O)*L
    Kbar = KmeanTheory_RPA(n, L, O)
    Kvar = Ksq - Kbar**2
    return np.sqrt(Kvar)

def gl1l2gl1gl2Cor(n, O):
    # haar average of (\parital_{l1l2} \epsilon)(\parital_{l1}\epsilon)(\parital_{l2}\epsilon) with l1<l2
    # only U_{0l1}, U_{l1l2}, U_{l2L} are independent unitary
    d = 2**n
    Otr, O2tr, O3tr = np.real(np.trace(O)), np.real(np.trace(O @ O)), np.real(np.trace(O @ O @ O))
    coeff = d/(8*(d-1)**2*(d+1)**3*(d+2))
    gl1l2gl1gl2 = coeff*(d**2*O3tr-3*d*O2tr*Otr+2*Otr**3)
    return gl1l2gl1gl2

def MumeanCor_RPA(n, L, O):
    # analytical result of haar average of mean of mu
    # assuming correlated Haar unitaries
    mu = L*(L-1)*gl1l2gl1gl2Cor(n, O)
    return mu

def epsgllglgl(n, O):
    # haar average of \epsilon(\parital^2_{l^2}\epsilon)(\parital_l\epsilon)^2
    d = 2.**n
    Otr, O2tr = np.real(np.trace(O)), np.real(np.trace(O @ O))
    O3tr, O4tr = np.real(np.trace(O @ O @ O)), np.real(np.trace(O @ O @ O @ O))
    denom = 4*d*(d-1)*(d+1)**2*(d-2)*(d+2)**2*(d-3)*(d+3)**2
    c1 = d**4 - 2*d**3 + 8*d + 20
    c2 = 2*d*(-d**4 + d**3 + 10*d**2 + d - 29)
    c3 = 4*(d**5 - 11*d**3 - 6*d**2 + 38*d + 14)
    c4 = d**6 + 3*d**5 - 11*d**4 -41*d**3 + 18*d**2 + 96*d + 60
    c5 = d*(d**5 - 13*d**3 - 4*d**2 + 56*d - 4)
    numer = c1*Otr**4 + c2*O2tr*Otr**2 - c3*O3tr*Otr\
            + c4*O2tr**2 + c5*O4tr
    return -numer/denom

def epsgl1l2gl1gl2(n, O, O0):
    # haar average of \epsilon(\parital^2_{l1l2}\epsilon)(\parital_l1\epsilon)(\parital_l2\epsilon)
    d = 2**n
    Otr, O2tr = np.real(np.trace(O)), np.real(np.trace(O @ O))
    O3tr, O4tr = np.real(np.trace(O @ O @ O)), np.real(np.trace(O @ O @ O @ O))
    denom = 8*(d-1)**2*(d+1)**3*(d+2)*(d+3)
    numer = 2.*Otr**4.-3.*(d-1)*(Otr**2.)*O2tr-3*(d+1)*(O2tr**2.)+(d**2-d+4)*Otr*O3tr\
        +(d**2-d)*O4tr
    return d*numer/denom - O0*gl1l2gl1gl2Cor(n, O)

def epsmuMeanCor_RPA(n, L, O, O0):
    epsmu = L*epsgllglgl(n, O) + L*(L-1)*epsgl1l2gl1gl2(n, O, O0)
    return epsmu

# def toyUnitary(d, num):
#     '''
#     generate toy unitaries in block diagonal form for state preparation
#     '''
#     Us = np.zeros((num, d, d), dtype=np.complex128)
#     Us[:, 0, 0] = 1.
#     Us[:, 1:, 1:] = unitary_group.rvs(dim=d-1, size=num)
#     return Us

# def glsq_toy(Xl, O, U1, U2):
#     '''
#     Average of mean of K over toy unitary
#     '''
#     d = O.shape[0]
#     U1_h = np.transpose(U1.conj(), (0, 2, 1))
#     U2_h = np.transpose(U2.conj(), (0, 2, 1))
#     Ou = contract('mij,jk,mkl->mil', U2_h, O, U2) # O_U
#     C = contract('ij,mjk->mik', Xl, Ou) - contract('mij,jk->mik', Ou, Xl) # [X, Ou]

#     rho0 = qt.ket2dm(qt.basis(d, 0)).full()
#     rho0u = contract('mij,jk,mkl->mil', U1, rho0, U1_h)

#     glsq = contract('mij,mjk,mkl,mln->min', rho0u, C, rho0u, C)
#     glsq = contract('mii->m', glsq)

#     return -np.mean(glsq)/4.

# def Kmean_toy(n, Xs, O, num):
#     L = len(Xs)
#     Us = toyUnitary(2**n, num)

#     U1 = unitary_group.rvs(2**n, size=num)
#     U2 = contract('mij, mjk->mik', Us, np.transpose(U1.conj(), (0, 2, 1)))
#     idx = np.random.choice(L)
#     K = L*glsq_toy(Xs[idx], O, U1, U2)
#     return np.real(K)


# def epsgllglgl_toy(Xl, O, O0, U1, U2):
#     '''
#     average of \epsilon(\parital^2_{l^2}\epsilon)(\parital_l\epsilon)^2
#     over unitaries
#     '''
#     d = O.shape[0]
#     U1_h = np.transpose(U1.conj(), (0, 2, 1))
#     U2_h = np.transpose(U2.conj(), (0, 2, 1))
#     Ou = contract('mij,jk,mkl->mil', U2_h, O, U2) # O_U
#     C = contract('ij,mjk->mik', Xl, Ou) - contract('mij,jk->mik', Ou, Xl) # [X, Ou]
#     C2 = contract('ij,mjk->mik', Xl, C) - contract('mij,jk->mik', C, Xl) # [X, [X, Ou]]
#     rho0 = qt.ket2dm(qt.basis(d, 0)).full()
#     rho0u = contract('mij,jk,mkl->mil', U1, rho0, U1_h) # U \rho_0 U^\dagger

#     term1 = contract('mij,mjk,mkl,mln->min', rho0u, Ou, rho0u, C2)
#     term1 = contract('mij,mjk,mkl->mil', term1, rho0u, C)
#     term1 = contract('mij,mjk,mkl->mil', term1, rho0u, C)
#     term1 = np.mean(contract('mii->m', term1))/16.

#     # calculate average gllglgl
#     gllglgl = contract('mij,mjk,mkl,mln->min', rho0u, C2, rho0u, C)
#     gllglgl = contract('mij,mjk,mkl->mil', gllglgl, rho0u, C)
#     gllglgl = np.mean(contract('mii->m', gllglgl))/16.

#     return term1 - O0*gllglgl

# def epsgl1l2gl1gl2_toy(Xl1, Xl2, O, O0, U1, U2, U3):
#     '''
#     average of \epsilon(\parital^2_{l1l2}\epsilon)(\parital_l1\epsilon)(\parital_l2\epsilon)
#     over unitaries
#     '''
#     d = O.shape[0]
#     U1_h = np.transpose(U1.conj(), (0, 2, 1))
#     U2_h = np.transpose(U2.conj(), (0, 2, 1))
#     U3_h = np.transpose(U3.conj(), (0, 2, 1))

#     Ou2 = contract('mij, jk, mkl->mil', U3_h, O, U3)
#     C2 = contract('ij,mjk->mik', Xl2, Ou2) - contract('mij,jk->mik', Ou2, Xl2)
#     Ou1 = contract('mij, mjk, mkl->mil', U2_h, Ou2, U2)
#     C1 = contract('ij,mjk->mik', Xl1, Ou1) - contract('mij,jk->mik', Ou1, Xl1)
    
#     C2_12 = contract('mij,mjk,mkl->mil', U2_h, C2, U2)
#     C12 = contract('ij,mjk->mik', Xl1, C2_12) - contract('mij,jk->mik', C2_12, Xl1)

#     rho0 = qt.ket2dm(qt.basis(d, 0)).full()
#     rho0u = contract('mij,jk,mkl->mil', U1, rho0, U1_h)
    
#     term1 = contract('mij,mjk,mkl,mln->min', rho0u, Ou1, rho0u, C12)
#     term1 = contract('mij,mjk,mkl->mil', term1, rho0u, C1)
#     term1 = contract('mij,mjk,mkl->mil', term1, rho0u, C2_12)
#     term1 = np.mean(contract('mii->m', term1))/16.

#     # calculate (\parital^2_{l1l2}\epsilon)(\parital_l1\epsilon)(\parital_l2\epsilon)
#     gl1l2gl1gl2 = contract('mij,mjk,mkl,mln->min', rho0u, C12, rho0u, C1)
#     gl1l2gl1gl2 = contract('mij,mjk,mkl->mil', gl1l2gl1gl2, rho0u, C2_12)
#     gl1l2gl1gl2 = np.mean(contract('mii->m', gl1l2gl1gl2))/16.

#     return term1 - O0*gl1l2gl1gl2

# def epsMu_toy(n, Xs, O, O0, num):
#     L = len(Xs)
#     Us = toyUnitary(2**n, num)
    
#     U1 = unitary_group.rvs(2**n, size=num)
#     U2 = contract('mij, mjk->mik', Us, np.transpose(U1.conj(), (0, 2, 1)))
#     idx = np.random.choice(L)
#     term1 = L*epsgllglgl_toy(Xs[idx], O, O0, U1, U2)

#     U1 = unitary_group.rvs(2**n, size=num)
#     U2 = unitary_group.rvs(2**n, size=num)
#     U3 = contract('mij, mjk, mkl->mil', Us, np.transpose(U1.conj(), (0, 2, 1)), 
#                     np.transpose(U2.conj(), (0, 2, 1)))
#     x1, x2 = np.random.choice(L, 2, replace=False)
#     term2 = L*(L-1)*epsgl1l2gl1gl2_toy(Xs[x1], Xs[x2], O, O0, U1, U2, U3)
#     return np.real(term1 + term2)

def K_toy(n, L, O0, R):
    '''
    the analytical QNTK under restricted Haar ensemble
    '''
    d = 2.**n
    return L*d*(O0 + R)* (1.- O0 - R)/(2.*(d**2-1.))

def zeta_toy(n, L, O0, R):
    '''
    \zeta = \epsilon*\mu/K^2 under restricted Haar ensemble
    '''
    d = 2.**n
    denom = (O0+R) * (O0 + R-1)
    term1 = (L-1)/(2*L) * (2*O0 + 2*R-1)*R/denom
    term2 = (d+2)*(d**2-1)/(L*d**2 * (d+3)) * ((d+2) * (O0+R)-2)*R/denom
    return term1 + term2

def lambda_toy(n, L, O0, R):
    '''
    \lambda = \epsilon*\mu/K^2 under restricted Haar ensemble
    '''
    d = 2.**n
    term1 = (L-1) * d/(4*(d**2 - 1)) * (1-2* O0 - 2* R)
    term2 = (d+2)/(2*d*(d+3)) * ((d+2)* (O0+R) - 2)
    return term1 - term2



