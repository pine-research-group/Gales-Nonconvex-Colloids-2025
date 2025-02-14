# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:33:53 2021

@author: MinJae
"""


import hoomd
import hoomd.md
import math
import os
import random
import numpy as np
from hoomd import deprecated


hoomd.context.initialize()

# -------------Poential---------------------------------------

def P_A(epsilon, sigma, x):
    U = (4 * epsilon * ((sigma / x)**(2 * n) - (sigma / x)**n))
    return round(U, 10)


def P_WCA(epsilon, sigma, x):
    U = (4 * epsilon * ((sigma / x)**(2 * n) - (sigma / x)**n) + 1 / 4)
    return round(U, 10)

def P_Morse(epsilon, a, r, r0):
    U = epsilon*(np.exp(-2*a*(r-r0)) - 2*np.exp(-1*a*(r-r0)))
    return round(U, 10)

def P_PACS(epsilon,Psi_p,Psi_n,r_p,r_n,DL,sigma,L,rcc): #x is surface to surface distance
    r = 2/(1/r_n+1/r_p)
    x = rcc - r_n - r_p
    DL = DL/1e-6
    U_E = (2*math.pi*epsilon*Psi_p*Psi_n*(r*1e-6)*math.exp(-x/DL))/4.1e-21
    if x <= 2*L and x >0:
        U_B = 16*math.pi*L**2*r*sigma**(3/2)/35*(28*((2*L/x)**(1/4)-1)+(20/11)*(1-(x/(2*L))**(11/4))+12*(x/(2*L)-1))
    else:
        U_B = 0
    U = U_B + U_E
    return U
# -------------Forces---------------------------------------


def F_A(epsilon, sigma, x):
    F = (192 * epsilon / x) * (2 * (sigma / x)**(2 * n) - ((sigma / x)**n))
    return round(F, 10)


def F_WCA(epsilon, sigma, x):
    F = (192 * epsilon / x) * (2 * (sigma / x)**(2 * n) - ((sigma / x)**n))
    return round(F, 10)

def F_Morse(epsilon, a, r, r0):
    F = epsilon*(2*a)*(np.exp(-2*a*(r-r0))-np.exp(-1*a*(r-r0)))
    return round(F, 10)
# Attractive potential
def ljA(r, rmin, rmax, epsilon, sigma):
    V = P_A(epsilon, sigma, r)
    F = F_A(epsilon, sigma, r)
    return (V, F)

# Brush repulsive

def ljR(r, rmin, rmax, epsilon, sigma):
    V = P_WCA(epsilon, sigma, r)
    F = F_WCA(epsilon, sigma, r)
    return (V, F)

def MorseA(r, rmin, rmax, r0, epsilon, a):
    V = P_Morse(epsilon, a, r, r0)
    F = F_Morse(epsilon, a, r, r0)
    return(V, F)

def P_WF(r, epsilon=1.0, rc=2., sigma=1.):
    r2 = (rc / sigma)**2
    alpha = 2. * r2 * (1.5 / (r2 - 1.))**3
    wf = epsilon * alpha * ((sigma / r)**2 - 1.) * ((rc / r)**2 - 1.)**2
    U = np.where(r < rc, wf, 0.0)
    return U


def F_WF(r, epsilon,rc,sigma):
    r2 = (rc / sigma)**2
    rsq = r**2
    rc2 = rc**2
    s2 = sigma**2
    alpha = -2. * r2 * (1.5 / (r2 - 1.))**3
    F = 2 * epsilon * alpha * ((rc2-rsq)*(s2*(rsq-3*rc2)+2*rc2*rsq))/(r**7)
    
    return F

def F_PACS(epsilon,Psi_p,Psi_n,r_p,r_n,DL,sigma,L,rcc):
    r = 2/(1/r_n+1/r_p) #in um
    x = rcc - r_n - r_p #in um
    DL = DL/1e-6 #converted to um
    F_E = 1/DL*(2*math.pi*epsilon*Psi_p*Psi_n*(r*1e-6)*math.exp(-x/DL))/4.11e-21
    if x <= 2*L and x>0:
        F_B = 16*math.pi*r*L**2*sigma**(3/2)/35*((7*(2*L)**(1/4))/(x**(5/4))+5*(x)**(7/4)/(2*L)**(11/4)-6/L)
    else:
        F_B = 0
    F = F_B + F_E
    return F

def WFA(r, rmin, rmax, epsilon, rc, sigma):
    V = P_WF(r, epsilon, rc, sigma)
    F = F_WF(r, epsilon, rc, sigma)
    return (V, F)
def PACS(rcc, rmin, rmax,epsilon,Psi_p,Psi_n,r_p,r_n,sigma,L,DL):
    V = P_PACS(epsilon,Psi_p,Psi_n,r_p,r_n,DL,sigma,L,rcc)
    F = F_PACS(epsilon,Psi_p,Psi_n,r_p,r_n,DL,sigma,L,rcc)
    return (V, F)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def make_cluster(compression, sizeratio, patchsize, rotate = False):
    lp = 1.0  # 0.650/2.0   # lattice parameter
    # Define Bravais primitive lattice vectors for FCC lattice
    a = np.zeros(9).reshape(3, 3)
    a[0] = lp*0.5*np.array([0, 1, 1])
    a[1] = lp*0.5*np.array([1, 0, 1])
    a[2] = lp*0.5*np.array([1, 1, 0])
    
    
    
    axis = [0,0,1]
    theta = np.pi/2.0
    # Define first 4-sphere tetrahedral cluster in terms of FCC lattice vectors
    
    cd0 = np.zeros(12).reshape(4, 3)
    cd0[0] = 0.5*(a[0] + a[1] + a[2])  # cluster particle 0
    cd0[1] = 0.5*(a[1] + a[2])         # cluster particle 1
    cd0[2] = 0.5*(a[0] + a[2])         # cluster particle 2
    cd0[3] = 0.5*(a[0] + a[1])         # cluster particle 3
    
    # Define 4-sphere patches of second tetrahedral cluster
    ps1 = np.zeros(12).reshape(4, 3)
    ps1[0] = 0.5*(a[0] + a[1] + a[2])  # cluster particle 0
    ps1[1] = 0.5*(a[1] + a[2])         # cluster particle 1
    ps1[2] = 0.5*(a[0] + a[2])         # cluster particle 2
    ps1[3] = 0.5*(a[0] + a[1])         # cluster particle 3
    
    # Make vector to displace cluster from the origin to second diamond sphere position
    displace = .25*(a[0] + a[1] + a[2])
    
    
    diaTetParticle = np.linalg.norm(cd0[2]-cd0[1])
    tspce = np.linalg.norm(a[0])
    print(tspce)
    
    
    
    
    cd0Center = np.average(cd0, 0)  # cluster center of mass (cm)
    cd0 -= cd0Center                # move cluster cm to (0, 0, 0)
    ps1Center = np.average(ps1, 0)
    ps1 -= ps1Center
    
    #Define second 4-sphere tetrahedral cluster
    cd1 = np.zeros(12).reshape(4, 3)
    cd1[0] = np.dot(rotation_matrix(axis,theta),cd0[0])
    cd1[1] = np.dot(rotation_matrix(axis,theta),cd0[1])
    cd1[2] = np.dot(rotation_matrix(axis,theta),cd0[2])
    cd1[3] = np.dot(rotation_matrix(axis,theta),cd0[3])  
    
    # Define 4-sphere patches for first tetrahedral cluster
    ps0 = np.zeros(12).reshape(4, 3)
    ps0[0] = np.dot(rotation_matrix(axis,theta),cd0[0])
    ps0[1] = np.dot(rotation_matrix(axis,theta),cd0[1])
    ps0[2] = np.dot(rotation_matrix(axis,theta),cd0[2])
    ps0[3] = np.dot(rotation_matrix(axis,theta),cd0[3])
     
    cd0 += displace
    dis_touch = np.linalg.norm(cd0[3]-cd1[0])
    #r_cluster = np.linalg.norm(cd0[1])+0.5*diaTetParticle
    cd0 -= displace
    
    # compress tetrahedral clusters
    cd0 *= compression
    cd1 *= compression
    cd0 += displace
    dis_comp = np.linalg.norm(cd0[3]-cd1[0])
    expand_factor = dis_comp/dis_touch
    
    cd0 -= displace
    
    # Expand so that all spheres are touching
    cd0 *= expand_factor
    cd1 *= expand_factor
    diaTetParticle *= expand_factor
    
    # Move cluster to position
    cd0 += displace
    
    
    # Determine radius of spheres that will make up the shell
    lobeRadius = diaTetParticle/2.0
    
    # Define 2-sphere  basis of diamond lattice in terms of the FCC
    # primitive lattice vectors, these are used to show the radius at which patches are touching
    d0 = np.zeros(6).reshape(2, 3)
    d0[0] = 0*(a[0] + a[1] + a[2])
    d0[1] = .25*(a[0] + a[1] + a[2])
    
    diaDiamondParticle = np.linalg.norm(d0[1]-d0[0])
    
    diaRadius = diaDiamondParticle/2.0
    
    
    
    
    # Check shrinkage calculation
    
    
    
    axis = [0,0,1]
    theta = np.pi/2.0
    
    
    #p0 += p0Center                # move cluster cm back to original position
    
    
    
    
    # parameter for the extent of the patches relative to the radius at which they touch
    szratio4 = sizeratio
    # patchsz determines size of the patch spheres relative to the the extent of the patches
    #patchsz4 = patchsize
    
    # Calculate radius of patch spheres

    # Old Way
    #extent4 = szratio4*lobeRadius
    #patchRadius4 = extent4*patchsz4
    
    extent4 = szratio4*lobeRadius
    patchRadius4 = patchsize*lobeRadius
    #move patches to position such that they are touching
    patchlen4 = np.linalg.norm(ps0[0])
    
    patchtarget4 = extent4 - patchRadius4
    patchcomp4 = patchtarget4/patchlen4
    
    ps0 *= patchcomp4
    
    # move patches to cluster
    ps0 += displace
    edge = 0
    # repeat for origin cluster
    patchlen4 = np.linalg.norm(ps1[0])
    patchtarget4 = extent4 - patchRadius4
    patchcomp4 = patchtarget4/patchlen4
    ps1 *= patchcomp4
    if rotate == True:
        axfin = [0,0,1]
        axinit = [-1,-1,-1]
        axinit = axinit/np.linalg.norm(axinit)
        theta = np.arccos(np.dot(axinit,axfin))
        axis = np.cross(axinit,axfin)
        print(axis)
        
        cd1[0] = np.dot(rotation_matrix(axis,theta),cd1[0])
        cd1[1] = np.dot(rotation_matrix(axis,theta),cd1[1])
        cd1[2] = np.dot(rotation_matrix(axis,theta),cd1[2])
        cd1[3] = np.dot(rotation_matrix(axis,theta),cd1[3])  
        
    
        ps1[0] = np.dot(rotation_matrix(axis,theta),ps1[0])
        ps1[1] = np.dot(rotation_matrix(axis,theta),ps1[1])
        ps1[2] = np.dot(rotation_matrix(axis,theta),ps1[2])
        ps1[3] = np.dot(rotation_matrix(axis,theta),ps1[3])
        edge = cd1[0]-cd1[1]
        axfin = [1,0,0]
        axinit = edge
        axinit = axinit/np.linalg.norm(axinit)
        axfin = axfin/np.linalg.norm(axfin)
        theta = np.arccos(np.dot(axinit,axfin))
        axis = np.cross(axinit,axfin)
        cd1[0] = np.dot(rotation_matrix(axis,theta),cd1[0])
        cd1[1] = np.dot(rotation_matrix(axis,theta),cd1[1])
        cd1[2] = np.dot(rotation_matrix(axis,theta),cd1[2])
        cd1[3] = np.dot(rotation_matrix(axis,theta),cd1[3])  
        
    
        ps1[0] = np.dot(rotation_matrix(axis,theta),ps1[0])
        ps1[1] = np.dot(rotation_matrix(axis,theta),ps1[1])
        ps1[2] = np.dot(rotation_matrix(axis,theta),ps1[2])
        ps1[3] = np.dot(rotation_matrix(axis,theta),ps1[3])

        
    lobes = list(cd1)
    patches = list(ps1)
    patchRadius = patchRadius4
    
    
    return lobes, patches, lobeRadius, patchRadius,extent4/diaRadius


sizeratio =  float(os.environ.get('x'))
print(sizeratio)
Factor = float(os.environ.get('Factor'))

comp_arg=  int(os.environ.get('COMP'))

if comp_arg == 7:
    PS_Dict = {'1.15':'1.096', '1.16':'1.091', '1.17':'1.088', '1.18':'1.083', '1.19':'1.077', '1.20': '1.068', '1.21':'1.050' , '1.22':'1.015' , '1.23': '0.896', '1.24':'0.776', '1.25':'0.681', '1.26':'0.608', '1.27':'0.545', '1.28':'0.498', '1.29':'0.450', '1.30':'0.419', '1.31':'0.393', '1.32':'0.366', '1.33':'0.342', '1.34':'0.328', '1.35':'0.312', '1.36':'0.299', '1.37':'0.289', '1.38':'0.279','1.39':'0.271', '1.40':'0.265'}
elif comp_arg == 8:
    PS_Dict = {'1.15':'0.957', '1.16':'0.911', '1.17':'0.808', '1.18':'0.687', '1.19':'0.606', '1.20': '0.539', '1.21':'0.491' , '1.22':'0.440' , '1.23': '0.406', '1.24':'0.375', '1.25':'0.351', '1.26':'0.3333', '1.27':'0.314', '1.28':'0.300', '1.29':'0.286', '1.30':'0.279', '1.31':'0.269', '1.32':'0.263', '1.33':'0.259', '1.34':'0.255', '1.35':'0.251', '1.36':'0.250', '1.37':'0.248'}
elif comp_arg == 6:
    PS_Dict = {'1.15':'1.15', '1.16':'1.15', '1.17':'1.15', '1.18':'1.15', '1.19':'1.14', '1.20': '1.14', '1.21':'1.14' , '1.22':'1.13' , '1.23': '1.12', '1.24':'1.11', '1.25':'1.07', '1.26':'0.950', '1.27':'0.836', '1.28':'0.742', '1.29':'0.663', '1.30':'0.595', '1.31':'0.539', '1.32':'0.492', '1.33':'0.454', '1.34':'0.420', '1.35':'0.399', '1.36':'0.370', '1.37':'0.350','1.38':'0.333','1.39':'0.320', '1.40':'0.306'}
elif comp_arg == 9:
    PS_Dict = {'1.15':'0.376', '1.16':'0.274', '1.17':'0.268', '1.18':'0.264', '1.19':'0.263', '1.20': '0.257', '1.21':'0.257' , '1.22':'0.253' , '1.23': '0.252', '1.24':'0.250', '1.25':'0.249', '1.26':'0.249', '1.27':'0.248', '1.28':'0.247'}# '1.29':'0.247', '1.30':'0.247', '1.31':'0.247', '1.32':'0.247', '1.33':'0.246', '1.34':'0.246', '1.35':'0.247', '1.36':'0.245', '1.37':'0.247','1.38':'0.245','1.39':'0.247', '1.40':'0.246'}
compression = comp_arg / 10.0

patchsize = float((PS_Dict.get(os.environ.get('x'))))


name = 'WFtest_C60_SR'+str(sizeratio)+'.gsd'
lobes, patches, lobeRadius, patchRadius,extent_ratio = make_cluster(compression,sizeratio,patchsize)

ext = extent_ratio
d_C = 2*patchRadius #Size of the sphere that acts a patch
d_B = 2*lobeRadius #Size of the Lobes
d_Ce = 0.2 #Dummy center-of-mass particle


#Universal Constant
kT = 4.11e-21
ep = 80*8.85419e-12
Na = 6.0221409e23
e = 1.602176634e-19

sigma = d_C + d_B*0.02
rc = Factor * sigma
WF_width = rc-sigma
sigma2 = d_B*1.02
rc2 =1.02 * sigma2

sig = 0.09*(1000)**2/d_B**2 #in um^-2
Psi_n_patch = 0
Psi_n_sat = 0 #Not in use
Psi_binder = 0
L = 0.01*d_B # in um

C = 0.0015*1000 #moles per m^3
DL = ((ep*kT)/(e**2*Na*2*C))**(1/2)

hoomd.context.initialize()

uc = hoomd.lattice.unitcell(N=1,
                                a1=[1.0, 0, 0],
                                a2=[0, 1.0, 0],
                                a3=[0, 0, 2.5],
                                dimensions=3,
                                position=[[0, 0, 0]],
                                type_name=['Ce'],
                                mass=[1.0],
                                charge=[0],
                                moment_inertia=[[1, 1, 1]],
                                diameter=[d_Ce])

#########

patch_closeness = np.linalg.norm(patches[0])
if patch_closeness > 2*WF_width:
    particle_positions = lobes + patches
    
    particle_type =['B']*4 + ['C']*4
    
    
    particle_diameter =[d_B]*4 + [d_C]*4
    
    print('Four Sphere Patch')
    
else:
    patches = [np.array([0,0,0])]
    d_C = sizeratio*d_B
    
    particle_positions = lobes + patches
    
    particle_type =['B']*4 + ['C']
    
    
    particle_diameter =[d_B]*4 + [d_C]
    print('Single Sphere Patch')


#------HOOMD Blue Related Code
system = hoomd.init.create_lattice(unitcell=uc, n=1)




# create the constituent particle type
# system.particles.types.add('A')
system.particles.types.add('B')
system.particles.types.add('C')
# define the rigid body type
rigid = hoomd.md.constrain.rigid()
rigid.set_param('Ce', positions=particle_positions,types =particle_type, diameters=particle_diameter)

# .. create pair.lj() ..
# create constituent particles and run
rigid.create_bodies()


system.replicate(nx=4, ny=4, nz=150)




typeCe = hoomd.group.type(type='Ce')
#typeA = hoomd.group.type(type='A')
typeB = hoomd.group.type(type='B')
typeC = hoomd.group.type(type='C')

L_z0 = system.box.Lz

system.particles.types.add('tB')
system.particles.types.add('tC')

tlobes, tpatches, tlobeRadius, tpatchRadius, edge = make_cluster(compression,sizeratio,patchsize, rotate = True)

template_type = ['tB']*4 + ['tC']*4

template_diameter = particle_diameter
system.particles.types.add('tCe')

temp_num = 3
temp_shift = np.array([0,0,-L_z0/2.0 + .1 + (d_B/2.0)*2.0 - tlobes[0][2]])

temploc = [[0,0,0],[.707*ext,0,0],[.707*ext/2, -.612*ext,0],[-.707*ext,0,0],[-.707*ext/2,-.612*ext,0],[-.707*ext/2,.612*ext,0],[.707*ext/2,.612*ext,0]]

for i in range(len(temploc)):
    t = system.particles.add('tCe')
    pt = system.particles[t]
    pt.type = 'tCe'
    pt.diameter = d_Ce
    pt.mass = 1.0
    pt.position = temp_shift+temploc[i]
    for j in range(len(tlobes)):
        t = system.particles.add('B')
        pt = system.particles[t]
        pt.type = 'B'
        pt.diameter = d_B
        pt.mass = 1.0
        pt.position = tlobes[j]+temp_shift+temploc[i]
    for k in range(len(tpatches)):
        t = system.particles.add('tC')
        pt = system.particles[t]
        pt.type = 'tC'
        pt.diameter = d_C
        pt.mass = 1.0
        pt.position = tpatches[k]+temp_shift+temploc[i]    

 

typetCe = hoomd.group.type(type='tCe')
typetB = hoomd.group.type(type='tB')
typetC = hoomd.group.type(type='tC')
all = hoomd.group.all()

walls = hoomd.md.wall.group()

walls.add_plane(origin=(0, 0, -L_z0/2.0+.1), normal=(0, 0, 1))
walls.add_plane(origin=(0, 0, L_z0/2.0-.1), normal=(0, 0, -1))
lj = hoomd.md.wall.lj(walls, r_cut = 0.5)
lj.force_coeff.set('C', epsilon = 3, alpha = 0, sigma = d_B/2.0, r_cut = 2.0*d_B)
lj.force_coeff.set('Ce', epsilon = 3, alpha = 0, sigma = d_B/2.0, r_cut = 2.0*d_B)
lj.force_coeff.set('B', epsilon = 3, alpha = 0, sigma = d_B/2.0, r_cut = 2.0*d_B)
lj.force_coeff.set('tCe', epsilon = 3, alpha = 0, sigma = d_B/2.0, r_cut = 2.0*d_B)
lj.force_coeff.set('tC', epsilon = 3, alpha = 0, sigma = d_B/2.0, r_cut = 2.0*d_B)
lj.force_coeff.set('tB', epsilon = 3, alpha = 0, sigma = d_B/2.0, r_cut = 2.0*d_B)





nl = hoomd.md.nlist.cell()
n = 16
table = hoomd.md.pair.table(width=10000, nlist=nl)
Coff = 2**(1.0 / n) + (np.log(2) / n) * (2.0 * np.log(10))**(1.0 / 2)
# attractive particles Sharp attraction, soft repulsion
table.pair_coeff.set('C', 'C', func=WFA, rmin=0.95*sigma,rmax=rc
                         , coeff=dict(epsilon = 10, rc = rc, sigma = sigma))
table.pair_coeff.set('C', 'tC', func=WFA, rmin=0.95*sigma,rmax=rc
                         , coeff=dict(epsilon = 10, rc = rc, sigma = sigma))


table.pair_coeff.set('B', 'B', func=WFA, rmin=0.95*sigma2,rmax=rc2
                       , coeff=dict(epsilon = 1.0, rc = rc2, sigma = sigma2))
table.pair_coeff.set('B', 'tB', func=WFA, rmin=0.95*sigma2,rmax=rc2
                       , coeff=dict(epsilon = 1.0, rc = rc2, sigma = sigma2))
# repulsive part Soft
table.pair_coeff.set('C', 'B', func=PACS, rmin=(d_C+d_B)/2+0.001*d_B, rmax=(d_C+d_B)/2+d_B*0.05,coeff = dict( epsilon = ep,Psi_p = Psi_n_patch, Psi_n = Psi_n_sat, r_p = d_C/2, r_n = d_B/2, sigma = sig, L = L, DL = DL))
n =48

table.pair_coeff.set('Ce', 'Ce', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))
table.pair_coeff.set('B', 'Ce', func=ljR, rmin=((d_B + d_Ce) / 2) * 0.9,
                         rmax=((d_B + d_Ce) / 2) * (2**(1. / n)), coeff=dict(epsilon=1, sigma=(d_B + d_Ce) / 2))
table.pair_coeff.set('C', 'Ce', func=ljR, rmin=((d_C + d_Ce) / 2) * 0.9,
                         rmax=((d_C + d_Ce) / 2) * (2**(1. / n)), coeff=dict(epsilon=1, sigma=(d_C + d_Ce) / 2))

table.pair_coeff.set('tCe', 'tCe', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))
table.pair_coeff.set('B', 'tCe', func=ljR, rmin=((d_B + d_Ce) / 2) * 0.9,
                         rmax=((d_B + d_Ce) / 2) * (2**(1. / n)), coeff=dict(epsilon=1, sigma=(d_B + d_Ce) / 2))
table.pair_coeff.set('C', 'tCe', func=ljR, rmin=((d_C + d_Ce) / 2) * 0.9,
                         rmax=((d_C + d_Ce) / 2) * (2**(1. / n)), coeff=dict(epsilon=1, sigma=(d_C + d_Ce) / 2))
table.pair_coeff.set('Ce', 'tCe', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))

table.pair_coeff.set('tCe', 'tC', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))
table.pair_coeff.set('B', 'tC', func=ljR, rmin=((d_B + d_Ce) / 2) * 0.9,
                         rmax=((d_B + d_Ce) / 2) * (2**(1. / n)), coeff=dict(epsilon=1, sigma=(d_B + d_Ce) / 2))

table.pair_coeff.set('Ce', 'tC', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))

table.pair_coeff.set('tCe', 'tB', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))

table.pair_coeff.set('C', 'tB', func=ljR, rmin=((d_C + d_Ce) / 2) * 0.9,
                         rmax=((d_C + d_Ce) / 2) * (2**(1. / n)), coeff=dict(epsilon=1, sigma=(d_C + d_Ce) / 2))
table.pair_coeff.set('Ce', 'tB', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))

table.pair_coeff.set('tC', 'tB', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))
table.pair_coeff.set('tB', 'tB', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))
table.pair_coeff.set('tC', 'tC', func=ljR, rmin=d_Ce * 0.9, rmax=d_Ce *
                         (2**(1. / n)), coeff=dict(epsilon=1, sigma=d_Ce))

NB = len(typeB)



PhiV = 0.40
Vol_Part = (4. / 3) * math.pi * ((NB * d_B * d_B * d_B) / (2.**3))
Vol_Box = system.box.Lx * system.box.Ly * system.box.Lz
Vol_Final = Vol_Part / PhiV
L_ini = system.box.Lx
L_final = (Vol_Final)**(1.0 / 3.0)
L_final = system.box.Lz/4.0
Const = hoomd.md.force.constant(fx=0, fy=0, fz=-0.025, group=all)
all = hoomd.group.all();
hoomd.md.integrate.mode_standard(dt=0.00005);

randseed = random.randrange(100000)
integrator = hoomd.md.integrate.langevin(group = typeCe, kT=1,seed=randseed)
integrator.set_gamma_r('Ce', gamma_r = 3.0)

dump=hoomd.dump.gsd(name,period=5e5,group=all, overwrite =True)
hoomd.run(4e8);
