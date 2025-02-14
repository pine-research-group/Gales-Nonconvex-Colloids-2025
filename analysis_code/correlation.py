#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:00:50 2022

@author: minjaekim
"""
import gsd.hoomd
from matplotlib import pyplot as plt
import gsd
import glob
import numpy as np
import math
import os.path

def nearest_nebrs(file_name, frame,B, radial_factor, dr, shell = True):
    
    ''' This funtion reads a frame from the output simulation, finds nearest neighbors and then calculates the dihedral angles of the bonds '''
    
    traj = gsd.hoomd.open(file_name, mode = 'rb')
    dummy = traj[frame] #using the last frame as a dummy frame to get some general simulation data
    N_particles = int(dummy.particles.N) - 7*9
    N_clusters = int(N_particles/9)
    N_patches = N_clusters*4
   
    ##Extract the wanted data
    positions = dummy.particles.position
    types = dummy.particles.typeid
    types_tag = dummy.particles.types
    radii = dummy.particles.diameter
    
    
    d_C = radii[6]
    d_B = radii[2]
    
    sigma = d_C + d_B*0.02
    rc = 1.02 * sigma

    cluster_position = np.zeros([int(N_clusters),3], dtype = float)  #[:,0] for Ce [:,1:4] for C (patches)
    patch_position = np.zeros([int(N_patches),3], dtype = float)
    lobe_position = np.zeros([int(N_patches),3], dtype = float)
    n = 0 #start dummy counter
    i = 0
    m = 0
    #Separate particles into cluster centers of mass and patches
    for k in range(0,N_particles):
        tag = types[k]
        if tag == 0:
            cluster_position[n,:] = positions[k]
            n = n+1
        elif tag == 1:
            lobe_position[m,:] = positions[k]
            m = m+1
            pass
        elif tag == 2:
            patch_position[i,:] = positions[k]
            i = i + 1
        else:
            pass
    lobe_distance = np.linalg.norm(lobe_position[0]-cluster_position[0])
    patch_distance = np.linalg.norm(patch_position[0]-cluster_position[0])
    #nearest neighbor cutoff
    nebr_cutoff = 2*(lobe_distance) + d_B + .04*d_B
    cutoff = radial_factor*d_B
    volume = 4/3*np.pi*cutoff**3
    if shell == True:
        volume -= 4/3*np.pi*(cutoff-dr)**3
    num_nebrs = []
    

    n = 0
    
    for k in range(0,N_clusters):
        
        #Pick out the kth cluster's center and its patches
        cluster =(cluster_position[k])
        other_clusters = np.array((list(cluster_position[:k]) + list(cluster_position[k+1:])))
        indices = []
        #Calculate center to center distances between the kth cluster and all other clusters
        dx = (cluster[0] - other_clusters[:,0])
        dx = dx - 4.0*np.floor(dx/4.0 + 0.5)

        dy = (cluster[1] - other_clusters[:,1])
        dy = dy - 4.0*np.floor(dy/4.0 + 0.5)        
        
        dz = (cluster[2] - other_clusters[:,2])
        
        dvec = np.vstack([dx,dy,dz]).T
        new_distance = (np.sum(dvec**2, axis = 1, dtype = float))**(0.5)
        if shell == True:
            shellindices = np.asarray(np.where((new_distance <= cutoff) & (new_distance >= (cutoff-dr))))
        else:
            shellindices = np.asarray(np.where((new_distance <= cutoff)))
        
        shellindices = shellindices[0]
        
        
        
            


        indices = np.array(indices)

        num_nebrs.append(len(shellindices))
    return num_nebrs, volume, nebr_cutoff

A = "70" #Compression Ratio

size_ratios = np.linspace(1.20,1.40,21)


for size_ratio in size_ratios:
    B = '%.2f'%size_ratio
    experiment_name = "WFtest_"
    parameter = "COMP" + A + "SR" + B + "Factor1.02"
    file_name = glob.glob("125kT_gsds/12.5kT_phase_diagram/" + parameter + "/*.gsd")[0]
   
    
    #Reading gsd file
    traj = gsd.hoomd.open(file_name, mode = 'rb')
    N_frames = len(traj)
    frames = np.linspace(N_frames-1,0,20)
    
    rs = np.linspace(1.0,5.0,500)
    
    frame = N_frames-1
    
    dummy = traj[frame]
    radii = dummy.particles.diameter
    micron = radii[2]
    bonds = []
    shell_rhos = []
    avg_rhos = []
    tot_nebrs, totvol, nebr_sphere = nearest_nebrs(file_name, int(frame),float(B), rs[-1], 0.02*micron, shell = False)
    local_rhos = tot_nebrs/totvol
    for r in rs:
        
        num_nebrs, vol, nebr_shell = nearest_nebrs(file_name, int(frame),float(B), r, 0.1*micron)
        
        
        shell_rhos = num_nebrs/vol
        
        
        norm_rhos = shell_rhos[local_rhos != 0]/local_rhos[local_rhos != 0]

        avg_rhos.append(np.mean(norm_rhos))
        

            
    data = np.array([rs, avg_rhos])
    data = data.T
    np.savetxt('20nmcorrelation' + parameter + '.txt',data)
    plt.figure(1)
    plt.plot(rs, avg_rhos, 'o', label = size_ratio)
    


shells = np.array([1.0,2.0,3.0])


#plt.vlines(nebr_shell*shells/micron, ymin = -1, ymax = 10, linestyles = 'dashed')
plt.legend()

plt.show()
    
    
    
        
        
