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

def neighbor_find(traj, frame,B):
    
    ''' This funtion reads a frame from the output simulation, finds nearest neighbors and then calculates the dihedral angles of the bonds '''
    

    dummy = traj[frame] #using the last frame as a dummy frame to get some general simulation data
    N_particles = int(dummy.particles.N) - 7*9
    N_clusters = int(N_particles/9)
    N_patches = N_clusters*4
   
    ##Extract the wanted data
    positions = dummy.particles.position
    types = dummy.particles.typeid
    types_tag = dummy.particles.types
    radii = dummy.particles.diameter
    
    neighbors = []
    d_C = radii[6]
    d_B = radii[2]
    
    sigma = d_C + d_B*0.02
    rc = 1.02 * sigma
    bond_cutoff = 1.02*rc
    cutoff = float(B)/100*0.18071*2 +  0.0275*sigma
    cluster_position = np.zeros([int(N_clusters),3], dtype = float)  #[:,0] for Ce [:,1:4] for C (patches)
    patch_position = np.zeros([int(N_patches),3], dtype = float)
    n = 0 #start dummy counter
    i = 0
    
    #Separate particles into cluster centers of mass and patches
    for k in range(0,N_particles):
        tag = types[k]
        if tag == 0:
            cluster_position[n,:] = positions[k]
            n = n+1
        elif tag == 1:
            pass
        elif tag == 2:
            patch_position[i,:] = positions[k]
            i = i + 1
        else:
            pass
    
    tors_angles = []
    
    #Analyze the bond angle
    patch1 = np.zeros([4,3], dtype =float)
    patch2 = np.zeros([4,3], dtype = float)
    n = 0
    
    for k in range(0,N_clusters):
        
        #Pick out the kth cluster's center and its patches
        cluster =(cluster_position[k])
        patch1[0] = (patch_position[int(4*k),:]) 
        patch1[1] = (patch_position[int(4*k+1),:]) 
        patch1[2] = (patch_position[int(4*k+2),:]) 
        patch1[3] = (patch_position[int(4*k+3),:]) 
        other_patches = (list(patch_position[:int(4*k)]) + list(patch_position[int(4*k+3+1):]))
        patches_bound = 0
        indices = []
        #Calculate center to center distances between the kth cluster and all other clusters
        distance = (np.sum((cluster - cluster_position)**2, axis = 1, dtype = float))**(0.5)
        c2cindices = np.asarray(np.where(distance <= cutoff))
        c2cindices = c2cindices[0]
        for patch in patch1:
            
            patch = np.array(patch)
            other_patches = np.array(other_patches)
            
            dx = (patch[0] - other_patches[:,0])
            dx = dx - 4.0*np.floor(dx/4.0 + 0.5)
    
            dy = (patch[1] - other_patches[:,1])
            dy = dy - 4.0*np.floor(dy/4.0 + 0.5)        
            
            dz = (patch[2] - other_patches[:,2])
            
            dvec = np.vstack([dx,dy,dz]).T
            patch_distance = (np.sum(dvec**2, axis = 1, dtype = float))**(0.5)
            
            
            
            patch_indices = np.asarray(np.where(patch_distance <= bond_cutoff))
            patch_indices = patch_indices[0]
            for thing in patch_indices:
                
                index = int(thing/4)
                if index>k:
                    index+=1
                indices.append(index)
            


        indices = np.array(indices)
        neighbors.append(len(indices))
        
        if len(indices)>4:
            print('yes')

    return neighbors

A = "725" #Compression Ratio

size_ratios = np.linspace(1.23,1.40,18)
size_ratios = [1.39]
for size_ratio in size_ratios:
    f_ndx = 0

    B = '%.2f'%size_ratio
    experiment_name = "WFtest_"
    parameter = "C" + A + "0SR" + B + "Factor1.02"
    file_name = glob.glob("paper_gsds/C" + A + "_10kT/" + parameter + "/*.gsd")[0]
   
    
    #Reading gsd file
    traj = gsd.hoomd.open(file_name, mode = 'rb')
    N_frames = len(traj)
    frames = np.linspace(N_frames-2,0,int(N_frames/2))

    bond_data = np.empty([len(frames),2400])
    
    bonds = []
    allbonds = []
    bondframes = []
    for frame in frames:
    
        neighbors = neighbor_find(traj, int(frame),float(B))
        
        if frame == frames[0]:
            plt.figure(1)
            plt.hist(neighbors,bins = 10,label = str(frame))
                            
            plt.ylabel('Number of Counts')
            plt.xlabel('Bond Number')
            
            plt.legend(loc="upper right")
            
        bond_data[f_ndx] = neighbors
        f_ndx += 1
        '''
        allbonds += list(neighbors)
        
        flist = np.zeros(len(neighbors)) + int(frame)
        bondframes += list(flist)
        '''
        
        print(frame)
    #bonds_data = np.array([allbonds,bondframes])
    np.savetxt(parameter + "_bonds.txt", bond_data)
    

    plt.show()
