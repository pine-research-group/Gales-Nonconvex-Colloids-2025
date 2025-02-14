#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:00:50 2022

@author: minjaekim
"""
import gsd.hoomd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gsd
import glob
import numpy as np
import math
import os.path

def p_dist(cen1,cen2,sx,sy):
    cen1 = np.array(cen1)
    cen2 = np.array(cen2)
    distx = (cen1[0] - cen2[0])
    distx = distx - sx*np.floor(distx/sx + 0.5)
    disty = (cen1[1] - cen2[1])
    disty = disty - sy*np.floor(disty/sy + 0.5)
    distz = (cen1[2] - cen2[2])
    
    dist_vec = np.array([distx,disty,distz]).T
    return dist_vec
    
def torsion_angles(file_name, frame,B):
    
    ''' This funtion reads a frame from the output simulation, finds nearest neighbors and then calculates the dihedral angles of the bonds '''
    badnum = 0
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
    
    smallx = []
    smally = []
    smallz = []
    
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
        other_clusters = (list(cluster_position[:k])+list(cluster_position[k+1:]))
        patch1[0] = (patch_position[int(4*k),:]) 
        patch1[1] = (patch_position[int(4*k+1),:]) 
        patch1[2] = (patch_position[int(4*k+2),:]) 
        patch1[3] = (patch_position[int(4*k+3),:]) 
        
        other_patches = (list(patch_position[:int(4*k)]) + list(patch_position[int(4*k+3+1):]))
        patches_bound = 0
        indices = []
        #Calculate center to center distances between the kth cluster and all other clusters
        cluster = np.array(cluster)
        patch1 = np.array(patch1)
        for p in range(len(patch1)):
            if cluster[0] - patch1[p][0] > 2.0:
                patch1[p][0] += 4.0
            if cluster[0] - patch1[p][0] < -2.0:
                patch1[p][0] -= 4.0    
            if cluster[1] - patch1[p][1] > 2.0:
                patch1[p][1] += 4.0
            if cluster[1] - patch1[p][1] < -2.0:
                patch1[p][1] -= 4.0   
                
            patch_extent = np.linalg.norm(patch1[p] - cluster)
            #print(patch_extent)
        other_clusters = np.array(other_clusters)
        c2cx = (cluster[0] - other_clusters[:,0])
        c2cx = c2cx - 4.0*np.floor(c2cx/4.0 + 0.5)
        c2cy = (cluster[1] - other_clusters[:,1])
        c2cy = c2cy - 4.0*np.floor(c2cy/4.0 + 0.5)
        c2cz = (cluster[2] - other_clusters[:,2])
        
        c2cvec = np.vstack([c2cx,c2cy,c2cz]).T
       
        distance = (np.sum(c2cvec**2, axis = 1, dtype = float))**(0.5)
        c2cindices = np.asarray(np.where(distance <= cutoff))
        c2cindices = c2cindices[0]
        for patch in patch1:
            
            patch = np.array(patch)

            if cluster[0] - patch[0] > 2.0:
                patch[0] += 4.0
            if cluster[0] - patch[0] < -2.0:
                patch[0] -= 4.0    
            if cluster[1] - patch[1] > 2.0:
                patch[1] += 4.0
            if cluster[1] - patch[1] < -2.0:
                patch[1] -= 4.0   
                
            patch_extent = np.linalg.norm(patch - cluster)
            #print(patch_extent)
    
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
        
        
        if len(indices)>4:
            print('yes')
        #Pass if its only neighbor is itself
        if indices.size == 0:
            pass
        else:
            
            #Loop through the kth clusters nearest neighbors to calculate torsion angles
            for nebr in indices:
                
                #Only calculate torion angles for bonds that haven't already been counted
                if nebr > k:
                    index = nebr
                    cluster2 = (cluster_position[index,:]) 

                    #Define center to center vector of the bond

                    
                    c2c = p_dist(cluster,cluster2,4.0,4.0)
                    
                    if np.linalg.norm(c2c)>1.0:
                        print('bad')
                    
                    c2c_unit = c2c/np.linalg.norm(c2c)
                    
                    #Find all the patches on the nearest neighbor
                    patch2[0] = (patch_position[int(4*index),:]) 
                    patch2[1] = (patch_position[int(4*index+1),:]) 
                    patch2[2] = (patch_position[int(4*index+2),:]) 
                    patch2[3] = (patch_position[int(4*index+3),:]) 
                    
                    cluster2 = np.array(cluster2)
                    patch2 = np.array(patch2)
                    for p in range(len(patch2)):
                        if cluster2[0] - patch2[p][0] > 2.0:
                            patch2[p][0] += 4.0
                        if cluster2[0] - patch2[p][0] < -2.0:
                            patch2[p][0] -= 4.0    
                        if cluster2[1] - patch2[p][1] > 2.0:
                            patch2[p][1] += 4.0
                        if cluster2[1] - patch2[p][1] < -2.0:
                            patch2[p][1] -= 4.0   
                            
                        patch_extent = np.linalg.norm(patch2[p] - cluster2)
                        #print(patch_extent)
                    
                    #Throw out the patches that are touching
                    nebrdelete = False
                    for p_1 in range(len(patch1)):

                        for p_2 in range(len(patch2)):
                            
                            
                            patch = np.array(patch1[p_1])
                            other_patch = np.array(patch2[p_2])
                            
                            dx = (patch[0] - other_patch[0])    
                            dx = dx - 4.0*np.floor(dx/4.0 + 0.5)
                    
                            dy = (patch[1] - other_patch[1])
                            dy = dy - 4.0*np.floor(dy/4.0 + 0.5)        
                            
                            dz = (patch[2] - other_patch[2])
                            
                            dvec = np.vstack([dx,dy,dz]).T
                            
                            
                            patch_distance = (np.sum(dvec**2, axis = 1, dtype = float))**(0.5)
                            #print(np.linalg.norm(patch1[p_1]-patch2[p_2]))
                            if patch_distance < bond_cutoff:
        
                                patch1new = np.delete(patch1, p_1, axis = 0)
                                patch2new = np.delete(patch2, p_2, axis = 0)
                                
                                nebrdelete = True

                    
                    dummy_vector1 = patch1new - cluster
                        
                    dummy_vector2 = patch2new - cluster2
    
    
                    #Choose one patch on the kth cluster to be the reference patch that we will measure angles 
                    reference_patch =dummy_vector1[1] 
                    
                    #Project the reference patch vector onto the plane defined by the center to center axis
                    ref_xy = np.dot(c2c_unit,reference_patch)
                    
                    reference_patch = dummy_vector1[1] - ref_xy*c2c_unit
                    reference_patch = p_dist(dummy_vector1[1],ref_xy*c2c_unit,4.0,4.0)
                    reference_patch = reference_patch/np.linalg.norm(reference_patch)
                    
                    
                    #For each of the non-bond patches' vectors calculate the torsion angle relative to reference patch
                    for j in range(len(dummy_vector2)):
                        
                        #Project the patch vector onto the plane defined by the center to center axis
                        ref_zmag = np.dot(dummy_vector2[j],c2c_unit)
                        dummy_v2 = dummy_vector2[j] - ref_zmag*c2c_unit
                        dummy_v2 = dummy_v2/np.linalg.norm(dummy_v2)
                        
                        
                        #Calculate the angle between the projected patch on the neighbor and the projected reference patch on the kth cluster
                        angle = np.arccos(np.dot(dummy_v2,reference_patch))
                        angle = angle
                        if angle*180/math.pi > 82 and angle*180/math.pi < 92:
                            #print(badnum)
                            badnum+=1
                            smallx.append(cluster[0])
                            smally.append(cluster[1])
                            smallz.append(cluster[2])
                        #Keep track of them all
                        tors_angles = tors_angles + [angle]

    
    tors_angles = np.asarray(tors_angles, dtype = float)
    
    tors_angles = tors_angles * 180/math.pi
    return tors_angles, smallx, smally, smallz

A = "650" #Compression Ratio

size_ratios = np.linspace(1.22,1.40,19)


for size_ratio in size_ratios:
    B = '%.2f'%size_ratio
    experiment_name = "WFtest_"
    parameter = "C" + A + "SR" + B + "Factor1.02"
    file_name = glob.glob("paper_gsds/C" + A + "_10kT/" + parameter + "/*.gsd")[0]
   
    
    #Reading gsd file
    traj = gsd.hoomd.open(file_name, mode = 'rb')
    N_frames = len(traj)
    frames = np.linspace(N_frames-2,0,int(N_frames/2))

    
    bonds = []
    alltors = []
    torsframes = []
    for frame in frames:
    
        tors_angles, sx, sy, sz = torsion_angles(file_name, int(frame),float(B))
        
        if frame == frames[0]:
            plt.figure(1)
            plt.hist(tors_angles,bins = 60,label = str(frame))
                            
            plt.ylabel('Number of Counts')
            plt.xlabel('Dihedral Angle(degrees)')
            plt.xlim([0,120])
            plt.legend(loc="upper right")
            
        
        bonds.append(len(tors_angles))
        
        alltors += list(tors_angles)
        
        flist = np.zeros(len(tors_angles)) + int(frame)
        torsframes += list(flist)
    
        
        print(frame)
    tors_data = np.array([alltors,torsframes])
    np.savetxt(parameter + "_torsion.txt", tors_data.T)
    
    bonds = np.array(bonds)
    bond_fraction = bonds
    #time = 0.32*frames
    
    bond_data = np.array([bond_fraction, frames])
    #np.savetxt(parameter + "_bondnum.txt", bond_data.T)
    


    
