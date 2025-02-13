#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:45:27 2024

@author: jbg6127
"""



'''
This code uses the Blender Python API to make an FCC lattice of spheres in Blender.
To run it, open Blender, navigate to the Text Editor, open this file, and hit the play button.
'''

import numpy as np
import math
import bpy
from bpy_extras.node_shader_utils import PrincipledBSDFWrapper



def makeMaterial(name, diffuse, roughness):
    # Choose blender material color and properties
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    principled = PrincipledBSDFWrapper(mat, is_readonly=False)
    principled.base_color = diffuse
    principled.roughness = roughness
    return mat


def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)


# Select all objects in current blender scene
bpy.ops.object.select_all(action='SELECT')


# remove everything
bpy.ops.object.delete()

# Make material colors
red = makeMaterial('Red', (1, 0, 0), 0.5)
blue = makeMaterial('Blue', (0.111, 0.647, 0.8), 0.5)
yellow = makeMaterial('Yellow', (1, 0.880, 0.176), 0.5)
green = makeMaterial('Green', (0, 1, 0), 0.5)
white = makeMaterial('White', (0.8, 0.8, 0.8), 0.0)
purple = makeMaterial('Purple', (0.372, 0.068, 0.8), 0.0)

seg = 32
ring = 16

radrat = 1.33

partnum = 0
sr2 = 1/np.sqrt(2)
sr3 = 1/np.sqrt(3)
pos_spheres = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                        [1,1,0],[1,0,1],[0,1,1],[1,1,1],
                       [0.5,0,0.5],[0.5,0.5,0],[0,0.5,0.5],
                       [0.5,0.5,1.0],[0.5,1.0,0.5],[1.0,0.5,0.5]])




    


d = np.linalg.norm(np.array([0.5, 0.5, 0]))
r = d/2.0


for pos in pos_spheres:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=seg, ring_count=ring, location=pos, radius=r)
    # Name the object
    bpy.context.active_object.name = 'Sphere{:.0f}'.format(partnum)
    # Smooth the object for viewing
    bpy.ops.object.shade_smooth()
    # Set the material color
    setMaterial(bpy.context.object, white)
    partnum += 1

