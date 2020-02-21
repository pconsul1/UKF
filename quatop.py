#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:40:37 2020

@author: poojaconsul
"""
import math
from  quaternion import Quaternion
import numpy as np
    
def rot_to_quat(w): 
    w = np.array(w)
    norm = np.linalg.norm(w)
    if(norm == 0):
        # print('norm is 0 in rot_to_quat function')
        return Quaternion(0, [0,0,0], "value")
        
    axis = w/norm
    q1 = Quaternion(norm, axis, "angle")
    q1 = q1.unit_quaternion()
    return q1
    
#verified
def quat_to_rot(q):
    q = Quaternion(q[0], q[1:4], "value")
    
    if(q.norm() == 0):
        return np.array([0,0,0])
    
    q = q.unit_quaternion()
    y = np.array(q.quaternion)
    q0 = y[0]
    angle = math.acos(q0)
    
    if(angle == 0):
        return np.array([0,0,0])
        # return np.array([y[1],y[2],y[3]])
    
    sin_angle = math.sin(angle)
    angular = (2*angle*y[1:4])/sin_angle #unit vector of angular velocity
    return angular


def quat_average(quaternions, current_state_quaternion, thresh = 0.005): #verified

    '''
    0.92707 +   0.02149i +   0.19191j +   0.32132k
    0.90361 + 0.0025836i +  0.097279j +   0.41716k
    0.75868 -   0.21289i +   0.53263j +   0.30884k
    '''
    '''
    quaternions = []
    quaternions.append(Quaternion(0.92707, [0.02149,   0.19191,   0.32132], "qu"))
    quaternions.append(Quaternion(0.90361, [0.0025836,  0.097279,   0.41716], "qu"))
    quaternions.append(Quaternion(0.75868, [-0.21289,   0.53263,   0.30884], "qu"))
    '''
    #Declare a random unit qauternion
    #estimated_mean = Quaternion(0.75868, [0.21289,   0.53263,   0.30884], "qu")
    estimated_mean = current_state_quaternion
    counter = 0   
    while(True):
        counter = counter+1
        ei_s = np.zeros((3,len(quaternions)))
        # ei_s = np.zeros((len(quaternions),3))
        for i in range(len(quaternions)):
         
            ei = quaternions[i].multiply(estimated_mean.inverse())
            # print("\n\nei ",ei.quaternion)
            ei_rotvec = quat_to_rot(ei.quaternion)
            # ei_s[i] = ei_rotvec
            ei_s[:,i] = ei_rotvec
            # print(ei_s[:,i])
            # print(ei_rotvec)
            
        # e = np.mean(ei_s,axis=0)
        e = np.mean(ei_s,axis=1)
        
        # print("e ", e)
        
        e_quaternion = rot_to_quat(e)
        
        # print(e_quaternion.quaternion)
        
        estimated_mean = e_quaternion.multiply(estimated_mean)  
        
        # print('estimate mean \n',estimated_mean.quaternion)
        
        
        if(np.linalg.norm(e)< thresh or counter>10):
            break
    # print('steps to converge ',counter)
    
    # return estimated_mean, ei_s, counter
    return estimated_mean, ei_s.T, counter