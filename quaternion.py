#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:19:46 2020

@author: poojaconsul
"""
import math
import numpy as np

class Quaternion:
    def __init__(self, angle, axis, typ = "angle"):
        if typ == "angle":
            self.q0 = math.cos(angle/2)
            sin_angle = math.sin(angle/2)
            self.q1 = sin_angle * axis[0] #x
            self.q2 = sin_angle * axis[1] #y
            self.q3 = sin_angle * axis[2] #y
            
        elif typ == "value":
            self.q0 = angle
            self.q1 = axis[0]
            self.q2 = axis[1]
            self.q3 = axis[2]
        
        else:
            raise ValueError("Incorrect type to initialize Quaternion object")
            
        self.quaternion = [self.q0, self.q1, self.q2, self.q3]
    
    def conjugate(self):
        return Quaternion(self.q0, [-self.q1, -self.q2, -self.q3], "value")
    
    def norm(self):
        return math.sqrt(np.sum(np.array([i**2 for i in self.quaternion])))
        
    def add(self, q2):
        add = []
        for i in range(4):
            add.append(self.quaternion[i] + q2.quaternion[i])
        return add
    
    def unit_quaternion(self):
        norm = self.norm()
        if norm == 0:
            return Quaternion(0, [0,0,0], "value")
        norm_div = np.array(self.quaternion)/norm
        return Quaternion(norm_div[0] , norm_div[1:], "value")
    
    def multiply(self, q2):
        # mult = [self.q0 * q2.q0 - np.dot(self.quaternion[1:], q2.quaternion[1:])] +\
        #     list(self.q0 * np.array(q2.quaternion[1:]) +\
        #           q2.q0 * np.array(self.quaternion[1:]) +\
        #         np.cross(self.quaternion[1:], q2.quaternion[1:]))
        mult = [0,0,0,0]
        w0, x0, y0, z0 = self.q0, self.q1, self.q2, self.q3
        w1, x1, y1, z1 = q2.q0, q2.q1, q2.q2, q2.q3
        mult[0] = (-x1 * x0) - (y1 * y0) - (z1 * z0) + (w1 * w0)
        mult[1] = (x1 * w0) - (y1 * z0) + (z1 * y0) + (w1 * x0)
        mult[2] = (x1 * z0) + (y1 * w0) - (z1 * x0) + (w1 * y0)
        mult[3] = - (x1 * y0) + (y1 * x0) + (z1 * w0) + (w1 * z0)
        return Quaternion(mult[0], mult[1:], "value")
    
    def inverse(self):
        conj = self.conjugate()
        norm = self.norm()
        if norm == 0:
            return Quaternion(0, [0,0,0], "value")
        norm_sq = norm**2
        conj_div = np.array(conj.quaternion)/norm_sq
        inverse = Quaternion(conj_div[0], conj_div[1:], "value")
        return inverse
    
    def quat_to_eular(self):
        r = math.atan2(2*(self.q0*self.q1+self.q2*self.q3), \
                  1 - 2*(self.q1**2 + self.q2**2))
        p = math.asin(2*(self.q0*self.q2 - self.q3*self.q1))
        y = math.atan2(2*(self.q0*self.q3+self.q1*self.q2), \
                  1 - 2*(self.q2**2 + self.q3**2))
        return r,p,y