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
        mult = [self.q0 * q2.q0 - np.dot(self.quaternion[1:], q2.quaternion[1:])] +\
            list(self.q0 * np.array(q2.quaternion[1:]) +\
                  q2.q0 * np.array(self.quaternion[1:]) +\
                np.cross(self.quaternion[1:], q2.quaternion[1:]))
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
    # def average(qlist):
    
# q1 = Quaternion(3, [1,-2,1], "value")
# q2 = Quaternion(2, [-1,2,3], "value")
# # print(q1.quaternion, q2.quaternion)
# # print(q1.add(q2))
# print(q1.conjugate())
# print(q1.multiply(q2).quaternion)
# print(q1.norm())
# q3 = q1.inverse()
# print(q1.multiply(q3).norm())
# print(q1.q0*q2.q0)