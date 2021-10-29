# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:18:00 2017

@author: daniele
"""

import numpy as np
import sys

"""    
GEO2ECEF function computes ECEF coordinates starting by geodetic
    input:
        geodetic coordinates lat,long,h (rad,rad,m)
    
    output:
        ECEF coordinates Xo,Yo,Zo (m,m,m)

    called functions: calcN
"""
    
def geo2ecef( lat, long, hh):
        # WGS84 parameters
        # a = 6378137                 # semi-major axis (m)
        f = 1/298.257222101         # flatness
        ecc = np.sqrt(2*f - f**2)   # eccentricity

        # prime vertical radius computation
        N = calcN(lat)

        Xo = (N+hh) * np.cos(long) * np.cos(lat)
        Yo = (N+hh) * np.sin(long) * np.cos(lat)
        Zo = (N * (1 - ecc**2) + hh) * np.sin(lat)
        
        return Xo, Yo, Zo
    
"""    
CALCN calculates the prime vertical radius of curvature at a given
latitude for a given ellipsoid.

The WGS84 ellipsoid parameters are the default values
The latitude parameter should be given in radians.
"""    
        
def calcN( lat, a = 6378137.0, b = 6356752.31425 ):

        e2 = (a**2 - b**2)/(a**2)
        W  = np.sqrt(1.0 - e2 * ((np.sin(lat))**2))
        N  = a/W

        return N
"""
Calculates the ellipsoidal suface distance given a latitude from the equator
plane for a given ellipsoid.
    
The WGS84 ellipsoid parameters are the default values
The latitude parameter should be given in radians.

The latitude parameter should be given in radians.
"""
def calcM( lat, a = 6378137.0, b = 6356752.31425 ):

        e2 = (a**2 - b**2) / (a**2)

        sM = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*lat 
            -(3*e2/8 +3*e2**2/32 +45*e2**3/1024)*np.sin(2*lat)
            +(15*e2**2/256 + 45*e2**3/1024)*np.sin(4*lat)
            -(35*e2**3/3072)*np.sin(6*lat))

        return sM

"""
ecef2enu transform ECEF coordinates in ENU, centered in an input origin
input:
    - ECEF coordinates Xecef,Yecef,Zecef (meter)
    - origin geodetic coordinates lat, lon, h (rad,rad,meter)
    - kind of use MODE: 'pos' for position vector transformation (rotation &
           translation), 'vel' for velocity or acceleration vector
           transformation (only rotation)
           
output:
    - ENU coordinates (meter)

    note: Xecef,Yecef,Zecef can be array with same length
          lat, long, h are scalar

    called functions: matrix_rot 
                      geo2ecef 
                      calcN
"""
    
def ecef2enu( Xecef, Yecef, Zecef, lat, lon, height, mode):
    
        # # Xecef,Yecef,Zecef must be array
        # Xecef = Xecef.flatten()
        # Yecef = Yecef.flatten()
        # Zecef = Zecef.flatten()

        # first rotation matrix
        R1 = matrix_rot( np.pi/2 + lon, 'A', 'z')

        # second rotazion matrix
        R2 = matrix_rot( np.pi/2 - lat, 'A', 'x')

        R = R2.dot( R1 )

        # Geodetic coordinates conversion
        Xo, Yo, Zo = geo2ecef( lat, lon, height)

        if mode == 'pos':
            XYZecef = np.vstack((Xecef-Xo, Yecef-Yo, Zecef-Zo))

        elif mode == 'vel':
            XYZecef = np.vstack((Xecef, Yecef, Zecef))
   
        XYZenu = np.dot(R, XYZecef )
        Xenu = XYZenu[0,:].flatten()
        Yenu = XYZenu[1,:].flatten()
        Zenu = XYZenu[2,:].flatten()

        return Xenu, Yenu, Zenu
        
"""
    matrix_rot builds rotation matrix with respect to a specific
    rotation axis

    theta in radians    
    
    direction = 'O' clock-wise rotations
    directuion = 'A' anticlock wise rotations

    axis = 'x' rotation around the x axis
    axis = 'y' rotation around the y axis
    axis = 'z' rotation around the z axis
"""
    
def matrix_rot( theta, direction, axis ):

        if direction == 'A':
            theta = -theta
    
        else: 
            if direction != 'O':
                sys.stderr.write("Unsupported rotation direction")
                sys.exit(1)


        if axis == 'x':
            R = np.array([[1, 0, 0], 
                          [0, np.cos(theta), -np.sin(theta)], 
                          [0, np.sin(theta), np.cos(theta)]])
        elif axis == 'y':
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
        elif axis == 'z':
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
        else:
            sys.stderr.write("Unsupported rotation axis")
            sys.exit(1) 
        
        return R
        
"""
    ecef2geo converts ECEF coordinates into geographic coordinates
    (lat, lon, hight) referred to the WGS84 ellissoid
    B.R.Bowring formulas

    input in meters
    
    output in rad, rad and meters
"""
    
def ecef2geo(  Xecef, Yecef, Zecef):
        
        # elements of tf the WGS84 ellissoid
        a = 6378137 # in m
        f = 1/298.257222101
        e = np.sqrt(2*f - f**2)
        b = a*(1-f)
        e_sec = np.sqrt((a**2 - b**2)/(b**2))

        # formule di Bowring
        lon = np.arctan2(Yecef, Xecef)
        r = np.sqrt(Xecef**2 + Yecef**2)
        rlat = np.arctan2(Zecef, (1 - f) * r)

        lat = np.arctan((Zecef + e_sec**2 * b * (np.sin(rlat)**3)) / (r - e**2 * a * (np.cos(rlat)**3)))
        h = r * np.cos(lat) + Zecef*np.sin(lat) - a * np.sqrt(1-e**2*(np.sin(lat))**2)
        
        return lat, lon, h
        
"""    
ecef2enu transform ECEF coordinates in ENU, centered in an input origin
    
    input:
         - ECEF coordinates Xecef,Yecef,Zecef (meter)
         - origin geodetic coordinates lat, lon, h (rad,rad,meter)
         - mode: 'pos' for position vector transformation (rotation &
           translation), 'vel' for velocity or acceleration vector
           transformation (only rotation)

    output:
         - ENU coordinates (meter)

"""

def enu2ecef( Xenu, Yenu, Zenu, lat, lon, hh, mode):

        # first rotation matrix
        R1 = matrix_rot( np.pi/2 + lon,'A','z')

        # second rotation matrix
        R2 = matrix_rot( np.pi/2 - lat,'A','x');

        R = np.dot(R1.T, R2.T)

        #  WGS84 parameters
        a = 6378137.0   # major semi-axis (WGS84) in meters
        f = 1/298.257223563  # flattening
        e = np.sqrt(2*f-f**2) # eccentricity

        # radius at the obsever latitude
        Rv = a * np.sqrt(1 - 2*e**2 * np.sin(lat)**2) / np.sqrt(1 - e**2 * np.sin(lat)**2)

        # geocentric latitude
        psi = np.arctan((1-e**2)*np.tan(lat))

        if mode == 'pos':
            XYZenu = np.vstack(( Xenu,           
                                 Yenu - Rv * np.sin(lat - psi),
                                 Zenu + Rv * np.cos(lat-psi) + hh ) )
        elif mode == 'vel':
            XYZenu = np.vstack(( Xenu, Yenu, Zenu ) )
        else:
            sys.stderr.write("Unsupported rotation axis")
            sys.exit(1)     
   
        XYZecef = R.dot( XYZenu )    # apply coordinate rotation
        
        return XYZecef[0,:].flatten(), XYZecef[1,:].flatten(), XYZecef[2,:].flatten()

"""
enu2geo
"""
    
def enu2geo( Xenu, Yenu, Zenu, lat, lon, hh, mode):

        Xecef, Yecef, Zecef = enu2ecef( Xenu, Yenu, Zenu,lat, lon, hh, mode)
        return ecef2geo(Xecef, Yecef, Zecef)
        
"""
geo2enu
"""
def geo2enu( lat, long, hh, lat0, lon0, h0 ):

        Xo, Yo, Zo = geo2ecef( lat, long, hh )
        a = 1
        return ecef2enu(Xo, Yo, Zo, lat0, lon0, h0, 'pos')    