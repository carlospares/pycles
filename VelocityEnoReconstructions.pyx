#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
import numpy as np
cimport numpy as np
cimport ParallelMPI
from libc.math cimport fabs, fmax

import cython


cdef extern from "advection_interpolation.h":
    double interp_4_pt(double phim1, double phi, double phip1, double phip2) nogil
    double interp_6_pt(double phim2, double phim1, double phi, double phip1,
                double phip2, double phip3) nogil
    double interp_8_pt(double phim3, double phim2, double phim1, double phi, 
                       double phip1, double phip2, double phip3, double phip4) nogil
    double interp_10_pt(double phim4, double phim3, double phim2, double phim1, double phi, 
                       double phip1, double phip2, double phip3, double phip4, double phip5) nogil


@cython.boundscheck(False)
cdef inline double dot(double [:] vec1, double [:] vec2, int n) nogil:
    cdef double s = 0
    cdef int i
    for i in xrange(n):
        s = s + vec1[i]*vec2[i]
    return s

cdef class VelocityEnoReconstructions:
    def __init__(self, namelist, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        DV.add_variables('ucc','m/s','sym',Pa)
        DV.add_variables('vcc','m/s','sym',Pa)
        DV.add_variables('wcc','m/s','asym',Pa)
        
        ### cross reconstructions
        DV.add_variables('u@v','m/s','sym',Pa)
        DV.add_variables('u@w','m/s','sym',Pa)
        DV.add_variables('v@u','m/s','sym',Pa)
        DV.add_variables('v@w','m/s','sym',Pa)
        DV.add_variables('w@u','m/s','asym',Pa)
        DV.add_variables('w@v','m/s','asym',Pa)
        
        # Important! ENO rec will not work if gw < order-2
        self.enoOrder = namelist['scalar_transport']['order']
            
        # self.udd_x = np.zeros(Gr.dims.nlg[0]*self.enoOrder*Gr.dims.nl[1]*Gr.dims.nl[2], dtype=np.double, order='c')
        # self.udd_y = np.zeros(Gr.dims.nl[0]*self.enoOrder*Gr.dims.nlg[1]*Gr.dims.nl[2], dtype=np.double, order='c')
        # self.udd_z = np.zeros(Gr.dims.nl[0]*self.enoOrder*Gr.dims.nl[1]*Gr.dims.nlg[2], dtype=np.double, order='c')
        self.udd = np.zeros(self.enoOrder * Gr.dims.npg)
        
        return
        
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        
        # self.computeUndividedDifferences(Gr, PV)
        # self.EnoRecCellCenter(Gr, PV, DV)
        
        # self.computeUndividedDifferenceVdir(Gr, PV.values, PV.get_varshift(Gr, 'u'), 0)
        # self.EnoRecCellCenterVdir(Gr, DV, PV.values, PV.get_varshift(Gr, 'u'), 0, DV.get_varshift(Gr, 'ucc'), -1)
        
        # self.computeUndividedDifferenceVdir(Gr, PV.values, PV.get_varshift(Gr, 'v'), 1)
        # self.EnoRecCellCenterVdir(Gr, DV, PV.values, PV.get_varshift(Gr, 'v'), 1, DV.get_varshift(Gr, 'vcc'), -1)
        
        # self.computeUndividedDifferenceVdir(Gr, PV.values, PV.get_varshift(Gr, 'w'), 2)
        # self.EnoRecCellCenterVdir(Gr, DV, PV.values, PV.get_varshift(Gr, 'w'), 2, DV.get_varshift(Gr, 'wcc'), -1)
        
        self.CentralCellCenterVdir(Gr, DV, PV.values, PV.get_varshift(Gr, 'u'), 0, DV.get_varshift(Gr, 'ucc'), -1);
        self.CentralCellCenterVdir(Gr, DV, PV.values, PV.get_varshift(Gr, 'v'), 1, DV.get_varshift(Gr, 'vcc'), -1);
        self.CentralCellCenterVdir(Gr, DV, PV.values, PV.get_varshift(Gr, 'w'), 2, DV.get_varshift(Gr, 'wcc'), -1);
        
        DV.communicate_variable(Gr, Pa, DV.get_nv('ucc'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('vcc'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('wcc'))
        
        return
        
    @cython.boundscheck(False)           
    cdef void CentralCellCenterVdir(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                                    double [:] velocities, int vel_shift, int d, int cc_shift, int offset):
    # offset = -1 if the reconstruction for cell i goes spatially between velocities[i-1] and velocities[i]
    # offset = 0 if the reconstruction goes between velocities[i] and velocities[i+1]
        cdef:
            Py_ssize_t ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t order = self.enoOrder
            Py_ssize_t strides[3];
            Py_ssize_t i, j, k
            
        strides = [istride, jstride, 1]
        cdef Py_ssize_t stride = strides[d]
        cdef Py_ssize_t offsetstr = (1 + offset)*stride;
        
        
        with nogil:
            if self.enoOrder == 3:
                for i in range(gw, Gr.dims.nlg[0]-gw):
                    for j in range(gw, Gr.dims.nlg[1]-gw):
                        for k in range(gw, Gr.dims.nlg[2]-gw):
                            ijk = i*istride + j*jstride + k
                            DV.values[cc_shift + ijk] = interp_4_pt(velocities[ vel_shift + ijk + -2*stride + offsetstr],
                                                                    velocities[ vel_shift + ijk - stride + offsetstr],
                                                                    velocities[ vel_shift + ijk + offsetstr ],
                                                                    velocities[ vel_shift + ijk + stride + offsetstr ] )
            # 5 is at the end as default case
            
            elif self.enoOrder == 7:
                for i in range(gw, Gr.dims.nlg[0]-gw):
                    for j in range(gw, Gr.dims.nlg[1]-gw):
                        for k in range(gw, Gr.dims.nlg[2]-gw):
                            ijk = i*istride + j*jstride + k
                            DV.values[cc_shift + ijk] = interp_8_pt(velocities[ vel_shift + ijk + -4*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + -3*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + -2*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk - stride + offsetstr],
                                                                    velocities[ vel_shift + ijk + offsetstr ],
                                                                    velocities[ vel_shift + ijk + stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + 2*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + 3*stride + offsetstr ])
            elif self.enoOrder == 9:
                for i in range(gw, Gr.dims.nlg[0]-gw):
                    for j in range(gw, Gr.dims.nlg[1]-gw):
                        for k in range(gw, Gr.dims.nlg[2]-gw):
                            ijk = i*istride + j*jstride + k
                            DV.values[cc_shift + ijk] = interp_10_pt(velocities[ vel_shift + ijk + -5*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + -4*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + -3*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + -2*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk - stride + offsetstr],
                                                                    velocities[ vel_shift + ijk + offsetstr ],
                                                                    velocities[ vel_shift + ijk + stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + 2*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + 3*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + 4*stride + offsetstr ])
            else: # 5 or default
                for i in range(gw, Gr.dims.nlg[0]-gw):
                    for j in range(gw, Gr.dims.nlg[1]-gw):
                        for k in range(gw, Gr.dims.nlg[2]-gw):
                            ijk = i*istride + j*jstride + k
                            DV.values[cc_shift + ijk] = interp_6_pt(velocities[ vel_shift + ijk + -3*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + -2*stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk - stride + offsetstr],
                                                                    velocities[ vel_shift + ijk + offsetstr ],
                                                                    velocities[ vel_shift + ijk + stride + offsetstr ],
                                                                    velocities[ vel_shift + ijk + 2*stride + offsetstr ] )
        
    cpdef enoCrossReconstructions(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        ### interpolate u at v's location: u@v
        self.computeUndividedDifferenceVdir(Gr, DV.values, DV.get_varshift(Gr, 'ucc'), 1)
        self.EnoRecCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'ucc'), 1, DV.get_varshift(Gr, 'u@v'), -1)
        
        ### interpolate u at w's location: u@w
        self.computeUndividedDifferenceVdir(Gr, DV.values, DV.get_varshift(Gr, 'ucc'), 2)
        self.EnoRecCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'ucc'), 2, DV.get_varshift(Gr, 'u@w'), -1)
        
        ### interpolate v at u's location: v@u
        self.computeUndividedDifferenceVdir(Gr, DV.values, DV.get_varshift(Gr, 'vcc'), 0)
        self.EnoRecCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'vcc'), 0, DV.get_varshift(Gr, 'v@u'), -1)
        
        ### interpolate v at w's location: v@w
        self.computeUndividedDifferenceVdir(Gr, DV.values, DV.get_varshift(Gr, 'vcc'), 2)
        self.EnoRecCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'vcc'), 2, DV.get_varshift(Gr, 'v@w'), -1)
        
        ### interpolate w at u's location: w@u
        self.computeUndividedDifferenceVdir(Gr, DV.values, DV.get_varshift(Gr, 'wcc'), 0)
        self.EnoRecCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'wcc'), 0, DV.get_varshift(Gr, 'w@u'), -1)
        
        ### interpolate u at w's location: w@v
        self.computeUndividedDifferenceVdir(Gr, DV.values, DV.get_varshift(Gr, 'wcc'), 1)
        self.EnoRecCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'wcc'), 1, DV.get_varshift(Gr, 'w@v'), -1)
        
        DV.communicate_variable(Gr, Pa, DV.get_nv('u@v'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('u@w'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('v@u'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('v@w'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('w@u'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('w@v'))
        
                                    
    cpdef CentralCrossReconstructions(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        self.CentralCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'ucc'), 1, DV.get_varshift(Gr, 'u@v'), 0)
        self.CentralCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'ucc'), 2, DV.get_varshift(Gr, 'u@w'), 0)
        self.CentralCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'vcc'), 0, DV.get_varshift(Gr, 'v@u'), 0)
        self.CentralCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'vcc'), 2, DV.get_varshift(Gr, 'v@w'), 0)
        self.CentralCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'wcc'), 0, DV.get_varshift(Gr, 'w@u'), 0)
        self.CentralCellCenterVdir(Gr, DV, DV.values, DV.get_varshift(Gr,'wcc'), 1, DV.get_varshift(Gr, 'w@v'), 0)
        
        DV.communicate_variable(Gr, Pa, DV.get_nv('u@v'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('u@w'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('v@u'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('v@w'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('w@u'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('w@v'))
        
        
        
# Denote D in {x,y,z} a direction. Denote E, F the other two, following alphabetical order
# (ie: D=x => E=y, F=z; D=y => E=x, F=z; D=z => E=x, F=y).
# udd is a matrix with nEg*nFg blocks of size nDg*enoOrder        

# blocks correspond to 1D slices (of nDg elements) of the grid in the direction of D.
# they are indexed by increasing values of E then F. I.e., for the 1D slice corresponding
# to indices E = e, F = f, the index to be accessed is (f*nE + e)*blockSize

# structure of each block:
# first, a row of nDg elements, corresponding to one entire 1D slice of the grid across D (incl. ghost points)
# These are the 0th-order undivided differences.
# Then, a row of nDg elements with the first order undivided differences (last element is zero)
# followed by rows of nDg elements with the corresponding higher order undiv diffs up to same order of ENO rec
# (where the last (level) elements are zero)

# So, to access the n-th order undivided difference in the direction of D, for indices (i,j,k), one would access, e.g.
# D=x:   udd_x[ (k*ny + j)*(nxg*enoOrder) + n*nxg + (i+gw)]
# or in general:
# udd[ (i_f*nEg + i_e)*(nDg*enoOrder) + n*nDg + i_d]

    cdef void computeUndividedDifferenceVdir(self, Grid.Grid Gr, double [:] velocities, int vel_shift, int d):
        cdef:
            Py_ssize_t block_size, block_offset, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t order = self.enoOrder
            Py_ssize_t nlgd, nlge, nlgf;
            Py_ssize_t strides[3];
            Py_ssize_t i_d, i_e, i_f, n
            

        if d == 0: # (d e f) = (0 1 2)
            strides = [istride, jstride, 1]
        elif d == 1: # (d e f) = (1 0 2)
            strides = [jstride, istride, 1]
        elif d == 2: # (d e f) = (2 0 1)
            strides = [1, istride, jstride]
                                           ####   d e f
        nlgd = Gr.dims.nlg[d]           # d=0 => (0 1 2)
        nlge = Gr.dims.nlg[(d==0)]      # d=1 => (1 0 2)
        nlgf = Gr.dims.nlg[2 - (d==2)]  # d=2 => (2 0 1)
            
        block_size = order*nlgd
        for i_e in range(gw, nlge-gw):
            for i_f in range(gw, nlgf-gw):
                block_offset = (i_f*nlge + i_e)*block_size
                for i_d in range(nlgd):
                    ijk = i_d*strides[0] + i_e*strides[1] + i_f*strides[2]
                    self.udd[ block_offset + i_d ] = velocities[vel_shift + ijk]
                for n in range(1,order):
                    for i in range(nlgd - n):
                        self.udd[ block_offset + n*nlgd + i_d] = self.udd[block_offset + (n-1)*nlgd + i_d+1] - self.udd[block_offset + (n-1)*nlgd + i_d]
        
        
    #### TODO: Fix indexing!
    @cython.boundscheck(False)           
    cdef void EnoRecCellCenterVdir(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                                    double [:] velocities, int vel_shift, int d, int cc_shift, int offset):
    # offset = -1 if the reconstruction for cell i goes spatially between velocities[i-1] and velocities[i]
    # offset = 0 if the reconstruction goes between velocities[i] and velocities[i+1]
        cdef:
            Py_ssize_t block_size, block_offset, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t order = self.enoOrder
            Py_ssize_t nlgd, nlge, nlgf;
            Py_ssize_t strides[3];
            Py_ssize_t i_d, i_e, i_f, k, n
            Py_ssize_t left, right, start, direction, ijl, ijr, src, dst, level, lshift, var_shift, offsetl, offsetr
            cdef double [:] c
            
        if d == 0: # (d e f) = (0 1 2)
            strides = [istride, jstride, 1]
        elif d == 1: # (d e f) = (1 0 2)
            strides = [jstride, istride, 1]
        elif d == 2: # (d e f) = (2 0 1)
            strides = [1, istride, jstride]
                                             ###  d e f
        nlgd = Gr.dims.nlg[d]           # d=0 => (0 1 2)
        nlge = Gr.dims.nlg[(d==0)]      # d=1 => (1 0 2)
        nlgf = Gr.dims.nlg[2 - (d==2)]  # d=2 => (2 0 1)
        
        #pointwise interpolating polynomial coefficients; indexed by lshift (-1, 0, 1, ..., order-1), separated by double spaces
        if order == 3:
            c = np.array([1.875,-1.25,0.375,  0.375,0.75,-0.125,   -0.125,0.75,0.375,   0.375,-1.25,1.875 ]) #lshift = (-1, 0, 1, 2)
        elif order == 5:
            c = np.array([2.4609375,-3.28125,2.953125,-1.40625,0.2734375,   0.2734375,1.09375,-0.546875,0.21875,-0.0390625,  -0.0390625,0.46875,0.703125,-0.15625,0.0234375,  0.0234375,-0.15625,0.703125,0.46875,-0.0390625,  -0.0390625,0.21875,-0.546875,1.09375,0.2734375,  0.2734375,-1.40625,2.953125,-3.28125,2.4609375 ])
        elif order == 7:
            c = np.array([2.9326171875,-5.865234375,8.7978515625,-8.37890625,4.8876953125,-1.599609375,0.2255859375,  0.2255859375,1.353515625,-1.1279296875,0.90234375,-0.4833984375,0.150390625,-0.0205078125,  -0.0205078125,0.369140625,0.9228515625,-0.41015625,0.1845703125,-0.052734375,0.0068359375,  0.0068359375,-0.068359375,0.5126953125,0.68359375,-0.1708984375,0.041015625,-0.0048828125,  -0.0048828125,0.041015625,-0.1708984375,0.68359375,0.5126953125,-0.068359375,0.0068359375,  0.0068359375,-0.052734375,0.1845703125,-0.41015625,0.9228515625,0.369140625,-0.0205078125,  -0.0205078125,0.150390625,-0.4833984375,0.90234375,-1.1279296875,1.353515625,0.2255859375,  0.2255859375,-1.599609375,4.8876953125,-8.37890625,8.7978515625,-5.865234375,2.9326171875 ])
        elif order == 9:
            c = np.array([3.338470458984375,-8.902587890625,18.6954345703125,-26.707763671875,25.96588134765625,-16.995849609375,7.1905517578125,-1.780517578125,0.196380615234375,  0.196380615234375,1.571044921875,-1.8328857421875,2.199462890625,-1.96380615234375,1.221923828125,-0.4998779296875,0.120849609375,-0.013092041015625,  -0.013092041015625,0.314208984375,1.0997314453125,-0.733154296875,0.54986572265625,-0.314208984375,0.1221923828125,-0.028564453125,0.003021240234375,  0.003021240234375,-0.040283203125,0.4229736328125,0.845947265625,-0.35247802734375,0.169189453125,-0.0604248046875,0.013427734375,-0.001373291015625,   -0.001373291015625,0.015380859375,-0.0897216796875,0.538330078125,0.67291259765625,-0.179443359375,0.0538330078125,-0.010986328125,0.001068115234375,  0.001068115234375,-0.010986328125,0.0538330078125,-0.179443359375,0.67291259765625,0.538330078125,-0.0897216796875,0.015380859375,-0.001373291015625,  -0.001373291015625,0.013427734375,-0.0604248046875,0.169189453125,-0.35247802734375,0.845947265625,0.4229736328125,-0.040283203125,0.003021240234375,  0.003021240234375,-0.028564453125,0.1221923828125,-0.314208984375,0.54986572265625,-0.733154296875,1.0997314453125002,0.314208984375,-0.013092041015625,  -0.013092041015625,0.120849609375,-0.4998779296875,1.221923828125,-1.96380615234375,2.199462890625,-1.8328857421875,1.571044921875,0.196380615234375,  0.196380615234375,-1.780517578125,7.1905517578125,-16.995849609375,25.96588134765625,-26.707763671875,18.695434570312496,-8.902587890625,3.338470458984375])

        with nogil:
            block_size = order*nlgd
            for i_e in range(gw, nlge-gw):
                for i_f in range(gw, nlgf-gw):
                    block_offset = (i_f*nlge + i_e)*block_size
                    for i_d in range(gw, nlgd-gw):
                        ijk = i_d*strides[0] + i_e*strides[1] + i_f*strides[2]
                        left = i_d + offset
                        right = i_d+1 + offset
                        for level in range(2,order):
                            direction = fabs(self.udd[ block_offset + level*nlgd + left-1]) < fabs(self.udd[ block_offset + level*nlgd + left]) # 1 if true, 0 if false
                            left = left - direction
                            right = right + (1-direction)
                        lshift = i_d + offset - left
                        offsetl = left - i_d - offset
                        offsetr = right - i_d - offset
                        DV.values[cc_shift + ijk] = dot(c[(lshift+1)*order : (lshift+2)*order], velocities[ (vel_shift + ijk + offsetl*strides[0]) : (vel_shift + ijk + offsetr*strides[0])+1 : strides[0]], order)
