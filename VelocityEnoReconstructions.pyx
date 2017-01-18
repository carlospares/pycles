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

#cdef extern from "scalar_diffusion.h":
#    void compute_diffusive_flux(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *diffusivity,
#                                double *scalar, double *flux, double dx, size_t d, Py_ssize_t scheme, double factor)
#    void compute_qt_diffusion_s_source(Grid.DimStruct *dims, double *p0_half, double *alpha0, double *alpha0_half,
#                                       double *flux, double *qt, double *qv, double *T, double *tendency, double (*lam_fp)(double),
#                                       double (*L_fp)(double, double), double dx, Py_ssize_t d )

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
        
        self.enoOrder = 3;
            
        self.udd_x = np.zeros(Gr.dims.nlg[0]*self.enoOrder*Gr.dims.nl[1]*Gr.dims.nl[2], dtype=np.double, order='c')
        self.udd_y = np.zeros(Gr.dims.nl[0]*self.enoOrder*Gr.dims.nlg[1]*Gr.dims.nl[2], dtype=np.double, order='c')
        self.udd_z = np.zeros(Gr.dims.nl[0]*self.enoOrder*Gr.dims.nl[1]*Gr.dims.nlg[2], dtype=np.double, order='c')
        
        return
        
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
        self.computeUndividedDifferences(Gr, PV)
        self.EnoRecCellCenter(Gr, PV, DV)
        
             
        DV.communicate_variable(Gr, Pa, DV.get_nv('ucc'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('vcc'))
        DV.communicate_variable(Gr, Pa, DV.get_nv('wcc'))
        
        return
        
    cdef void computeUndividedDifferences(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
        cdef:            
            Py_ssize_t block_size, block_offset, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t gw = Gr.dims.gw
            int order = self.enoOrder
            Py_ssize_t nlgx = Gr.dims.nlg[0]
            Py_ssize_t nlgy = Gr.dims.nlg[1]
            Py_ssize_t nlgz = Gr.dims.nlg[2]
            Py_ssize_t nlx = Gr.dims.nl[0]
            Py_ssize_t nly = Gr.dims.nl[1]
            Py_ssize_t nlz = Gr.dims.nl[2]
            int i,j,k,n
        
        # Denote D in {x,y,z} a direction. Denote E, F the other two, following alphabetical order
        # (ie: D=x => E=y, F=z; D=y => E=x, F=z; D=z => E=x, F=y).
        # udd_D is a matrix with nE*nF blocks of size nDg*enoOrder        
        
        # blocks correspond to 1D slices (of nDg elements) of the grid in the direction of D.
        # they are indexed by increasing values of E then F. I.e., for the 1D slice corresponding
        # to E = e, F = f, the index to be accessed is (f*nE + e)*blockSize
        
        # structure of each block:
        # first, a row of nDg elements, corresponding to one entire 1D slice of the grid across D (incl. ghost points)
        # then, a row of nDg elements with the first order undivided differences (last element is zero)
        # followed by rows of nDg elements with the corresponding higher order undiv diffs up to same order of ENO rec
        
        # So, to access the n-th order undivided difference in the direction of D, for indices (i,j,k), one would access
        # D=x:   udd_x[ (k*ny + j)*(nxg*enoOrder) + (n-1)*nxg + (i+gw)]
        # D=y:   udd_y[ (k*nx + i)*(nyg*enoOrder) + (n-1)*nyg + (j+gw)]
        # D=z:   udd_z[ (j*nx + i)*(nzg*enoOrder) + (n-1)*nzg + (k+gw)]
        
        
        # udd_x
        block_size = order*nlgx
        for j in range(nly):
            for k in range(nlz):
                block_offset = (k*nly + j)*block_size
                for i in range(nlgx):
                    ijk = i*istride + (j+gw)*jstride + (k+gw)
                    self.udd_x[ block_offset + i ] = PV.values[u_shift + ijk]
                for n in range(1,order):
                    for i in range(nlgx - n):
                        self.udd_x[ block_offset + n*nlgx + i] = self.udd_x[block_offset + (n-1)*nlgx + i+1] - self.udd_x[block_offset + (n-1)*nlgx + i]
        
        # udd_y
        block_size = order*nlgy
        for i in range(nlx):
            for k in range(nlz):
                block_offset = (k*nlx + i)*block_size
                for j in range(nlgy):
                    ijk = (i+gw)*istride + j*jstride + (k+gw)
                    self.udd_y[ block_offset + j ] = PV.values[v_shift + ijk]
                for n in range(1,order):
                    for j in range(nlgy - n):
                        self.udd_y[ block_offset + n*nlgy + j] = self.udd_y[block_offset + (n-1)*nlgy + j+1] - self.udd_y[block_offset + (n-1)*nlgy + j]
        
        # udd_z
        block_size = order*nlgz
        for i in range(nlx):
            for j in range(nly):
                block_offset = (j*nlx + i)*block_size
                for k in range(nlgz):
                    ijk = (i+gw)*istride + (j+gw)*jstride + k
                    self.udd_z[ block_offset + k ] = PV.values[w_shift + ijk]
                for n in range(1,order):
                    for k in range(nlgz - n):
                        self.udd_z[ block_offset + n*nlgz + k] = self.udd_z[block_offset + (n-1)*nlgz + k+1] - self.udd_z[block_offset + (n-1)*nlgz + k]
        return
                       
                       
                       
    @cython.boundscheck(False)           
    cdef void EnoRecCellCenter(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables Velocities, DiagnosticVariables.DiagnosticVariables DV):
        cdef:            
            Py_ssize_t block_size, block_offset, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t kstride = 1
            Py_ssize_t u_shift = Velocities.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = Velocities.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = Velocities.get_varshift(Gr, 'w')
            Py_ssize_t ucc_shift = DV.get_varshift(Gr,'ucc')
            Py_ssize_t vcc_shift = DV.get_varshift(Gr,'vcc')
            Py_ssize_t wcc_shift = DV.get_varshift(Gr,'wcc')
            Py_ssize_t gw = Gr.dims.gw
            int order = self.enoOrder
            Py_ssize_t nlgx = Gr.dims.nlg[0]
            Py_ssize_t nlgy = Gr.dims.nlg[1]
            Py_ssize_t nlgz = Gr.dims.nlg[2]
            Py_ssize_t nlx = Gr.dims.nl[0]
            Py_ssize_t nly = Gr.dims.nl[1]
            Py_ssize_t nlz = Gr.dims.nl[2]
            
            int vi, i, j, k
            int left, right, start, direction, ijl, ijr, src, dst, level, lshift, var_shift, offsetl, offsetr
#            double a
            cdef double [:] c
        
        # pointwise interpolating polynomial coefficients; indexed by lshift (-1, 0, 1, ..., order-1), separated by double spaces
        if order == 3:
            c = np.array([1.875,-1.25,0.375,  0.375,0.75,-0.125,   -0.125,0.75,0.375,   0.375,-1.25,1.875 ]) #lshift = (-1, 0, 1, 2)
        elif order == 5:
            c = np.array([2.4609375,-3.28125,2.953125,-1.40625,0.2734375,   0.2734375,1.09375,-0.546875,0.21875,-0.0390625,  -0.0390625,0.46875,0.703125,-0.15625,0.0234375,  0.0234375,-0.15625,0.703125,0.46875,-0.0390625,  -0.0390625,0.21875,-0.546875,1.09375,0.2734375,  0.2734375,-1.40625,2.953125,-3.28125,2.4609375 ])
        elif order == 7:
            c = np.array([2.9326171875,-5.865234375,8.7978515625,-8.37890625,4.8876953125,-1.599609375,0.2255859375,  0.2255859375,1.353515625,-1.1279296875,0.90234375,-0.4833984375,0.150390625,-0.0205078125,  -0.0205078125,0.369140625,0.9228515625,-0.41015625,0.1845703125,-0.052734375,0.0068359375,  0.0068359375,-0.068359375,0.5126953125,0.68359375,-0.1708984375,0.041015625,-0.0048828125,  -0.0048828125,0.041015625,-0.1708984375,0.68359375,0.5126953125,-0.068359375,0.0068359375,  0.0068359375,-0.052734375,0.1845703125,-0.41015625,0.9228515625,0.369140625,-0.0205078125,  -0.0205078125,0.150390625,-0.4833984375,0.90234375,-1.1279296875,1.353515625,0.2255859375,  0.2255859375,-1.599609375,4.8876953125,-8.37890625,8.7978515625,-5.865234375,2.9326171875 ])
        elif order == 9:
            c = np.array([3.338470458984375,-8.902587890625,18.6954345703125,-26.707763671875,25.96588134765625,-16.995849609375,7.1905517578125,-1.780517578125,0.196380615234375,  0.196380615234375,1.571044921875,-1.8328857421875,2.199462890625,-1.96380615234375,1.221923828125,-0.4998779296875,0.120849609375,-0.013092041015625,  -0.013092041015625,0.314208984375,1.0997314453125,-0.733154296875,0.54986572265625,-0.314208984375,0.1221923828125,-0.028564453125,0.003021240234375,  0.003021240234375,-0.040283203125,0.4229736328125,0.845947265625,-0.35247802734375,0.169189453125,-0.0604248046875,0.013427734375,-0.001373291015625,   -0.001373291015625,0.015380859375,-0.0897216796875,0.538330078125,0.67291259765625,-0.179443359375,0.0538330078125,-0.010986328125,0.001068115234375,  0.001068115234375,-0.010986328125,0.0538330078125,-0.179443359375,0.67291259765625,0.538330078125,-0.0897216796875,0.015380859375,-0.001373291015625,  -0.001373291015625,0.013427734375,-0.0604248046875,0.169189453125,-0.35247802734375,0.845947265625,0.4229736328125,-0.040283203125,0.003021240234375,  0.003021240234375,-0.028564453125,0.1221923828125,-0.314208984375,0.54986572265625,-0.733154296875,1.0997314453125002,0.314208984375,-0.013092041015625,  -0.013092041015625,0.120849609375,-0.4998779296875,1.221923828125,-1.96380615234375,2.199462890625,-1.8328857421875,1.571044921875,0.196380615234375,  0.196380615234375,-1.780517578125,7.1905517578125,-16.995849609375,25.96588134765625,-26.707763671875,18.695434570312496,-8.902587890625,3.338470458984375])

        with nogil:
            
            # reconstruction of u
            block_size = order*nlgx
            for j in range(gw, nlgy-gw):
                for k in range(gw, nlgz-gw):
                    block_offset = ((k-gw)*nly + j-gw)*block_size
                    for i in range(gw, nlgx-gw):
                        ijk = i*istride + j*jstride + k
                        left = i
                        right = i+1
                        for level in range(2,order):
                            direction = fabs(self.udd_x[ block_offset + level*nlgx + left-1]) < fabs(self.udd_x[ block_offset + level*nlgx + left]) # 1 if true, 0 if false
                            left = left - direction
                            right = right + (1-direction)
                        lshift = i - left
                        offsetl = left - i
                        offsetr = right - i
                        DV.values[ucc_shift + ijk] = dot(c[(lshift+1)*order : (lshift+2)*order], Velocities.values[ (u_shift + ijk + offsetl*istride) : (u_shift + ijk + offsetr*istride)+1 : istride], order)
            
            # reconstruction of v
            block_size = order*nlgy
            for i in range(gw, nlgx-gw):
                for k in range(gw, nlgz-gw):
                    block_offset = ((k-gw)*nlx + i-gw)*block_size
                    for j in range(gw, nlgy-gw):
                        ijk = i*istride + j*jstride + k
                        left = j
                        right = j+1
                        for level in range(2,order):
                            direction = fabs(self.udd_y[ block_offset + level*nlgy + left-1]) < fabs(self.udd_y[ block_offset + level*nlgy + left]) # 1 if true, 0 if false
                            left = left - direction
                            right = right + (1-direction)
                        lshift = j - left
#                            ijl = jpts + left
#                            ijr = jpts + right
                        offsetl = left - j
                        offsetr = right - j
                        DV.values[vcc_shift + ijk] = dot(c[(lshift+1)*order : (lshift+2)*order], Velocities.values[(v_shift + ijk + offsetl*jstride) : (v_shift + ijk + offsetr*jstride)+1 : jstride], order)
                    
            # reconstruction of w
            block_size = order*nlgz
            for i in range(gw, nlgx-gw):
                for j in range(gw, nlgy-gw):
                    block_offset = ((j-gw)*nlx + i-gw)*block_size
                    for k in range(gw, nlgz-gw):
                        ijk = i*istride + j*jstride + k
                        left = k
                        right = k+1
                        for level in range(2,order):
                            direction = fabs(self.udd_z[ block_offset + level*nlgz + left-1]) < fabs(self.udd_z[ block_offset + level*nlgz + left]) # 1 if true, 0 if false
                            left = left - direction
                            right = right + (1-direction)
                        lshift = k - left
                        offsetl = left - k
                        offsetr = right - k
                        DV.values[wcc_shift + ijk] = dot(c[(lshift+1)*order : (lshift+2)*order], Velocities.values[(w_shift + ijk + offsetl*kstride) : (w_shift + ijk + offsetr*kstride)+1 : kstride], order)
             
             
#            # Boundary conditions
#            for vi in range(Velocities.nvars):
#                var_shift = vi * ng
##                for i in range(gw, nx + gw):
##                    jpts = i * nyg
##                    for j in range(gw):
##                        src = ny + j
##                        dst = j
##                        self.rec_ctr[var_shift + jpts + dst ] = self.rec_ctr[var_shift + jpts + src] #copy right into left
##                        
##                        src = gw + j
##                        dst = gw + ny + j
##                        self.rec_ctr[var_shift + jpts + dst ] = self.rec_ctr[var_shift + jpts + src] #copy left into right
##                ######################################
##                # reflecting bcs in vertical domain
##                #####################################
#                if vi == uindex: # u: just reflect on the boundary
#                    for i in range(gw, nx+gw):
#                        jpts = i * nyg
#                        for j in range(gw):
#                            src = 2*gw - 1 - j
#                            dst = j
#                            self.rec_ctr[var_shift + jpts + dst] = self.rec_ctr[var_shift + jpts + src] #copy right into left (in the sense of the array, not physical)
#                            
#                            src = gw + ny - 1 - j
#                            dst = gw + ny + j
#                            self.rec_ctr[var_shift + jpts + dst] = self.rec_ctr[var_shift + jpts + src] #copy left into right (")
#                else: # w: reflect with change of sign
#                    for i in range(0, nxg):
#                        jpts = i * nyg
#                        for j in range(gw):
#                            src = 2*gw - 1 - j
#                            dst = j
#                            self.rec_ctr[var_shift + jpts + dst ] = -self.rec_ctr[var_shift + jpts + src] #copy right into left (in the sense of the array, not physical)
#                        for j in range(gw-1):
#                            src = gw + ny - 1 - j
#                            dst = gw + ny + j
#                            self.rec_ctr[var_shift + jpts + dst ] = -self.rec_ctr[var_shift + jpts + src] #copy left into right (")
#    
#                for i in range(gw):
#                    src = nx + i
#                    dst = i
#                    for j in range(nyg):
#                        self.rec_ctr[var_shift + dst*nyg + j ] = self.rec_ctr[var_shift + src*nyg + j]
#                    
#                    src = gw + i
#                    dst = gw + nx + i
#                    for j in range(nyg):
#                        self.rec_ctr[var_shift + dst*nyg + j ] = self.rec_ctr[var_shift + src*nyg + j]
                        
                        
#        
        

