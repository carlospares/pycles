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

import cython

#cdef extern from "scalar_diffusion.h":
#    void compute_diffusive_flux(Grid.DimStruct *dims, double *alpha0, double *alpha0_half, double *diffusivity,
#                                double *scalar, double *flux, double dx, size_t d, Py_ssize_t scheme, double factor)
#    void compute_qt_diffusion_s_source(Grid.DimStruct *dims, double *p0_half, double *alpha0, double *alpha0_half,
#                                       double *flux, double *qt, double *qv, double *T, double *tendency, double (*lam_fp)(double),
#                                       double (*L_fp)(double, double), double dx, Py_ssize_t d )

cdef class VelocityEnoReconstructions:
    def __init__(self, namelist, DiagnosticVariables.DiagnosticVariables DV):
        DV.add_variables('ucc','m/s','sym',Pa)
        DV.add_variables('vcc','m/s','sym',Pa)
        DV.add_variables('vcc','m/s','asym',Pa)
        
            
        self.udd_x = np.zeros(Gr.dims.ng[0]*self.enoOrder*Gr.dims.n[1]*Gr.dims.n[2], dtype=np.double, order='c')
        self.udd_y = np.zeros(Gr.dims.n[0]*self.enoOrder*Gr.dims.ng[1]*Gr.dims.n[2], dtype=np.double, order='c')
        self.udd_z = np.zeros(Gr.dims.n[0]*self.enoOrder*Gr.dims.n[1]*Gr.dims.ng[2], dtype=np.double, order='c')
        return
        
        
#    cdef void ComputeCenterReconstructions(self, Grid G, PrognosticVariables Velocities):
#        self.UndividedDifferences(G, Velocities)
#        self.EnoRecCellCenter(G, Velocities)
        
    cdef void computeUndividedDifferences(self, Grid Gr, PrognosticVariables.PrognosticVariables PV):
        cdef:            
            Py_ssize_t block_size, block_offset, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t ucc_shift = DV.get_varshift(Gr,'ucc')
            Py_ssize_t vcc_shift = DV.get_varshift(Gr,'vcc')
            Py_ssize_t wcc_shift = DV.get_varshift(Gr,'wcc')
        
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
        
        
        # the following three loops could be rewritten into one (but hurting readability)
        # udd_x
        block_size = self.enoOrder*Gr.dims.ng[0]
        for j in xrange(Gr.dims.n[1]):
            for k in xrange(Gr.dims.n[2]):
                block_offset = (k*Gr.dims.n[1] + j)*block_size
                for i in xrange(Gr.dims.ng[0]):
                    ijk = i*istride + (j+gw)*jstride + (k+gw)
                    self.udd_x[ block_offset + i ] = PV.values[u_shift + ijk]
                for n in xrange(1,self.enoOrder):
                    for i in xrange(Gr.dims.ng[0] - n):
                        self.udd_x[ block_offset + n*Gr.dims.ng[0] + i] = self.udd_x[block_offset + (n-1)*Gr.dims.ng[0] + i+1] - self.udd_x[block_offset + (n-1)*Gr.dims.ng[0] + i]
        
        # udd_y
        block_size = self.enoOrder*Gr.dims.ng[1]
        for i in xrange(Gr.dims.n[0]):
            for k in xrange(Gr.dims.n[2]):
                block_offset = (k*Gr.dims.n[0] + i)*block_size
                for j in xrange(Gr.dims.ng[1]):
                    ijk = (i+gw)*istride + j*jstride + (k+gw)
                    self.udd_y[ block_offset + j ] = PV.values[v_shift + ijk]
                for n in xrange(1,self.enoOrder):
                    for j in xrange(Gr.dims.ng[1] - n):
                        self.udd_y[ block_offset + n*Gr.dims.ng[1] + j] = self.udd_y[block_offset + (n-1)*Gr.dims.ng[1] + j+1] - self.udd_y[block_offset + (n-1)*Gr.dims.ng[1] + j]
        
        # udd_z
        block_size = self.enoOrder*Gr.dims.ng[2]
        for i in xrange(Gr.dims.n[0]):
            for j in xrange(Gr.dims.n[1]):
                block_offset = (j*Gr.dims.n[0] + i)*block_size
                for k in xrange(Gr.dims.ng[2]):
                    ijk = (i+gw)*istride + (j+gw)*jstride + k
                    self.udd_z[ block_offset + k ] = PV.values[w_shift + ijk]
                for n in xrange(1,self.enoOrder):
                    for k in xrange(Gr.dims.ng[2] - n):
                        self.udd_z[ block_offset + n*Gr.dims.ng[2] + k] = self.udd_z[block_offset + (n-1)*Gr.dims.ng[2] + k+1] - self.udd_z[block_offset + (n-1)*Gr.dims.ng[2] + k]
        return
                       
                       
                       
    @cython.boundscheck(False)           
    cdef void EnoRecCellCenter(self, Grid Gr, PrognosticVariables.PrognosticVariables Velocities, DiagnosticVariables.DiagnosticVariables DV):
        cdef:            
            Py_ssize_t block_size, block_offset, ijk
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t kstride = 1
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t w_shift = PV.get_varshift(Gr, 'w')
            Py_ssize_t ucc_shift = DV.get_varshift(Gr,'ucc')
            Py_ssize_t vcc_shift = DV.get_varshift(Gr,'vcc')
            Py_ssize_t wcc_shift = DV.get_varshift(Gr,'wcc')
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t block_size, block_offset
            
            
            int vi, i, j
            int left, right, start, direction, ijl, ijr, src, dst, level, lshift, var_shift, offsetl, offsetr
#            double a
            cdef double [:] c
        
        # pointwise interpolating polynomial coefficients; indexed by lshift (-1, 0, 1, ..., order-1), separated by double spaces
        if self.order == 3:
            c = np.array([1.875,-1.25,0.375,  0.375,0.75,-0.125,   -0.125,0.75,0.375,   0.375,-1.25,1.875 ]) #lshift = (-1, 0, 1, 2)
        elif self.order == 5:
            c = np.array([2.4609375,-3.28125,2.953125,-1.40625,0.2734375,   0.2734375,1.09375,-0.546875,0.21875,-0.0390625,  -0.0390625,0.46875,0.703125,-0.15625,0.0234375,  0.0234375,-0.15625,0.703125,0.46875,-0.0390625,  -0.0390625,0.21875,-0.546875,1.09375,0.2734375,  0.2734375,-1.40625,2.953125,-3.28125,2.4609375 ])
        elif self.order == 7:
            c = np.array([2.9326171875,-5.865234375,8.7978515625,-8.37890625,4.8876953125,-1.599609375,0.2255859375,  0.2255859375,1.353515625,-1.1279296875,0.90234375,-0.4833984375,0.150390625,-0.0205078125,  -0.0205078125,0.369140625,0.9228515625,-0.41015625,0.1845703125,-0.052734375,0.0068359375,  0.0068359375,-0.068359375,0.5126953125,0.68359375,-0.1708984375,0.041015625,-0.0048828125,  -0.0048828125,0.041015625,-0.1708984375,0.68359375,0.5126953125,-0.068359375,0.0068359375,  0.0068359375,-0.052734375,0.1845703125,-0.41015625,0.9228515625,0.369140625,-0.0205078125,  -0.0205078125,0.150390625,-0.4833984375,0.90234375,-1.1279296875,1.353515625,0.2255859375,  0.2255859375,-1.599609375,4.8876953125,-8.37890625,8.7978515625,-5.865234375,2.9326171875 ])
        elif self.order == 9:
            c = np.array([3.338470458984375,-8.902587890625,18.6954345703125,-26.707763671875,25.96588134765625,-16.995849609375,7.1905517578125,-1.780517578125,0.196380615234375,  0.196380615234375,1.571044921875,-1.8328857421875,2.199462890625,-1.96380615234375,1.221923828125,-0.4998779296875,0.120849609375,-0.013092041015625,  -0.013092041015625,0.314208984375,1.0997314453125,-0.733154296875,0.54986572265625,-0.314208984375,0.1221923828125,-0.028564453125,0.003021240234375,  0.003021240234375,-0.040283203125,0.4229736328125,0.845947265625,-0.35247802734375,0.169189453125,-0.0604248046875,0.013427734375,-0.001373291015625,   -0.001373291015625,0.015380859375,-0.0897216796875,0.538330078125,0.67291259765625,-0.179443359375,0.0538330078125,-0.010986328125,0.001068115234375,  0.001068115234375,-0.010986328125,0.0538330078125,-0.179443359375,0.67291259765625,0.538330078125,-0.0897216796875,0.015380859375,-0.001373291015625,  -0.001373291015625,0.013427734375,-0.0604248046875,0.169189453125,-0.35247802734375,0.845947265625,0.4229736328125,-0.040283203125,0.003021240234375,  0.003021240234375,-0.028564453125,0.1221923828125,-0.314208984375,0.54986572265625,-0.733154296875,1.0997314453125002,0.314208984375,-0.013092041015625,  -0.013092041015625,0.120849609375,-0.4998779296875,1.221923828125,-1.96380615234375,2.199462890625,-1.8328857421875,1.571044921875,0.196380615234375,  0.196380615234375,-1.780517578125,7.1905517578125,-16.995849609375,25.96588134765625,-26.707763671875,18.695434570312496,-8.902587890625,3.338470458984375])

        with nogil:
            
            # reconstruction of u
            block_size = self.enoOrder*Gr.dims.ng[0]
            for j in range(gw, Gr.dims.nlg[1]-gw):
                for k in range(gw, Gr.dims.nlg[2]-gw):
                    block_offset = ((k-gw)*Gr.dims.n[1] + j-gw)*block_size
                    for i in range(gw, Gr.dims.nlg[0]-gw):
                        ijk = i*istride + j*jstride + k
                        left = i
                        right = i+1
                        for level in range(2,self.enoOrder):
                            direction = fabs(self.udd_x[ block_offset + level*Gr.dims.nlg[0] + left-1]) < fabs(self.udd_x[ block_offset + level*Gr.dims.nlg[0] + left]) # 1 if true, 0 if false
                            left = left - direction
                            right = right + (1-direction)
                        lshift = i - left
                        offsetl = left - i
                        offsetr = right - i
                        DV.values[ucc_shift + ijk] = dot(c[(lshift+1)*self.order : (lshift+2)*self.order], Velocities.values[ (u_shift + ijk + offsetl*istride : (u_shift + ijk + offsetr*istride)+1 : istride], self.order)
            
            # reconstruction of v
            block_size = self.enoOrder*Gr.dims.ng[1]
            for i in range(gw, Gr.dims.nlg[0]-gw):
                for k in range(gw, Gr.dims.nlg[2]-gw):
                    block_offset = ((k-gw)*Gr.dims.n[0] + i-gw)*block_size
                    for j in range(gw, Gr.dims.nlg[1]-gw):
                        ijk = i*istride + j*jstride + k
                        left = j
                        right = j+1
                        for level in range(2,self.enoOrder):
                            direction = fabs(self.udd_y[ block_offset + level*Gr.dims.nlg[1] + left-1]) < fabs(self.udd_y[ block_offset + level*Gr.dims.nlg[1] + left]) # 1 if true, 0 if false
                            left = left - direction
                            right = right + (1-direction)
                        lshift = j - left
#                            ijl = jpts + left
#                            ijr = jpts + right
                        offsetl = left - j
                        offsetr = right - j
                        DV.values[vcc_shift + ijk] = dot(c[(lshift+1)*self.order : (lshift+2)*self.order], Velocities.values[(v_shift + ijk + offsetl*jstride) : (v_shift + ijk + offsetr*jstride)+1 : jstride], self.order)
                    
            # reconstruction of w
            block_size = self.enoOrder*Gr.dims.ng[2]
            for i in range(gw, Gr.dims.nlg[0]-gw):
                for j in range(gw, Gr.dims.nlg[1]-gw):
                    block_offset = ((j-gw)*Gr.dims.n[0] + i-gw)*block_size
                    for k in range(gw, Gr.dims.nlg[2]-gw):
                        ijk = i*istride + j*jstride + k
                        left = k
                        right = k+1
                        for level in range(2,self.enoOrder):
                            direction = fabs(self.udd_z[ block_offset + level*Gr.dims.nlg[2] + left-1]) < fabs(self.udd_z[ block_offset + level*Gr.dims.nlg[2] + left]) # 1 if true, 0 if false
                            left = left - direction
                            right = right + (1-direction)
                        lshift = k - left
                        offsetl = left - k
                        offsetr = right - k
                        DV.values[wcc_shift + ijk]] = dot(c[(lshift+1)*self.order : (lshift+2)*self.order], Velocities.values[(w_shift + ijk + offsetl*kstride) : (w_shift + ijk + offsetr*kstride)+1 : kstride], self.order)
                    
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
        
    cpdef update(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV):
        '''
        Update method for scalar diffusion class, based on a second order finite difference scheme. The method should
        only be called following a call to update method for the SGS class.
        :param Gr: Grid class
        :param RS: ReferenceState class
        :param PV: PrognosticVariables class
        :param DV: DiagnosticVariables class
        :return:
        '''

        self.computeUndividedDifferences(Gr, PV)
        
                        
        # and now that we have computed the undivided differences, we need to perform ENO
                
        
#         
#        if 'qt' in PV.name_index:
#            n_qt = PV.name_index['qt']
#            s_shift = PV.get_varshift(Gr,'s')
#            qt_shift = PV.get_varshift(Gr,'qt')
#            t_shift = DV.get_varshift(Gr,'temperature')
#            qv_shift = DV.get_varshift(Gr,'qv')
#        if 'e' in PV.name_index:
#            n_e = PV.name_index['e']
#
#
#        for i in xrange(PV.nv):
#            #Only compute fluxes for prognostic variables here
#            if PV.var_type[i] == 1:
#                scalar_shift = i * Gr.dims.npg
#                if i == n_e:
#                    diff_shift_n = DV.get_varshift(Gr,'viscosity')
#                    flux_factor = 2.0
#                else:
#                    diff_shift_n = DV.get_varshift(Gr,'viscosity')
#                    flux_factor = 1.0
#                for d in xrange(Gr.dims.dims):
#
#                    flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d* Gr.dims.npg
#
#
#                    compute_diffusive_flux(&Gr.dims,&RS.rho0[0],&RS.rho0_half[0],
#                                           &DV.values[diff_shift],&PV.values[scalar_shift],
#                                           &self.flux[flux_shift],Gr.dims.dx[d],d,2, flux_factor)
#
#                    scalar_flux_divergence(&Gr.dims,&RS.alpha0[0],&RS.alpha0_half[0],
#                                           &self.flux[flux_shift],&PV.tendencies[scalar_shift],Gr.dims.dx[d],d)
#
#                    if i == n_qt and self.qt_entropy_source:
#                        compute_qt_diffusion_s_source(&Gr.dims, &RS.p0_half[0], &RS.alpha0[0],&RS.alpha0_half[0],
#                                                      &self.flux[flux_shift],&PV.values[qt_shift], &DV.values[qv_shift],
#                                                      &DV.values[t_shift],&PV.tendencies[s_shift],self.Lambda_fp,
#                                                      self.L_fp,Gr.dims.dx[d],d)
#                scalar_count += 1

        return

