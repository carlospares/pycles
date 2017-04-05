#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ParallelMPI
cimport ReferenceState
cimport DiagnosticVariables
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats

from FluxDivergence cimport scalar_flux_divergence #, scalar_flux_divergence_nonconserv
from FluxDivergence cimport scalar_flux_divergence_wrapper
from Thermodynamics cimport LatentHeat

import numpy as np
cimport numpy as np

import cython

cdef extern from "scalar_advection.h":
    void compute_advective_fluxes_a(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *velocity,
                                    double *scalar, double* flux, int d, int scheme) nogil
                                    
    void compute_advective_fluxes_hiweno_nonconserv(Grid.DimStruct *dims, double *rho0, double *rho0_half,
                                        double *velocity, double *scalar, double* flux, int d, int scheme) nogil
                                        
    void compute_advective_fluxes_wrapper(Grid.DimStruct *dims, double *rho0, double *rho0_half,
                                        double *velocity_intercell, double* velocity_cc, double *scalar, 
                                        double* flux, int d, int scheme) nogil

    void compute_qt_sedimentation_s_source(Grid.DimStruct *dims, double *p0_half, double* rho0_half, double *flux,
                                    double* qt, double* qv, double* T, double* tendency, double (*lam_fp)(double),
                                    double (*L_fp)(double, double), double dx, Py_ssize_t d)nogil
                                    
cdef extern from "positivity_preservation.h":
    void positivity_preservation_xu(Grid.DimStruct *dims, double *ucc, double *vcc, double *wcc,
                                    double *scalar, double *flux, double *tendencies, double dt) nogil
                                    
    void MmP_preservation_xu(Grid.DimStruct *dims, double *ucc, double *vcc, double *wcc,
                                    double *scalar, double *flux, double *tendencies, double dt) nogil


cdef class ScalarAdvection:
    def __init__(self, namelist, LatentHeat LH,  ParallelMPI.ParallelMPI Pa):

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp

        try:
            self.order = namelist['scalar_transport']['order']
        except:
            Pa.root_print('scalar_transport order not given in namelist')
            Pa.root_print('Killing simulation now!')
            Pa.kill()
            Pa.kill()
        try:
            self.order_sedimentation = namelist['scalar_transport']['order_sedimentation']
        except:
            self.order_sedimentation = self.order
            
        try:
            self.sa_type = namelist['scalar_transport']['type']
        except:
            print 'scalar_transport: type not given in namelist, using default PyCLES'
            self.sa_type = u'original'
            
        try:
            self.positivity = namelist['scalar_transport']['positivity']
        except:
            print 'scalar_transport: positivity not given in namelist, defaulting to no positivity'
            self.positivity = False
    
        try:
            self.Mmpp = namelist['scalar_transport']['Mmpp']
        except:
            'scalar_transport: Mmpp not given in namelist, defaulting to no maximum/minimum preservation'
            self.Mmpp = False
        return

    cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims.dims,),dtype=np.double,order='c')

        #Initialize output fields
        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                NS.add_profile(PV.index_name[i] + '_flux_z',Gr,Pa)


        return
                

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV, 
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, double dt):

        cdef:
            Py_ssize_t d, i, vel_shift,scalar_shift, scalar_count = 0, flux_shift
            Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            Py_ssize_t ql_shift, qv_shift, qt_shift
            int scheme
            Py_ssize_t velcc_shift, velic_shift
            bint isNonConservative
        
        if self.sa_type == "hiweno_conserv" and self.order % 2 == 1:
            scheme = 100 + self.order
        elif self.sa_type == "hiweno_nonconserv" and self.order % 2 == 1:
            scheme = 200 + self.order
        else:
            scheme = self.order
        isNonConservative = int(self.sa_type == "hiweno_nonconserv")
        
        
        
        if self.positivity:
            for i in xrange(PV.nv): #Loop over the prognostic variables
                if PV.var_type[i] == 1: #Only compute advection if variable i is a scalar
                    scalar_shift = i * Gr.dims.npg
                    
                    for d in xrange(Gr.dims.dims):
                        flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d * Gr.dims.npg
                        vel_shift = DV.get_varshift(Gr, PV.velocity_names_directional[d] + 'cc')
                        compute_advective_fluxes_hiweno_nonconserv(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&DV.values[vel_shift],
                                                            &PV.values[scalar_shift],&self.flux[flux_shift],d,self.order)
                    
                    if self.Mmpp:
                        MmP_preservation_xu(&Gr.dims, &DV.values[DV.get_varshift(Gr, 'ucc')],
                                                &DV.values[DV.get_varshift(Gr, 'vcc')], &DV.values[DV.get_varshift(Gr, 'wcc')],
                                                &PV.values[scalar_shift], &self.flux[scalar_count * (Gr.dims.dims * Gr.dims.npg)],
                                                &PV.tendencies[scalar_shift], dt)
                    else:
                        positivity_preservation_xu(&Gr.dims, &DV.values[DV.get_varshift(Gr, 'ucc')],
                                                &DV.values[DV.get_varshift(Gr, 'vcc')], &DV.values[DV.get_varshift(Gr, 'wcc')],
                                                &PV.values[scalar_shift], &self.flux[scalar_count * (Gr.dims.dims * Gr.dims.npg)],
                                                &PV.tendencies[scalar_shift], dt)
                    scalar_count += 1
        else:
            for i in xrange(PV.nv): #Loop over the prognostic variables
                if PV.var_type[i] == 1: #Only compute advection if variable i is a scalar
                    scalar_shift = i * Gr.dims.npg
                    #No rescaling of fluxes
                    for d in xrange(Gr.dims.dims): #Loop over the cardinal direction
                        #The flux has a different shift since it is only for the scalars
                        flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d * Gr.dims.npg

                        #Make sure that we get the velocity components in the correct order
                        #Check for a scalar-specific velocity
                        sc_vel_name = PV.velocity_names_directional[d] + '_' + PV.index_name[i]
                        if sc_vel_name in DV.name_index:
                            vel_shift = DV.get_varshift(Gr, sc_vel_name)
                            if sc_vel_name == 'w_qt':
                                ql_shift = DV.get_varshift(Gr,'ql')
                                qt_shift = PV.get_varshift(Gr,'qt')
                                qv_shift =  DV.get_varshift(Gr,'qv')

                                compute_advective_fluxes_a(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&DV.values[vel_shift],
                                                       &DV.values[ql_shift],&self.flux[flux_shift],d,self.order_sedimentation)
                                scalar_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[flux_shift],
                                                   &PV.tendencies[scalar_shift],Gr.dims.dx[d],d)

                                compute_qt_sedimentation_s_source(&Gr.dims, &Rs.p0_half[0],  &Rs.rho0_half[0], &self.flux[flux_shift],
                                                                  &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[t_shift],
                                                                  &PV.tendencies[s_shift], self.Lambda_fp,self.L_fp, Gr.dims.dx[d],d)
                            else:

                                # print(sc_vel_name, ' detected as sedimentation velocity')
                                #First get the tendency associated with the sedimentation velocity
                                compute_advective_fluxes_a(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&DV.values[vel_shift],
                                                       &PV.values[scalar_shift],&self.flux[flux_shift],d,self.order_sedimentation)
                                scalar_flux_divergence(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&self.flux[flux_shift],
                                                   &PV.tendencies[scalar_shift],Gr.dims.dx[d],d)

                        # now the advective flux for all scalars
                        velcc_shift = DV.get_varshift(Gr, PV.velocity_names_directional[d] + 'cc')
                        velic_shift = PV.velocity_directions[d]*Gr.dims.npg
                        compute_advective_fluxes_wrapper(&Gr.dims,&Rs.rho0[0],&Rs.rho0_half[0],&PV.values[velic_shift], 
                                                        &DV.values[velcc_shift],&PV.values[scalar_shift],
                                                        &self.flux[flux_shift], d, scheme)
                        scalar_flux_divergence_wrapper(&Gr.dims,&Rs.alpha0[0],&Rs.alpha0_half[0],&DV.values[velcc_shift],
                                                       &self.flux[flux_shift], &PV.tendencies[scalar_shift],Gr.dims.dx[d],
                                                       d, isNonConservative)
                                                   


                    scalar_count += 1

        return

    cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t scalar_count =  0, i, d = 2, flux_shift, k
            double[:] tmp
            double [:] tmp_interp = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        for i in xrange(PV.nv):
            if PV.var_type[i] == 1:
                flux_shift = scalar_count * (Gr.dims.dims * Gr.dims.npg) + d * Gr.dims.npg
                tmp = Pa.HorizontalMean(Gr, &self.flux[flux_shift])
                for k in xrange(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
                    tmp_interp[k] = 0.5*(tmp[k-1]+tmp[k])
                NS.write_profile(PV.index_name[i] + '_flux_z', tmp_interp[Gr.dims.gw:-Gr.dims.gw], Pa)
                scalar_count += 1



        return
