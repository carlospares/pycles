cimport Grid
cimport PrognosticVariables
cimport ReferenceState
cimport DiagnosticVariables
cimport ParallelMPI

cdef class VelocityEnoReconstructions:
    cdef:
        # double [:] udd_x, udd_y, udd_z # undivided differences
        double [:] udd
        int enoOrder;
        
        
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    
    # cdef void computeUndividedDifferences(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    # cdef void EnoRecCellCenter(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables Velocities, DiagnosticVariables.DiagnosticVariables DV)
    cdef void computeUndividedDifferenceVdir(self, Grid.Grid Gr, double [:] velocities, int vel_shift, int d)
    cdef void EnoRecCellCenterVdir(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                                    double [:] velocities, int vel_shift, int d, int cc_shift, int offset)
    cdef void CentralCellCenterVdir(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                                    double [:] velocities, int vel_shift, int d, int cc_shift, int offset)
    cpdef enoCrossReconstructions(Self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cpdef CentralCrossReconstructions(Self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)