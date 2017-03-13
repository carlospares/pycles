cimport Grid
cimport PrognosticVariables
cimport ReferenceState
cimport DiagnosticVariables
cimport ParallelMPI

cdef class VelocityEnoReconstructions:
    cdef:
        double [:] udd
        int enoOrder;
        int recType;
        
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    
    cdef void computeUndividedDifference(self, Grid.Grid Gr, double [:] velocities, int vel_shift, int d)
    cdef void enoRec(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                                    double [:] velocities, int vel_shift, int d, int cc_shift, int offset)
    cdef void centralRec(self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
                                    double [:] velocities, int vel_shift, int d, int cc_shift, int offset)
#     cpdef enoCrossReconstructions(Self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
#     cpdef centralCrossReconstructions(Self, Grid.Grid Gr, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)