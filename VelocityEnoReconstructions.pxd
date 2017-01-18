cimport Grid
cimport PrognosticVariables
cimport ReferenceState
cimport DiagnosticVariables
cimport ParallelMPI

cdef class VelocityEnoReconstructions:
    cdef:
        double [:] udd_x, udd_y, udd_z # undivided differences
        int enoOrder;
        
        
        
    cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa)
    cdef void computeUndividedDifferences(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cdef void EnoRecCellCenter(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables Velocities, DiagnosticVariables.DiagnosticVariables DV)
        