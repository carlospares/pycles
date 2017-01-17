from Grid cimport Grid
from PrognosticVariables cimport PrognosticVariables
from ReferenceState cimport ReferenceState
        
cdef double fluxlim(double r) nogil

cdef class VelocityEnoReconstruction:
    cdef:
        double [:] udd_x, udd_y, udd_z # undivided differences
        int enoOrder = 3;
        
        
        
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,PrognosticVariables.PrognosticVariables PV,
                     DiagnosticVariables.DiagnosticVariables DV)
        