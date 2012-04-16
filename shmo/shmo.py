import numpy

#====================================================================================
class HuckelSolver(object):
    """class for solving (Eigen values/vectors) of Simple Huckel Molecular Orbitals"""

    #----------------------------------------------------------------------
    def __init__(self, data,num_electrons=None):

        self.data = numpy.array(data,copy=True)
        
        if len(self.data.shape) != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Invalid input data. Input data must be a square matrix")
        
        
        self.num_electrons = num_electrons
        if num_electrons is None:
            self.num_electrons = self.data.shape[0]
        
        if not (0 < self.num_electrons <= self.data.shape[0]*2):
            raise ValueError("Number of electrons must be greater than zero and less than 2*number of atoms")
    
        self.solve()
        
    #---------------------------------------------------------------------------
    def solve(self):
        """Recalculated all SHMO parameters"""
        
        vals,vecs = numpy.linalg.eigh(self.data,'L')
        
        self.energies = vals
        self.eigen_vectors = list(vecs.T)
        
        
        