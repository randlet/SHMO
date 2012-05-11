import collections
import numpy

PopulatedLevel = collections.namedtuple("PopulatedLevel",["energy","eigen_vector","num_electrons"])

MAX_ELECTRONS_PER_LEVEL = 2
EPSILON = 1E-10 #equality test
ROUND_TO = int(-1*numpy.log10(EPSILON))

#====================================================================================
class HuckelSolver(object):
    """class for solving (Eigen values/vectors) of Simple Huckel Molecular Orbitals"""

    #----------------------------------------------------------------------
    def __init__(self, data = None, num_electrons=None):
        """SHMO system solver
        
        Keyword Arguments:
        data -- square input matrix representing bonds between atoms         
        num_electrons -- optional number of electrons for system.
        """
        
        if data is not None:
            self.set_data(data,num_electrons)
    #---------------------------------------------------------------------------
    def set_data(self,data,num_electrons=None):
        """set SHMO data and solve the system for the input data matrix and number of electrons
        
        Keyword Arguments:
        data -- square input matrix representing bonds between atoms         
        num_electrons -- optional number of electrons for system.
        """
        
        self.data = numpy.array(data,copy=True)
        
        if len(self.data.shape) != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Invalid input data. Input data must be a square matrix")
        
        
        self.num_electrons = num_electrons
        if num_electrons is None:
            self.num_electrons = self.data.shape[0]
        
        if not (0 < self.num_electrons <= self.data.shape[0]*2):
            raise ValueError("Number of electrons must be greater than zero and less than 2*number of atoms")
    
        self._solve()
        
    #---------------------------------------------------------------------------
    def _solve(self):
        """Recalculate all SHMO parameters"""
        self._solve_eigens()
        self._populate_levels()
        self._calc_bond_orders()
        self._calc_charges()
        self._calc_aa_polarizability()
        
    #---------------------------------------------------------------------------
    def _solve_eigens(self):
        """calculate eigenvalues (energies) and eigen vectors"""
        
        vals,vecs = numpy.linalg.eigh(self.data)
        
        #round so that we can test for degeneracy e.g so 0.6800000000000001 == 0.68 is considered degenerate
        self.energies = numpy.around(vals,decimals=ROUND_TO)
        self.eigen_vectors = list(vecs.T)
        self.energy_eigens = collections.OrderedDict()
        
        for e,vec in zip(self.energies,self.eigen_vectors):
            vectors = self.energy_eigens.get(e,[])
            vectors.append(vec)
            self.energy_eigens[e] = vectors
    #---------------------------------------------------------------------------
    def _populate_levels(self):
        """set electron population of each energy level"""
        
        self.populated_levels = []
        electrons_left = self.num_electrons

        for energy, eigenvecs in self.energy_eigens.items():
            
            degeneracy = len(eigenvecs)
            electrons_per_degen_level = min(MAX_ELECTRONS_PER_LEVEL, electrons_left/float(degeneracy))
            electrons_left -= electrons_per_degen_level*degeneracy
            
            for eigenvec in eigenvecs:
                populated_level = PopulatedLevel(energy=energy,eigen_vector=eigenvec,num_electrons=electrons_per_degen_level)
                self.populated_levels.append(populated_level)
    #---------------------------------------------------------------------------
    def _calc_bond_orders(self):
        """calculate pi bond orders for system"""
        size = self.data.shape[0]
        self.bond_orders = numpy.mat(numpy.zeros((size,size),numpy.float))
        for ii in range(size):
            for jj in range(ii,size):
                bond_order = sum(ne*ev[ii]*ev[jj] for (e,ev,ne) in self.populated_levels)
                self.bond_orders[ii,jj] = bond_order
                self.bond_orders[jj,ii] = bond_order
    #---------------------------------------------------------------------------
    def _calc_charges(self):
        """calculate net charge per atom"""
        size = self.data.shape[0]
        self.net_charges =  numpy.zeros(size,dtype=float)
        if self.bond_orders.any():
            self.net_charges = numpy.array([1. - self.bond_orders[ii,ii] for ii in range(size)])
        self.charge_densities = self.net_charges - 1.
    #----------------------------------------------------------------------
    def num_doubly_occupied_orbitals(self):
        return sum(1 for l in self.populated_levels if abs(2.-l.num_electrons) < EPSILON)
    #----------------------------------------------------------------------
    def _calc_aa_polarizability(self):
        """Atom-Atom polarizabilities fom Computing methods in quantum organic chemistry - Greenwood: pg 54"""
        size = self.data.shape[0]
        self.aa_polar = numpy.mat(numpy.zeros((size,size),float))
        
        n_dbl = self.num_doubly_occupied_orbitals()
        evals = self.energies
        evecs = self.eigen_vectors
        
        for rr in range(size):
            for uu in range(rr+1):                
                aap = 0.                
                for jj in range(n_dbl):
                    tmp = sum(evecs[kk][rr]*evecs[kk][uu]/(evals[jj]-evals[kk]) for kk in range(n_dbl,size))
                    aap += evecs[jj][rr]*evecs[jj][uu]*tmp
                    
                self.aa_polar[rr,uu] = aap
                self.aa_polar[uu,rr] = aap

        self.aa_polar *= 4.