import numpy
import shmo
import unittest
TEST_PRECISION = 4
#====================================================================================
class Test(unittest.TestCase):
    #---------------------------------------------------------------------------
    def setUp(self):
        #cyclopentane
        self.input_data = numpy.matrix("""
             0 -1  0  0 -1; 
            -1  0 -1  0  0; 
             0 -1  0 -1  0; 
             0  0 -1  0 -1; 
            -1  0  0 -1  0
        """, dtype=numpy.float)
        
        self.energies = numpy.array([-2., -0.618, -0.618, 1.618, 1.618])
        self.eigen_vectors = [
            numpy.array([-0.4472, -0.4472, -0.4472, -0.4472, -0.4472 ]), #-2
            numpy.array([ 0.6325,  0.1954, -0.5117, -0.5117,  0.1954 ]), #-0.618
            numpy.array([ 0.0000,  0.6015,  0.3717, -0.3717, -0.6015 ]), #-0.618
            numpy.array([ 0.0000, -0.3717,  0.6015, -0.6015,  0.3717 ]), #1.618       
            numpy.array([-0.6325,  0.5117, -0.1954, -0.1954,  0.5117 ]), #1.618
        ]
        self.electron_count = [2,1.5,1.5,0,0]
        
        self.pi_bond_orders = numpy.matrix("""
             1.00000  0.58540 -0.08541 -0.08541  0.58540; 
             0.58540  1.00000  0.58540 -0.08541 -0.08541;
            -0.08541  0.58540  1.00000  0.58540 -0.08541;
            -0.08541 -0.08541  0.58540  1.00000  0.58540;
             0.58540 -0.08541 -0.08541  0.58540  1.00000
        """, dtype=numpy.float)
        
    #---------------------------------------------------------------------------
    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            non_square = numpy.matrix("1 2 3; 4 5 6",dtype=numpy.float)
            solver = shmo.HuckelSolver(non_square)
    #---------------------------------------------------------------------------
    def test_no_data(self):
        solver = shmo.HuckelSolver()
        self.assertEqual(0,solver.num_electrons)
    #---------------------------------------------------------------------------
    def test_set_num_e(self):
        solver = shmo.HuckelSolver(self.input_data)
        solver.set_num_electrons(4)
        self.assertEqual(solver.num_electrons,4)
        
    #---------------------------------------------------------------------------
    def test_too_few_electrons(self):
        
        with self.assertRaises(ValueError):
            shmo.HuckelSolver(self.input_data,num_electrons=-1)
    #---------------------------------------------------------------------------
    def test_too_many_electrons(self):        
        with self.assertRaises(ValueError):
            shmo.HuckelSolver(self.input_data,num_electrons=11)
            
    #---------------------------------------------------------------------------
    def test_auto_electrons(self):
        solver = shmo.HuckelSolver(self.input_data)
        self.assertEqual(solver.num_electrons,5)
                        
    #---------------------------------------------------------------------------
    def test_solver(self):
        
        solver = shmo.HuckelSolver(data=self.input_data,num_electrons=5)
        self.assertEqual(len(solver.energy_eigens.keys()),3)
        for e, expected_e in zip(solver.energies,self.energies):
            self.assertAlmostEqual(e, expected_e,places=TEST_PRECISION)
        
        for vec, expected_vec in zip(solver.eigen_vectors, self.eigen_vectors):
            for coef, expected_coef in zip(vec,expected_vec):
                
                self.assertAlmostEqual(coef, expected_coef,places=TEST_PRECISION)
    #---------------------------------------------------------------------------
    def test_population(self):
        solver = shmo.HuckelSolver(data=self.input_data,num_electrons=5)
        electron_counts = [x.num_electrons for x in solver.populated_levels]
        self.assertListEqual(self.electron_count,electron_counts)
        solver.set_data(self.input_data,num_electrons=6)
        electron_counts = [x.num_electrons for x in solver.populated_levels]
        
        self.assertListEqual([2,2,2,0,0],electron_counts)
    #---------------------------------------------------------------------------
    def test_bond_orders(self):
        solver = shmo.HuckelSolver(data=self.input_data,num_electrons=5)
        
        size = solver.data.shape[0]
        
        for ii in range(size):
            for jj in range(size):
                self.assertAlmostEqual(solver.bond_orders[ii,jj],self.pi_bond_orders[ii,jj],places=TEST_PRECISION)
    #---------------------------------------------------------------------------
    def test_net_charges(self):
        solver = shmo.HuckelSolver(data=self.input_data,num_electrons=5)
        for x in solver.net_charges:
            self.assertAlmostEqual(0,x,places=TEST_PRECISION)

        for x in solver.charge_densities:
            self.assertAlmostEqual(-1,x,places=TEST_PRECISION)
            
        solver.set_data(data=self.input_data,num_electrons=6)
        for x in solver.charge_densities:
            self.assertAlmostEqual(-1.2,x)
        for x in solver.net_charges:
            self.assertAlmostEqual(-0.2,x,places=TEST_PRECISION)
    #---------------------------------------------------------------------------
    def test_aa_polarizability(self):

        aa1 = numpy.matrix("""
            -0.32  0.00  0.16  0.16  0.00;
             0.00 -0.32  0.00  0.16  0.16;
             0.16  0.00 -0.32  0.00  0.16;
             0.16  0.16  0.00 -0.32  0.00;
             0.00  0.16  0.16  0.00 -0.32""", 
            dtype=float
        )
        
        aa2 = numpy.matrix("""
            -0.3747   0.1431   0.04422  0.04422  0.1431;
             0.1431  -0.3747   0.1431   0.04422  0.04422;
             0.04422  0.1431  -0.3747   0.1431   0.04422;
             0.04422  0.04422  0.1431  -0.3747   0.1431;
             0.1431   0.04422  0.04422  0.1431  -0.3747
        """, dtype=float)
        
        for ne,aa in [(5,aa1),(7,aa2)]:
            solver = shmo.HuckelSolver(data=self.input_data,num_electrons=ne)
            
            for ii in range(aa.shape[0]):
                for jj in range(aa.shape[0]):
                    self.assertAlmostEqual(aa[ii,jj],solver.aa_polar[ii,jj],places=4)
                                        
    #---------------------------------------------------------------------------
    def test_ab_polarizability(self):
        """"""
        ab1 = numpy.matrix("""
            -0.16  0.08  0.16 -0.16  0.08;
            -0.16 -0.16  0.08  0.08  0.16;
             0.08 -0.16 -0.16  0.16  0.08;
             0.16  0.08 -0.16  0.08 -0.16;
             0.08  0.16  0.08 -0.16 -0.16
        """, dtype=float)
        
        ab2 = numpy.matrix("""
             0.06311 -0.08522  0.04422  0.06311 -0.08522;
             0.06311  0.06311 -0.08522 -0.08522  0.04422;
            -0.08522  0.06311  0.06311  0.04422 -0.08522;
             0.04422 -0.08522  0.06311 -0.08522  0.06311;
            -0.08522  0.04422 -0.08522  0.06311  0.06311        
        """, dtype=float)
        
        solver = shmo.HuckelSolver(data=self.input_data,num_electrons=5)
        bond_pairs = solver.bond_pairs()
        self.assertListEqual(bond_pairs,[(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)])
        ab1 = [
            [-0.16, -0.16,  0.08,  0.16,  0.08],
            [-0.16,  0.08, -0.16,  0.08,  0.16],            
            [ 0.08,  0.16, -0.16, -0.16,  0.08],
            [ 0.16,  0.08,  0.08, -0.16, -0.16],
            [ 0.08, -0.16,  0.16,  0.08, -0.16],
        ]
        ab1 = map(lambda x: zip(bond_pairs,x),ab1)
        
        ab2 = [
            [ 0.06311,  0.06311, -0.08522,  0.04422, -0.08522],
            [ 0.06311, -0.08522,  0.06311, -0.08522,  0.04422],
            [-0.08522,  0.04422,  0.06311,  0.06311, -0.08522],
            [ 0.04422, -0.08522, -0.08522,  0.06311,  0.06311],
            [-0.08522,  0.06311,  0.04422, -0.08522,  0.06311]
        
        ]
        ab2 = map(lambda x: zip(bond_pairs,x),ab2)
        for ne,ab in [(5,ab1),(7,ab2)]:
            solver = shmo.HuckelSolver(data=self.input_data,num_electrons=ne)
            for atom in range(len(ab)):
                for ii,(bond_pair1, p1) in enumerate(ab[atom]):
                    bond_pair2, p2 = solver.ab_polar[atom][ii]
                    self.assertEqual(bond_pair1,bond_pair2)
                    self.assertAlmostEqual(p1,p2,places=TEST_PRECISION)
        
if __name__ == "__main__":
    unittest.main()            