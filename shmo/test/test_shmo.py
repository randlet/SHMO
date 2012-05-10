import numpy
import shmo
import unittest

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
        
    #---------------------------------------------------------------------------
    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            non_square = numpy.matrix("1 2 3; 4 5 6",dtype=numpy.float)
            solver = shmo.HuckelSolver(non_square)
    #---------------------------------------------------------------------------
    def test_too_few_electrons(self):
        
        with self.assertRaises(ValueError):
            shmo.HuckelSolver(self.input_data,num_electrons=0)
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
        
        for e, expected_e in zip(solver.energies,self.energies):
            self.assertAlmostEqual(e, expected_e,places=3)
        
        for vec, expected_vec in zip(solver.eigen_vectors, self.eigen_vectors):
            for coef, expected_coef in zip(vec,expected_vec):
                
                self.assertAlmostEqual(coef, expected_coef,places=4)
    #---------------------------------------------------------------------------
    def test_degenerate(self):
        solver = shmo.HuckelSolver(data=self.input_data,num_electrons=5)
        
        #self.assertLi

if __name__ == "__main__":
    unittest.main()            