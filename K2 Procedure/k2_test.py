import unittest
import pandas as pd
import numpy as np
from k2_core import *
import random

class TestUniqueInstantiations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global data, r
        data = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5"])
        data["x1"] = [5, 2, 1, 3] 
        data["x2"] = [1, 1, 1, 0]
        data["x3"] = [2, 0, 1, 2]
        data["x4"] = [3, 1, 2, 1]
        data["x5"] = [1, 1, 1, 1]

    def test_no_parents(self):
        # In this case i = 1, so the function should return an empty array
        self.assertTrue((unique_instantiations(1, [], data) == np.array([[]])).all())

    def test_single_parent(self):
        # x3 has a single parent x1, so the function should return the unique values of x1
        expected = np.array([[5], [2], [1], [3]])
        result = unique_instantiations(2, [0], data)

        temp_res = []

        for val in expected:
            if val not in result:
                temp_res.append(val)
        
        for val in result:
            if val not in expected:
                temp_res.append(val)
            
        np.testing.assert_array_equal(temp_res, [])

    def test_multiple_parents(self):
        # x4 has two parents x2 and x3, so the function should return the unique combinations of x2 and x3
        expected = np.array([[1, 2], [1, 0], [1, 1], [0, 2]])
        result = unique_instantiations(3, [1, 2], data)

        temp_res = []

        for val in expected:
            if val not in result:
                temp_res.append(val)
                
        for val in result:
            if val not in expected:
                temp_res.append(val)

        np.testing.assert_array_equal(temp_res, [])

    def test_single_parent_same_value(self):
        # x4 has a single parent x5, but x5 has the same value for all instances
        expected = np.array([[1]])
        result = unique_instantiations(3, [4], data)

        temp_res = []

        for val in expected:
            if val not in result:
                temp_res.append(val)
        
        for val in result:
            if val not in expected:
                temp_res.append(val)

        np.testing.assert_array_equal(temp_res, [])
        
    def test_two_parents_no_duplicates(self):
        # x2 has two parents x1 and x4, and there are no duplicate combinations
        # ? Technically, in a correct execution, the function should be called in the topological order! 
        # ? So, x4 cannot be called before x2,  it's a debug test to validate the function behavior
        expected = np.array([[5, 3], [2, 1], [1, 2], [3, 1]])
        result = unique_instantiations(1, [0, 3], data)

        temp_res = []

        for val in expected:
            if val not in result:
                temp_res.append(val)

        for val in result:
            if val not in expected:
                temp_res.append(val)

        np.testing.assert_array_equal(temp_res, [])

    def test_two_parents_with_duplicates(self):
         # x3 has two parents x2 and x5, and there are duplicate combinations
        # ? Technically, in a correct execution, the function should be called in the topological order! 
        # ? So, x5 cannot be called before x3, it's a debug test to validate the function behavior
        expected = np.array([[1, 1], [0, 1]])
        result = unique_instantiations(2, [1, 4], data)

        temp_res = []

        for val in expected:
            if val not in result:
                temp_res.append(val)

        for val in result:
            if val not in expected:
                temp_res.append(val)

        np.testing.assert_array_equal(temp_res, [])

    def test_three_parents(self):
        # x4 has three parents x1, x2 and x3
        expected = np.array([[5, 1, 2], [2, 1, 0], [1, 1, 1], [3, 0, 2]])

        result = unique_instantiations(3, [0, 1, 2], data)

        temp_res = []

        for val in expected:
            if val not in result:
                temp_res.append(val)

        for val in result:
            if val not in expected:
                temp_res.append(val)

        np.testing.assert_array_equal(temp_res, [])

    def test_all_parents(self):
        # Tutti i nodi come genitori di x4
        expected = np.array([[5, 1, 2, 3], [2, 1, 0, 1], [1, 1, 1, 2], [3, 0, 2, 1]])
        result = unique_instantiations(4, [0, 1, 2, 3], data)

        temp_res = []

        for val in expected:
            if val not in result:
                temp_res.append(val)

        for val in result:
            if val not in expected:
                temp_res.append(val)

        np.testing.assert_array_equal(temp_res, [])

class TestNijkFunction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        global data, r
        data = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5"])
        data["x1"] = [5, 2, 1, 3, 2, 1, 1] 
        data["x2"] = [1, 1, 1, 0, 1, 1, 1]
        data["x3"] = [2, 0, 1, 2, 1, 1, 1]
        data["x4"] = [3, 1, 2, 1, 1, 2, 1]
        data["x5"] = [1, 1, 1, 1, 1, 1, 1]

    def test_no_parents(self):
        # Test when there are no parents
        xi = 1 # x2
        j = 0 # First unique instantiation
        k = 1 # Value to count
        parents_xi = [] 
        result = Nijk(xi, j, k, parents_xi, data) 

        # As there are no parents, the count should be the same as the total count of x2 == 1
        expected = 6  # All values of x2 are 1

        self.assertEqual(result, expected)

    def test_single_parent(self):
        # Test with a single parent
        xi = 2 # x3
        j = 0
        k = 2 # Value to count
        parents_xi = [0]

        # Considering unique instantiations of x1, should be [5], [2], [1], [3]
        # The first instantiation is [5] (1 instance)
        # The count of x3 == 2 with x1 == 5 should be 1 istance
        result = Nijk(xi, j, k, parents_xi, data)
        expected = 1  
        self.assertEqual(result, expected)

    def test_multiple_parents(self):
        # Test with multiple parents
        xi = 3
        j = 1
        k = 1
        parents_xi = [0, 1]

        # Considering unique instantiations of x1 and x2, should be [5, 1], [2, 1], [1, 1], [3, 0]
        # The second instantiation is [2, 1] (2 instances)
        # The count of x4 == 1 with x1 == 2 and x2 == 1 should be 2 (still 2 instances)
        result = Nijk(xi, j, k, parents_xi, data)
        expected = 2  
        self.assertEqual(result, expected)

    def test_no_matching_instantiation(self):
        # Test when there is no matching instantiation
        xi = 3
        j = 2
        k = 3
        parents_xi = [0, 1, 2]
        # Considering unique instantiations of x1, x2 and x3, should be [5, 1, 2], [2, 1, 0], [1, 1, 1], [3, 0, 2]
        # The first instantiation is [1, 1, 1] (3 istances)
        # There are no instances where x4 == 3 with x1 == 1, x2 == 1 and x3 == 1

        result = Nijk(xi, j, k, parents_xi, data)
        expected = 0  
        self.assertEqual(result, expected)

    def test_all_parents(self):
        xi = 4
        j = 2
        k = 1
        parents_xi = [0, 1, 2, 3]
        # Considering unique instantiations of x1, x2, x3 and x4, should be [5, 1, 2, 3], [2, 1, 0, 1], [1, 1, 1, 2], [3, 0, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1]
        # The first instantiation is [1, 1, 1, 2] (2 instances)
        # There are 2 instances where x5 == 1 with x1 == 1, x2 == 1, x3 == 1 and x4 == 2
        result = Nijk(xi, j, k, parents_xi, data)
        expected = 2 
        self.assertEqual(result, expected)

class TestNijFunction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        global data, r, V
        data = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5"])
        data["x1"] = [5, 2, 1, 3, 2, 1, 1] 
        data["x2"] = [1, 1, 1, 0, 1, 1, 1]
        data["x3"] = [2, 0, 1, 2, 1, 1, 1]
        data["x4"] = [3, 1, 2, 1, 1, 2, 1]
        data["x5"] = [1, 1, 1, 1, 1, 1, 1]

        V = [
            [0, 1, 2, 3, 4, 5], # valori assumibili da x1
            [0, 1], # valori assumibili da x2
            [0, 1, 2], # valori assumibili da x3
            [1, 2, 3], # valori assumibili da x4
            [0, 1] # valori assumibili da x5
        ]
        r = np.array([len(values) for values in V])

    def test_no_parents(self):
        # Test when there are no parents
        xi = 0
        j = 0
        parents_xi = []
        # As there are no parents, unique instantiations are empty
        # So, the value of j should be kinda useless
        # It should return the total count of x1
        expected = 7
        result = Nij(xi, parents_xi, j, data, r, V)
        self.assertEqual(result, expected)

    def test_single_parent(self):
        # Test with a single parent
        xi = 2
        j = 1
        # Considering unique instantiations of x1, should be [5], [2], [1], [3]
        # The second instantiation is [2] 
        # The count of x3 with x1 == 2 should be 2 = 1 (k=0) + 1 (k=1) + 0 (k=2)
        parents_xi = [0]
        result = Nij(xi, parents_xi, j, data, r, V)
        expected = 2
        self.assertEqual(result, expected)

    def test_multiple_parents(self):
        # Test with multiple parents
        xi = 2
        j = 2
        parents_xi = [0, 1]
        # Considering unique instantiations of x1 and x2, should be [5, 1], [2, 1], [1, 1], [3, 0]
        # The third instantiation is [1, 1]
        # The count of x3 with x1 == 1 and x2 == 1 should be 3 = 0 (k=0) + 3 (k=1) + 0 (k=2)
        result = Nij(xi, parents_xi, j, data, r, V)
        expected = 3 
        self.assertEqual(result, expected)

    def test_all(self):
        xi = 4
        j = 3
        parents_xi = [0, 1, 2, 3]
        # Considering unique instantiations of x1, x2, x3 and x4, should be [5, 1, 2, 3], [2, 1, 0, 1], [1, 1, 1, 2], [3, 0, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1]
        # The fourth instantiation is [3, 0, 2, 1]
        # 1 (K=1) + 0 (k=0) = 1
        result = Nij(xi, parents_xi, j, data, r, V)
        expected = 1
        self.assertEqual(result, expected)

class TestPredFunction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        global data, r, V, nodes
        data = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5"])
        data["x1"] = [5, 2, 1, 3, 2, 1, 1] 
        data["x2"] = [1, 1, 1, 0, 1, 1, 1]
        data["x3"] = [2, 0, 1, 2, 1, 1, 1]
        data["x4"] = [3, 1, 2, 1, 1, 2, 1]
        data["x5"] = [1, 1, 1, 1, 1, 1, 1]

        V = [
            [0, 1, 2, 3, 4, 5], # valori assumibili da x1
            [0, 1], # valori assumibili da x2
            [0, 1, 2], # valori assumibili da x3
            [1, 2, 3], # valori assumibili da x4
            [0, 1] # valori assumibili da x5
        ]
        r = np.array([len(values) for values in V])
        nodes = [{"id": idx, "name": col, "parents": []} for idx, col in enumerate(data.columns)]
    
    def test_node_with_no_predecessors(self):
        # Test for the first node x1 (ID=0), should return an empty list
        xi = 0
        result = Pred(xi, nodes)
        expected = []
        self.assertEqual(result, expected)

    def test_max_node(self):
        # Test for the maximum node x5 (ID=4), should return all nodes with ID < 4
        xi = 4
        result = Pred(xi, nodes)
        expected = [3, 2, 1, 0] 
        self.assertEqual(result, expected)

    def test_node_with_no_predecessors_intermediate(self):
        # Test for an intermediate node x3 (ID=2), should return all nodes with ID < 2
        xi = 2
        result = Pred(xi, nodes)
        expected = [1, 0]
        self.assertEqual(result, expected)

    def test_random_number(self):
        xi = random.randint(0, 4) # Test for a random node, should return all nodes with ID less than xi
        # ? Obviously, i cannot select a node that is not in the list of nodes!
        expected = [i for i in range(xi - 1, -1, -1)]
        result = Pred(xi, nodes) 
        self.assertEqual(result, expected)

    def test_single_predecessor(self):
        # Test for a node with a single predecessor x2 (ID=1), should return all nodes with ID < 2
        xi = 1
        result = Pred(xi, nodes)
        expected = [0]
        self.assertEqual(result, expected)

class TestParameter_R(unittest.TestCase):

    def test_unique_values_case1(self):

        V = [
            [0, 1], # valori assumibili da x1
            [0, 1], # valori assumibili da x2
            [0, 1], # valori assumibili da x3
            [0, 1] # valori assumibili da x4
        ]

        r = np.array([len(values) for values in V])
        expected = np.array([2, 2, 2, 2])
        np.testing.assert_array_equal(r, expected)

    def test_unique_values_case2(self):
        V = [
            [0, 1, 2], # valori assumibili da x1
            [1], # valori assumibili da x2
            [1, 2, 3, 4, 5, 6], # valori assumibili da x3
            [1, 2, 3, 4] # valori assumibili da x4
        ]
        r = np.array([len(values) for values in V])
        expected = np.array([3, 1, 6, 4])
        np.testing.assert_array_equal(r, expected)

    def test_unique_values_case3(self):

        V = [
            [0, 1], # valori assumibili da x1
            [0, 1, 2, 3], # valori assumibili da x2
            [1, 2], # valori assumibili da x3
            [0, 1, 2] # valori assumibili da x4
        ]
        r = np.array([len(values) for values in V])
        expected = np.array([2, 4, 2, 3])
        np.testing.assert_array_equal(r, expected)

    def test_unique_values_case4(self):
        V = [
            [1], # valori assumibili da x1
            [1, 2, 3, 4], # valori assumibili da x2
            [1, 2, 3], # valori assumibili da x3
            [3, 4] # valori assumibili da x4
        ]
        r = np.array([len(values) for values in V])
        expected = np.array([1, 4, 3, 2])
        np.testing.assert_array_equal(r, expected)

    def test_unique_values_case5(self):
        V = [
            [1, 2, 3], # valori assumibili da x1
            [1, 2], # valori assumibili da x2
            [0, 1], # valori assumibili da x3
            [1, 2] # valori assumibili da x4
        ]
        r = np.array([len(values) for values in V])
        expected = np.array([3, 2, 2, 2])
        np.testing.assert_array_equal(r, expected)