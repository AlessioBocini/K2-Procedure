import numpy as np
import pandas as pd
from mpmath import mp, loggamma
mp.dps = 200

def unique_instantiations(xi, parents_xi, data):
    """
    Finds the unique instantiations (combinations of values) of the parent nodes of node xi.
    
    Parameters:
        xi (int): The ID of the node for which parent instantiations are needed.
        parents_xi (list of int): The IDs of the parent nodes of xi.
        data (pd.DataFrame): The dataset containing the values of the nodes.
        
    Returns:
        np.ndarray: An array of unique combinations of the values of the parents of xi.
    """
    
    if not parents_xi:
        return np.array([[]]) 
    
    parents_xi = sorted(parents_xi)  
    parent_data = data.iloc[:, parents_xi]  

    unique_combinations = parent_data.drop_duplicates().values  # Removes duplicate combinations

    return unique_combinations


def j_th_unque_instantiation(xi, parents_xi, j, data):
    """
    Returns the j-th unique instantiation (combination of values) of the parent nodes of node xi.

    Parameters:
        xi (int): The ID of the node for which the j-th instantiation is needed.
        parents_xi (list of int): The IDs of the parent nodes of xi.
        j (int): The index of the desired unique instantiation.
    data (pd.DataFrame): The dataset containing the values of the nodes.
        
    Returns:
        np.ndarray: The j-th unique combination of values of the parents of xi.

        ? This function doesn't require a test case, as it's a helper function for other functions.
    """

    unique_combinations = unique_instantiations(xi, parents_xi, data) 
    return unique_combinations[j] 

def Nijk(xi, j, k, parents_xi, data):
    """
    Calculates the count of occurrences where node xi takes the value k, 
    given that its parents take the j-th unique instantiation of their values.

    Parameters:
        xi (int): The ID of the node.
        j (int): The index of the unique instantiation of parent nodes.
        k (int): The value of node xi to count.
        parents_xi (list of int): The IDs of the parent nodes of xi.
        data (pd.DataFrame): The dataset containing the values of the nodes.
        
    Returns:
        int: The count of occurrences where xi equals k and the parents of xi match the j-th instantiation.
    """
    parents_xi = sorted(parents_xi) # Sort the parent IDs, to make it consistent with the topological order
    parent_inst = j_th_unque_instantiation(xi, parents_xi, j, data) # Get the j-th unique instantiation of parents
    
    node_data = data.iloc[:, xi] # Select the column of node xi (its values)
    if parents_xi: # If there are parents, select the columns of the parents
        parent_data = data.iloc[:, parents_xi]  # Select parent columns (the values of the parents)
        mask = (parent_data == parent_inst).all(axis=1)  # Check if parents match instantiation
    else:
        mask = pd.Series([True] * len(data))  # If no parents, use entire dataset

    count = node_data[mask].eq(k).sum()  # Count occurrences of xi == k with given parent instantiation
    # ? The mask is used to filter the data to only include rows where the parent nodes match the j-th instantiation
    return count 


def Nij(xi, parents_xi, j, data, r, V):
    """
    Calculates the count of the j-th unique instantiation of the parent nodes of xi in the dataset.

    Parameters:
        xi (int): The ID of the node for which the count is needed.
        parents_xi (list of int): The IDs of the parent nodes of xi.
        j (int): The index of the unique instantiation.
        data (pd.DataFrame): The dataset containing the values of the nodes.
        r (list of int): The number of possible values for each node.
        V (list of list of int): The possible values for each node.
        
    Returns:
        int: The total count of the j-th instantiation in the dataset.
    """
    sum = 0 
    for k_index in range(r[xi]): # Iterate over all possible values of xi
        k = V[xi][k_index] # Get the k-th value of xi
        sum += Nijk(xi, j, k, parents_xi, data) 
    return sum 


def cooper_herkovits_score(xi, parents_xi, data, r, V):
    """
    Computes the Cooper-Herskovits score for a given node xi and its parent nodes.
    This score is used to evaluate how well a set of parents explains the data for a node.

    Parameters:
        xi (int): The ID of the node whose score is being calculated.
        parents_xi (list of int): The IDs of the parent nodes of xi.
        data (pd.DataFrame): The dataset containing the values of the nodes.
        r (list of int): The number of possible values for each node.
        V (list of list of int): The possible values for each node.
        
    Returns:
        float: The Cooper-Herskovits score for the node and its parent set.
    """
    # ? The following code is the original implementation of the Cooper-Herskovits score
    # ? It uses the math library, which is not suitable for very large numbers
    # ? It is kept here for reference, but it is not used in the final implementation
    # score = 1
    # for j in range(len(unique_instantiations(xi, parents_xi))):
    #     factorial = math.factorial((r[xi] - 1)) / math.factorial(Nij(xi, parents_xi, j) + r[xi] - 1)
    #     prod_njk = 1
    #     for k in range(r[xi]):
    #         prod_njk *= math.factorial(Nijk(xi, j, k, parents_xi))
    #     score *= (factorial * prod_njk)
    # return score

    # ? In order to understand what brought to the final implementation, it is important to understand the issues faced:
    # ? 1. The factorial function deals with numbers that are way smaller than 1, so it eventually leads to underflow
    # ? 2. The multiplication of very large numbers leads to overflow, and the result is not accurate
    # ? 3. As we need to deal with too little numbers, the precision of the float data type is not enough
    
    # ? So, in order to solve the issue of underflow and overflow, we need to use the logarithm of the factorial
    # ? The logarithm of the factorial is calculated as loggamma(n+1), which is the natural logarithm of the factorial of n
    # ? The logarithm of the product of the factorials is calculated as the sum of the logarithms of the factorials

    # ? In order to solve the issue of precision, we need to use a library that can handle arbitrary precision
    # ? The mpmath library is used to handle arbitrary precision arithmetic

    # ? The following code is the final implementation of the Cooper-Herskovits score (considering the above optimizations)

    score = mp.mpf(0)
    q = len(unique_instantiations(xi, parents_xi, data))
    for j in range(q):
        nij = Nij(xi, parents_xi, j, data, r, V)
        log_factorial = loggamma((r[xi] - 1) + 1) - loggamma(nij + r[xi])
        log_prod = mp.mpf(0)
        for k in V[xi]:
            log_prod += loggamma(Nijk(xi, j, k, parents_xi, data) + 1)
        score += log_factorial + log_prod
    return score

def Pred(xi, nodes):
    """
    Returns all precedent nodes  a node xi.

    Parameters:
        xi (int): The ID of the node for which previous nodes in the topological order are needed.
    
    Returns: 
        list: A list of IDs of the nodes that are previous nodes in the topological order.
    """

    temp_nodes = [node['id'] for node in nodes if node['id'] < xi] # Get all nodes with ID less than xi
    temp_nodes.sort(reverse=True) # Sort in reverse order to get the topological order [xi-1, ..., 1, 0]
    return temp_nodes

def k2_algorithm(upper_bound, nodes, data, r, V):
    """
    Executes the K2 algorithm to find the best parent set for each node in the Bayesian network.
    The algorithm tries to maximize the Cooper-Herskovits score by adding parents until the upper bound is reached.

    Parameters:
        upper_bound (int): The maximum number of parents allowed for any node.
        nodes (list of dict): A list of nodes in the Bayesian network, each represented as a dictionary with 'id' and 'parents'.
        data (pd.DataFrame): The dataset containing the values of the nodes.
        r (list of int): The number of possible values for each node.
        V (list of list of int): The possible values for each node.
        
    Returns:
        None
    """

    for i in range(len(nodes)):
        node = nodes[i]  # Get the current node
        parents_xi = node["parents"] # Get the current parent set
        xi = node["id"]  # Get the ID of the current node
        p_old = cooper_herkovits_score(xi, parents_xi, data, r, V) 
        OkToProceed = True 
        while OkToProceed and len(parents_xi) < upper_bound:
            best_z = None # Initialize the best parent to None
            predecessors = Pred(xi, nodes) # Get the predecessors of the node in the topological order
            for z in predecessors:
                # ? z is the candidate parent node from the predecessors in the topological order
                if z in parents_xi or z == xi:
                    continue
                new_parents = parents_xi + [z] # Add z to the parent set
                p_new = cooper_herkovits_score(xi, new_parents, data, r, V) 

                if p_new > p_old: # If the new score is better than the old score 
                    p_old = p_new 
                    best_z = z 

            if best_z is not None: # If a better parent was found
                parents_xi.append(best_z) 
            else:
                OkToProceed = False 
