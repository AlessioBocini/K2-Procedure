from k2_core import *
from synthetic_datasets.water_network_dataset import V as V_water, r as r_water, configure_water_dataset, network_structure_difference_water, network_counts_water, expected_node_configuration_water
from k2_test import *
import itertools
import time
from utility_functions import prepare_to_write_on_file

# parameters configuration
upper_bound = 100
n_tests = 2
n_samples = 10000

# Test the K2 algorithm on the CHILD dataset
for i in range(n_tests):

    data = configure_water_dataset(i, n_samples)

    # Creazione della lista di nodi con id, nome e lista di genitori inizialmente vuota
    nodes = [{"id": idx, "name": col, "parents": []} for idx, col in enumerate(data.columns)]

    start_time_water_exec = time.time()
    k2_algorithm(upper_bound, nodes, data, r_water, V_water)
    end_time_water_exec = time.time()
    print("Tempo esecuzione rete WATER: ", end_time_water_exec - start_time_water_exec)
    
    # Create a hashable representation of the network structure
    network_structure = tuple(tuple(sorted(node["parents"])) for node in nodes)

    if network_structure in network_counts_water:
        network_counts_water[network_structure] += 1
    else:
        network_counts_water[network_structure] = 1


    # Count differences between expected_node_configuration_water and nodes
    differences = 0
    for expected, actual in itertools.zip_longest(expected_node_configuration_water, nodes):

        if expected is None or actual is None:
            # Gestisci il caso in cui una delle liste è più corta
            differences += 1
            continue

        parents_expected = sorted(expected["parents"])
        parents_actual = sorted(actual["parents"])

        print(f"Expected: {parents_expected}, Actual: {parents_actual}")
        # Note: Sorting ensures the order is consistent for comparison.

        if len(parents_expected) != len(parents_actual):
            differences += 1
            continue

        for parent_expected, parent_actual in itertools.zip_longest(parents_expected, parents_actual):
            if parent_expected != parent_actual:
                differences += 1

    print(f"Differences between expected and actual nodes: {differences}")
    
    if network_structure not in network_structure_difference_water:
        network_structure_difference_water[network_structure] = differences


    print(f"Current network structure occurred {network_counts_water[network_structure]} time(s)")
    print(f"Total unique structures so far: {len(network_counts_water)}")

with open(f"n_tests_water_{n_samples}.txt", "w") as file:
#    Initialize dictionaries to store network structures, counts, and differences
#    Write network configurations, differences, and occurrences to a .txt file

    differences = prepare_to_write_on_file(network_counts_water, network_structure_difference_water)

    for difference, count in differences.items():
        file.write(f"Difference: {difference}\n")
        file.write(f"Occurrences: {count}\n")
        file.write("\n")



print("Esecuzione TESTS:")
unittest.main()