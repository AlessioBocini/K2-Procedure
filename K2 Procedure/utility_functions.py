def prepare_to_write_on_file(network_counts, network_structure_difference):
    differences = {}

    for network_structure, count in network_counts.items():
        difference = network_structure_difference.get(network_structure, "N/A")
        if difference is not None:
            if difference not in differences:
                differences[difference] = 0
            differences[difference] += count

    return differences