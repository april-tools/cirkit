
if __name__ == "__main__":
    structure = "quad_tree_dec"

    if structure == "quad_tree":
        graph = quad_tree_graph(width=28, height=28, stdec=False)
        serialize(graph, "quad_tree_28_28.json")
    elif structure == "quad_tree_dec":
        graph = quad_tree_graph(width=28, height=28, stdec=True)
        serialize(graph, "quad_tree_stdec_28_28.json")
    elif structure == "poon_domingos":
        pd_num_pieces = [4]
        pd_delta = [[28 / d, 28 / d] for d in pd_num_pieces]
        graph = poon_domingos_structure((28, 28), pd_delta)
        serialize(graph, "poon_domingos_28_28.json")
    else:
        raise AssertionError("Unknown region graph")
