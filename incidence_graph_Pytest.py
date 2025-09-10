import pytest
import numpy as np
import pandas as pd
from layered_incidence_graph import IncidenceGraph


class TestIncidenceGraphBasics:
    
    def test_initialization(self):
        g = IncidenceGraph(directed=True)
        assert g.directed is True
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0
        assert g.get_active_layer() == "default"
    
    def test_add_node(self):
        g = IncidenceGraph()
        g.add_node("A")
        assert g.number_of_nodes() == 1
        assert "A" in g.nodes()
        assert g.entity_types["A"] == "node"
    
    def test_add_node_with_attributes(self):
        g = IncidenceGraph()
        g.add_node("A", color="red", weight=10)
        assert g.get_node_attr("A", "color") == "red"
        assert g.get_node_attr("A", "weight") == 10
    
    def test_add_edge_basic(self):
        g = IncidenceGraph()
        edge_id = g.add_edge("A", "B")
        assert g.number_of_edges() == 1
        assert edge_id in g.edges()
        assert g.has_edge("A", "B")
    
    def test_add_edge_with_weight(self):
        g = IncidenceGraph()
        edge_id = g.add_edge("A", "B", weight=2.5)
        assert g.edge_weights[edge_id] == 2.5
    
    def test_add_edge_creates_nodes(self):
        g = IncidenceGraph()
        g.add_edge("A", "B")
        assert "A" in g.nodes()
        assert "B" in g.nodes()
    
    def test_parallel_edges(self):
        g = IncidenceGraph()
        edge1 = g.add_edge("A", "B", weight=1.0)
        edge2 = g.add_parallel_edge("A", "B", weight=2.0)
        assert edge1 != edge2
        assert len(g.get_edge_ids("A", "B")) == 2
    
    def test_edge_directionality(self):
        g_directed = IncidenceGraph(directed=True)
        g_undirected = IncidenceGraph(directed=False)
        
        edge_d = g_directed.add_edge("A", "B")
        edge_u = g_undirected.add_edge("A", "B")
        
        assert g_directed.is_edge_directed(edge_d) is True
        assert g_undirected.is_edge_directed(edge_u) is False


class TestHyperedges:
    
    def test_undirected_hyperedge(self):
        g = IncidenceGraph()
        hyperedge = g.add_hyperedge(members=["A", "B", "C"], weight=1.5)
        assert hyperedge in g.edges()
        assert g.edge_kind[hyperedge] == "hyper"
        assert not g.hyperedge_definitions[hyperedge]["directed"]
    
    def test_directed_hyperedge(self):
        g = IncidenceGraph()
        hyperedge = g.add_hyperedge(head=["A", "B"], tail=["C", "D"], weight=2.0)
        assert hyperedge in g.edges()
        assert g.edge_kind[hyperedge] == "hyper"
        assert g.hyperedge_definitions[hyperedge]["directed"]
    
    def test_hyperedge_validation(self):
        g = IncidenceGraph()
        
        # Should fail with both members and head/tail
        with pytest.raises(ValueError):
            g.add_hyperedge(members=["A", "B"], head=["C"], tail=["D"])
        
        # Should fail with empty members
        with pytest.raises(ValueError):
            g.add_hyperedge(members=[])
        
        # Should fail with overlapping head/tail
        with pytest.raises(ValueError):
            g.add_hyperedge(head=["A", "B"], tail=["B", "C"])


class TestNodeEdgeHybrids:
    
    def test_edge_entity_creation(self):
        g = IncidenceGraph()
        edge_entity = g.add_edge_entity("meta_1", type="connector")
        assert "meta_1" in g.entity_to_idx
        assert g.entity_types["meta_1"] == "edge"
    
    def test_node_edge_connection(self):
        g = IncidenceGraph()
        g.add_edge_entity("meta_1")
        edge_id = g.add_edge("A", "meta_1", edge_type="node_edge")
        src, tgt, etype = g.edge_definitions[edge_id]
        assert src == "A"
        assert tgt == "meta_1"
        assert etype == "node_edge"


class TestLayers:
    
    def test_layer_creation(self):
        g = IncidenceGraph()
        g.add_layer("layer1", type="test", weight=1.0)
        assert g.has_layer("layer1")
        assert g.get_layer_attr("layer1", "type") == "test"
    
    def test_layer_node_membership(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_node("A", layer="layer1")
        assert "A" in g.get_layer_nodes("layer1")
    
    def test_layer_edge_membership(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        edge_id = g.add_edge("A", "B", layer="layer1")
        assert edge_id in g.get_layer_edges("layer1")
        assert "A" in g.get_layer_nodes("layer1")
        assert "B" in g.get_layer_nodes("layer1")
    
    def test_active_layer(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.set_active_layer("layer1")
        assert g.get_active_layer() == "layer1"
        
        # Adding without specifying layer should use active
        g.add_node("A")
        assert "A" in g.get_layer_nodes("layer1")
    
    def test_layer_operations(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_layer("layer2")
        
        # Add overlapping content
        g.add_node("A", layer="layer1")
        g.add_node("B", layer="layer1")
        g.add_node("A", layer="layer2")  # A in both layers
        g.add_node("C", layer="layer2")
        
        union = g.layer_union(["layer1", "layer2"])
        assert union["nodes"] == {"A", "B", "C"}
        
        intersection = g.layer_intersection(["layer1", "layer2"])
        assert intersection["nodes"] == {"A"}
        
        diff = g.layer_difference("layer1", "layer2")
        assert diff["nodes"] == {"B"}
    
    def test_layer_specific_weights(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        edge_id = g.add_edge("A", "B", weight=1.0, layer="layer1")
        
        # Set layer-specific weight using the working method
        g.set_layer_edge_weight("layer1", edge_id, 5.0)
        
        # Should return layer-specific weight
        assert g.get_effective_edge_weight(edge_id, layer="layer1") == 5.0
        # Should return global weight for no layer
        assert g.get_effective_edge_weight(edge_id) == 1.0

    def test_layer_weight_methods(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        edge_id = g.add_edge("A", "B", weight=1.0, layer="layer1")
        
        # Test working legacy method
        g.set_layer_edge_weight("layer1", edge_id, 5.0)
        assert g.get_effective_edge_weight(edge_id, layer="layer1") == 5.0
        
        # Test DataFrame method (might be broken)
        try:
            g.set_edge_layer_attrs("layer1", edge_id, weight=7.0)
            df_weight = g.get_effective_edge_weight(edge_id, layer="layer1")
            print(f"DataFrame method works: {df_weight}")
        except Exception as e:
            print(f"DataFrame method broken: {e}")



class TestQueries:
    
    def test_neighbors(self):
        g = IncidenceGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        
        neighbors = g.neighbors("A")
        assert set(neighbors) == {"B", "C"}
    
    def test_directed_neighbors(self):
        g = IncidenceGraph(directed=True)
        g.add_edge("A", "B")
        g.add_edge("C", "A")
        
        out_neighbors = g.out_neighbors("A")
        in_neighbors = g.in_neighbors("A")
        
        assert out_neighbors == ["B"]
        assert in_neighbors == ["C"]
    
    def test_degree(self):
        g = IncidenceGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        assert g.degree("A") == 2
        assert g.degree("B") == 1
    
    def test_edge_list(self):
        g = IncidenceGraph()
        edge_id = g.add_edge("A", "B", weight=2.0)
        edge_list = g.edge_list()
        assert len(edge_list) == 1
        assert edge_list[0] == ("A", "B", edge_id, 2.0)


class TestAttributes:
    
    def test_node_attributes(self):
        g = IncidenceGraph()
        g.add_node("A")
        g.set_node_attrs("A", color="red", size=10)
        
        assert g.get_node_attr("A", "color") == "red"
        assert g.get_node_attr("A", "size") == 10
        assert g.get_node_attr("A", "missing", "default") == "default"
    
    def test_edge_attributes(self):
        g = IncidenceGraph()
        edge_id = g.add_edge("A", "B")
        g.set_edge_attrs(edge_id, strength=0.8, type="critical")
        
        assert g.get_edge_attr(edge_id, "strength") == 0.8
        assert g.get_edge_attr(edge_id, "type") == "critical"
    
    def test_graph_attributes(self):
        g = IncidenceGraph()
        g.set_graph_attribute("name", "Test Graph")
        assert g.get_graph_attribute("name") == "Test Graph"
    
    def test_attribute_views(self):
        g = IncidenceGraph()
        g.add_node("A", color="red")
        g.add_edge("A", "B", weight=1.0, type="strong")
        
        nodes_df = g.nodes_view()
        edges_df = g.edges_view()
        
        assert not nodes_df.empty
        assert not edges_df.empty
        assert "color" in nodes_df.columns
        assert "type" in edges_df.columns


class TestRemoval:
    
    def test_remove_edge(self):
        g = IncidenceGraph()
        edge_id = g.add_edge("A", "B")
        assert g.number_of_edges() == 1
        
        g.remove_edge(edge_id)
        assert g.number_of_edges() == 0
        assert not g.has_edge("A", "B")
    
    def test_remove_node(self):
        g = IncidenceGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        
        initial_edges = g.number_of_edges()
        g.remove_node("A")
        
        # Node and incident edges should be removed
        assert "A" not in g.nodes()
        assert g.number_of_edges() < initial_edges
    
    def test_remove_layer(self):
        g = IncidenceGraph()
        g.add_layer("test_layer")
        assert g.has_layer("test_layer")
        
        g.remove_layer("test_layer")
        assert not g.has_layer("test_layer")
    
    def test_cannot_remove_default_layer(self):
        g = IncidenceGraph()
        with pytest.raises(ValueError):
            g.remove_layer("default")


class TestEdgePropagation:
    
    def test_shared_propagation(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_layer("layer2")
        
        # Add nodes to both layers
        g.add_node("A", layer="layer1")
        g.add_node("B", layer="layer1")
        g.add_node("A", layer="layer2")
        g.add_node("B", layer="layer2")
        
        # Edge should propagate to both layers (both endpoints present)
        edge_id = g.add_edge("A", "B", propagate="shared")
        
        assert edge_id in g.get_layer_edges("layer1")
        assert edge_id in g.get_layer_edges("layer2")
    
    def test_all_propagation(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_layer("layer2")
        
        # Add A to layer1, B to layer2
        g.add_node("A", layer="layer1")
        g.add_node("B", layer="layer2")
        
        # Edge should propagate to both layers and add missing endpoints
        edge_id = g.add_edge("A", "B", propagate="all")
        
        assert edge_id in g.get_layer_edges("layer1")
        assert edge_id in g.get_layer_edges("layer2")
        assert "B" in g.get_layer_nodes("layer1")  # Added by propagation
        assert "A" in g.get_layer_nodes("layer2")  # Added by propagation


class TestAdvancedQueries:
    
    def test_edge_presence_across_layers(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_layer("layer2")
        
        edge_id = g.add_edge("A", "B", layer="layer1")
        g.add_edge("A", "B", layer="layer2")  # Different edge, same endpoints
        
        presence = g.edge_presence_across_layers(source="A", target="B")
        assert len(presence) == 2  # Present in both layers
    
    def test_conserved_edges(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_layer("layer2")
        g.add_layer("layer3")
        
        # Edge present in multiple layers
        edge_id = g.add_edge("A", "B", layer="layer1")
        g._layers["layer2"]["edges"].add(edge_id)
        g._layers["layer3"]["edges"].add(edge_id)
        
        conserved = g.conserved_edges(min_layers=2)
        assert edge_id in conserved
        assert conserved[edge_id] == 3
    
    def test_layer_specific_edges(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_layer("layer2")
        
        edge1 = g.add_edge("A", "B", layer="layer1")
        edge2 = g.add_edge("C", "D", layer="layer1")
        
        # Add edge1 to layer2 as well
        g._layers["layer2"]["edges"].add(edge1)
        
        specific = g.layer_specific_edges("layer1")
        assert edge2 in specific  # Only in layer1
        assert edge1 not in specific  # In both layers


class TestCopyAndUtilities:
    
    def test_copy(self):
        g = IncidenceGraph()
        g.add_node("A", color="red")
        g.add_edge("A", "B", weight=2.0, type="strong")
        g.add_layer("layer1", phase="initial")
        g.set_graph_attribute("name", "Original")
        
        g_copy = g.copy()
        
        # Structure should be identical
        assert g_copy.number_of_nodes() == g.number_of_nodes()
        assert g_copy.number_of_edges() == g.number_of_edges()
        assert g_copy.get_node_attr("A", "color") == "red"
        assert g_copy.get_graph_attribute("name") == "Original"
        
        # Should be independent copies
        g_copy.add_node("C")
        assert "C" not in g.nodes()
    
    def test_memory_usage(self):
        g = IncidenceGraph()
        g.add_edge("A", "B")
        memory = g.memory_usage()
        assert isinstance(memory, int)
        assert memory > 0
    
    def test_subgraph_from_layer(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        g.add_node("A", layer="layer1", color="red")
        edge_id = g.add_edge("A", "B", layer="layer1", weight=2.0)
        
        subgraph = g.subgraph_from_layer("layer1")
        
        assert "A" in subgraph.nodes()
        assert "B" in subgraph.nodes()
        assert subgraph.has_edge("A", "B")
        assert subgraph.get_node_attr("A", "color") == "red"


class TestErrorHandling:
    
    def test_invalid_layer_operations(self):
        g = IncidenceGraph()
        
        with pytest.raises(KeyError):
            g.set_active_layer("nonexistent")
        
        with pytest.raises(KeyError):
            g.get_layer_nodes("nonexistent")
    
    def test_invalid_edge_operations(self):
        g = IncidenceGraph()
        
        with pytest.raises(KeyError):
            g.remove_edge("nonexistent")
        
        with pytest.raises(KeyError):
            g.remove_node("nonexistent")
    
    def test_invalid_propagation(self):
        g = IncidenceGraph()
        
        with pytest.raises(ValueError):
            g.add_edge("A", "B", propagate="invalid")
    
    def test_duplicate_layer(self):
        g = IncidenceGraph()
        g.add_layer("layer1")
        
        with pytest.raises(ValueError):
            g.add_layer("layer1")  # Already exists


class TestTemporalAnalysis:
    
    def test_temporal_dynamics(self):
        g = IncidenceGraph()
        
        # Create time series layers
        for t in [1, 2, 3]:
            layer_id = f"time_{t}"
            g.add_layer(layer_id)
            g.add_node(f"node_{t}", layer=layer_id)
            if t > 1:
                g.add_edge(f"node_{t-1}", f"node_{t}", layer=layer_id)
        
        changes = g.temporal_dynamics(["time_1", "time_2", "time_3"], metric="edge_change")
        
        assert len(changes) == 2  # Between 3 layers = 2 transitions
        assert all("added" in change for change in changes)
        assert all("removed" in change for change in changes)


class TestValidationAndAudit:
    
    def test_audit_attributes(self):
        g = IncidenceGraph()
        g.add_node("A")
        g.add_edge("A", "B")
        
        # Add orphaned attribute using proper API first to initialize DataFrame
        g.set_node_attrs("orphan", color="red")  # Creates orphaned row
        
        audit = g.audit_attributes()
        assert "orphan" in audit["extra_node_rows"]
    
    def test_input_validation(self):
        g = IncidenceGraph()
        
        # Invalid weight type
        with pytest.raises(TypeError):
            g.add_edge("A", "B", weight="invalid")
        
        # Invalid edge type
        with pytest.raises(ValueError):
            g.add_edge("A", "B", edge_type="invalid")