# IncidenceGraph API Tutorial

The `IncidenceGraph` class implements a layered graph structure using sparse incidence matrices. It supports weighted edges, parallel edges, hyperedges, node-edge hybrid connections, and multi-layer organization.

## Core Concepts

- **Entities**: Nodes and edge-entities (edges that can connect to other edges)
- **Edges**: Binary edges, hyperedges, and node-edge hybrid connections
- **Layers**: Organize subsets of nodes/edges with optional attributes
- **Incidence Matrix**: Rows = entities, Columns = edges. Values indicate connection strength and direction

## Basic Graph Construction

### Initialize Graph
```python
# Directed graph (default)
g = IncidenceGraph(directed=True)

# Undirected graph
g = IncidenceGraph(directed=False)
```

### Add Nodes
```python
# Simple node
g.add_node("A")

# Node with attributes
g.add_node("B", color="red", weight=10)

# Node to specific layer
g.add_node("C", layer="layer1", type="important")
```

### Add Binary Edges
```python
# Basic edge
edge_id = g.add_edge("A", "B")

# Weighted edge
edge_id = g.add_edge("A", "C", weight=2.5)

# Edge with attributes
edge_id = g.add_edge("B", "C", weight=1.0, type="strong", color="blue")

# Specify edge ID (for parallel edges)
g.add_edge("A", "B", edge_id="custom_edge_1")
```

### Add Parallel Edges
```python
# Multiple edges between same nodes
edge1 = g.add_parallel_edge("A", "B", weight=1.0, type="road")
edge2 = g.add_parallel_edge("A", "B", weight=2.0, type="rail")
```

## Hyperedges

### Undirected Hyperedges
```python
# Connect multiple nodes
hyperedge = g.add_hyperedge(members=["A", "B", "C", "D"], weight=1.5)
```

### Directed Hyperedges
```python
# Head nodes point to tail nodes
hyperedge = g.add_hyperedge(
    head=["A", "B"], 
    tail=["C", "D"], 
    weight=2.0,
    type="group_interaction"
)
```

## Node-Edge Hybrid Connections

### Add Edge Entities
```python
# Create an edge that can be connected to
edge_entity = g.add_edge_entity("meta_edge_1", type="connector")
```

### Connect to Edge Entities
```python
# Connect node to edge-entity
g.add_edge("A", "meta_edge_1", edge_type="node_edge")

# Connect edge-entity to another node
g.add_edge("meta_edge_1", "B", edge_type="node_edge")
```

## Layer Management

### Create Layers
```python
# Create empty layer
g.add_layer("temporal_1", timestamp="2024-01-01", phase="initial")

# Set active layer
g.set_active_layer("temporal_1")

# Add nodes/edges to active layer
g.add_node("X")  # Goes to temporal_1
g.add_edge("X", "Y")  # Goes to temporal_1
```

### Layer Operations
```python
# Union of layers
union_result = g.layer_union(["layer1", "layer2"])

# Intersection of layers
intersection_result = g.layer_intersection(["layer1", "layer2"])

# Difference between layers
diff_result = g.layer_difference("layer1", "layer2")

# Create layer from operation result
g.create_layer_from_operation("combined", union_result, type="merged")
```

### Layer Analysis
```python
# Find conserved edges (present in multiple layers)
conserved = g.conserved_edges(min_layers=3)

# Layer-specific edges
specific = g.layer_specific_edges("layer1")

# Edge presence across layers
presence = g.edge_presence_across_layers(source="A", target="B")

# Temporal dynamics
changes = g.temporal_dynamics(["t1", "t2", "t3"], metric='edge_change')
```

## Query Operations

### Basic Graph Info
```python
# Counts
num_nodes = g.number_of_nodes()
num_edges = g.number_of_edges()
degree = g.degree("A")

# Lists
all_nodes = g.nodes()
all_edges = g.edges()
edge_list = g.edge_list()  # (source, target, edge_id, weight) tuples
```

### Neighborhood Queries
```python
# All neighbors (respects directionality)
neighbors = g.neighbors("A")

# Outgoing neighbors only
out_neighbors = g.out_neighbors("A")

# Incoming neighbors only  
in_neighbors = g.in_neighbors("A")
```

### Edge Queries
```python
# Check edge existence
exists = g.has_edge("A", "B")
exists_specific = g.has_edge("A", "B", edge_id="edge_1")

# Get edge IDs between nodes
edge_ids = g.get_edge_ids("A", "B")

# Get directed/undirected edges
directed_edges = g.get_directed_edges()
undirected_edges = g.get_undirected_edges()
```

## Attribute Management

### Set Attributes
```python
# Node attributes
g.set_node_attrs("A", color="red", size=10)

# Edge attributes  
g.set_edge_attrs("edge_1", strength=0.8, type="critical")

# Layer attributes
g.set_layer_attrs("layer1", created_at="2024-01-01")

# Graph-level attributes
g.set_graph_attribute("name", "My Network")
```

### Get Attributes
```python
# Single attribute
color = g.get_node_attr("A", "color", default="unknown")
strength = g.get_edge_attr("edge_1", "strength")

# Graph attribute
name = g.get_graph_attribute("name")
```

### Attribute Views (DataFrames)
```python
# Node attributes as DataFrame
nodes_df = g.nodes_view()

# Edge attributes as DataFrame (with structural info)
edges_df = g.edges_view(layer="layer1", resolved_weight=True)

# Layer attributes as DataFrame
layers_df = g.layers_view()
```

### Layer-Specific Edge Weights
```python
# Set per-layer edge weight
g.set_edge_layer_attrs("layer1", "edge_1", weight=5.0)

# Get effective weight (layer-specific if available, else global)
weight = g.get_effective_edge_weight("edge_1", layer="layer1")
```

## Advanced Operations

### Edge Propagation
```python
# Propagate to layers where both endpoints exist
g.add_edge("A", "B", propagate="shared")

# Propagate to all layers containing either endpoint
g.add_edge("A", "B", propagate="all")
```

### Subgraph Extraction
```python
# Extract single layer as new graph
subgraph = g.subgraph_from_layer("layer1", resolve_layer_weights=True)
```

### Graph Copying
```python
# Deep copy
g_copy = g.copy()
```

### Removal Operations
```python
# Remove edge
g.remove_edge("edge_1")

# Remove node (and incident edges)
g.remove_node("A")

# Remove layer
g.remove_layer("old_layer")
```

## Utility Functions

### Memory and Statistics
```python
# Memory usage estimate
memory_bytes = g.memory_usage()

# Layer statistics
stats = g.layer_statistics(include_default=False)

# Global counts
total_entities = g.global_entity_count()
total_edges = g.global_edge_count()
```

### Attribute Validation
```python
# Check for orphaned/missing attribute rows
audit = g.audit_attributes()
print(audit["extra_node_rows"])  # Nodes in attr table but not in graph
print(audit["invalid_edge_layer_rows"])  # Invalid layer-edge combinations
```

## Usage Patterns

### Time Series Network
```python
g = IncidenceGraph(directed=True)

# Create temporal layers
for t in range(1, 6):
    layer_id = f"time_{t}"
    g.add_layer(layer_id, timestamp=t)
    g.set_active_layer(layer_id)
    
    # Add time-specific nodes and edges
    g.add_node(f"node_{t}")
    if t > 1:
        g.add_edge(f"node_{t-1}", f"node_{t}", weight=1.0)

# Analyze temporal changes
changes = g.temporal_dynamics(
    [f"time_{t}" for t in range(1, 6)], 
    metric='edge_change'
)
```

### Multi-Modal Network
```python
g = IncidenceGraph(directed=False)

# Create modality layers
g.add_layer("social", type="interaction")
g.add_layer("spatial", type="proximity") 
g.add_layer("semantic", type="similarity")

# Add nodes to all layers
for person in ["Alice", "Bob", "Carol"]:
    g.add_node(person, layer="social")
    g.add_node(person, layer="spatial")
    g.add_node(person, layer="semantic")

# Add different types of edges
g.add_edge("Alice", "Bob", layer="social", weight=0.8, type="friend")
g.add_edge("Alice", "Bob", layer="spatial", weight=0.3, type="distance")
g.add_edge("Alice", "Bob", layer="semantic", weight=0.9, type="similarity")

# Find edges conserved across modalities
conserved = g.conserved_edges(min_layers=2)
```

### Hypergraph with Metadata
```python
g = IncidenceGraph(directed=True)

# Add nodes with roles
g.add_node("teacher", role="instructor")
g.add_node("student1", role="learner")
g.add_node("student2", role="learner")
g.add_node("classroom", role="location")

# Create directed hyperedge (teacher -> students in classroom)
hyperedge = g.add_hyperedge(
    head=["teacher", "classroom"],
    tail=["student1", "student2"],
    weight=1.0,
    interaction_type="instruction",
    duration_minutes=60
)
```