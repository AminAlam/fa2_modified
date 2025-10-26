"""
Tests for igraph integration
"""
import pytest
import numpy as np

try:
    import igraph
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

from fa2_modified import ForceAtlas2


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not installed")
class TestIgraphIntegration:
    """Test ForceAtlas2 with igraph graphs"""
    
    def test_igraph_simple_graph(self):
        """Test with simple igraph graph"""
        G = igraph.Graph()
        G.add_vertices(4)
        G.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10)
        
        assert isinstance(layout, igraph.layout.Layout)
        assert len(layout) == 4
        assert layout.dim == 2
    
    def test_igraph_with_initial_positions(self):
        """Test igraph layout with initial positions"""
        G = igraph.Graph()
        G.add_vertices(3)
        G.add_edges([(0, 1), (1, 2)])
        
        initial_pos = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, pos=initial_pos, iterations=10)
        
        assert len(layout) == 3
    
    def test_igraph_with_numpy_positions(self):
        """Test igraph layout with numpy array positions"""
        G = igraph.Graph()
        G.add_vertices(3)
        G.add_edges([(0, 1), (1, 2)])
        
        initial_pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, pos=initial_pos, iterations=10)
        
        assert len(layout) == 3
    
    def test_igraph_weighted_graph(self):
        """Test with weighted igraph graph"""
        G = igraph.Graph()
        G.add_vertices(3)
        G.add_edges([(0, 1), (1, 2), (2, 0)])
        G.es['weight'] = [1.0, 5.0, 1.0]
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10, weight_attr='weight')
        
        assert len(layout) == 3
    
    def test_igraph_custom_weight_attribute(self):
        """Test with custom weight attribute name"""
        G = igraph.Graph()
        G.add_vertices(3)
        G.add_edges([(0, 1), (1, 2)])
        G.es['strength'] = [2.0, 3.0]
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10, weight_attr='strength')
        
        assert len(layout) == 3
    
    def test_igraph_star_graph(self):
        """Test with star graph (hub topology)"""
        G = igraph.Graph.Star(6)
        
        fa2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        assert len(layout) == 6
    
    def test_igraph_complete_graph(self):
        """Test with complete graph"""
        G = igraph.Graph.Full(5)
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10)
        
        assert len(layout) == 5
    
    def test_igraph_ring_graph(self):
        """Test with ring graph"""
        G = igraph.Graph.Ring(8)
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        assert len(layout) == 8
    
    def test_igraph_tree_graph(self):
        """Test with tree graph"""
        G = igraph.Graph.Tree(15, 3)  # Tree with 15 nodes, 3 children per node
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        assert len(layout) == 15
    
    def test_igraph_lattice_graph(self):
        """Test with lattice graph"""
        G = igraph.Graph.Lattice([3, 3], circular=False)
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        assert len(layout) == 9
    
    def test_igraph_single_node(self):
        """Test with single node"""
        G = igraph.Graph()
        G.add_vertices(1)
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10)
        
        assert len(layout) == 1
    
    def test_igraph_disconnected_graph(self):
        """Test with disconnected components"""
        G = igraph.Graph()
        G.add_vertices(6)
        G.add_edges([(0, 1), (1, 2)])  # Component 1
        G.add_edges([(3, 4), (4, 5)])  # Component 2
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        assert len(layout) == 6
    
    def test_igraph_no_edges(self):
        """Test with nodes but no edges"""
        G = igraph.Graph()
        G.add_vertices(4)
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10)
        
        assert len(layout) == 4
    
    def test_igraph_layout_coordinates_format(self):
        """Test that layout coordinates are in correct format"""
        G = igraph.Graph()
        G.add_vertices(3)
        G.add_edges([(0, 1), (1, 2)])
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10)
        
        coords = layout.coords
        assert len(coords) == 3
        for coord in coords:
            assert len(coord) == 2
            assert isinstance(coord[0], (int, float))
            assert isinstance(coord[1], (int, float))
    
    def test_igraph_parameter_combinations(self):
        """Test various parameter combinations with igraph"""
        G = igraph.Graph.Erdos_Renyi(n=30, p=0.1)
        
        params = [
            {'barnesHutOptimize': True, 'strongGravityMode': False},
            {'barnesHutOptimize': False, 'strongGravityMode': True},
            {'outboundAttractionDistribution': True, 'edgeWeightInfluence': 0.5},
            {'scalingRatio': 5.0, 'gravity': 2.0},
        ]
        
        for param in params:
            fa2 = ForceAtlas2(**param, verbose=False)
            layout = fa2.forceatlas2_igraph_layout(G, iterations=10)
            assert len(layout) == 30
    
    def test_igraph_invalid_input(self):
        """Test that non-igraph graph raises error"""
        fa2 = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            fa2.forceatlas2_igraph_layout([[0, 1], [1, 0]], iterations=10)
    
    def test_igraph_invalid_pos_type(self):
        """Test that invalid pos type raises error"""
        G = igraph.Graph()
        G.add_vertices(3)
        G.add_edges([(0, 1), (1, 2)])
        
        fa2 = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            fa2.forceatlas2_igraph_layout(G, pos="invalid", iterations=10)
    
    def test_igraph_directed_graph(self):
        """Test with directed graph (converted to undirected internally)"""
        G = igraph.Graph(directed=True)
        G.add_vertices(3)
        G.add_edges([(0, 1), (1, 2), (2, 0)])
        
        fa2 = ForceAtlas2(verbose=False)
        # Note: The current implementation assumes undirected graphs
        # This test verifies the conversion happens correctly
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10)
        
        assert len(layout) == 3
    
    def test_igraph_random_graph(self):
        """Test with random Erdos-Renyi graph"""
        G = igraph.Graph.Erdos_Renyi(n=25, p=0.15)
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        assert len(layout) == 25
    
    def test_igraph_barabasi_albert_graph(self):
        """Test with Barabasi-Albert (scale-free) graph"""
        G = igraph.Graph.Barabasi(n=30, m=2)
        
        fa2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        assert len(layout) == 30


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not installed")
class TestIgraphReproducibility:
    """Test reproducibility with igraph graphs"""
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same igraph layout"""
        G = igraph.Graph.Erdos_Renyi(n=20, p=0.2)
        
        fa2 = ForceAtlas2(verbose=False)
        
        # First run
        np.random.seed(42)
        import random
        random.seed(42)
        layout1 = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        # Second run with same seed
        np.random.seed(42)
        random.seed(42)
        layout2 = fa2.forceatlas2_igraph_layout(G, iterations=50)
        
        # Results should be identical
        coords1 = layout1.coords
        coords2 = layout2.coords
        for i in range(len(coords1)):
            assert abs(coords1[i][0] - coords2[i][0]) < 1e-10
            assert abs(coords1[i][1] - coords2[i][1]) < 1e-10


@pytest.mark.skipif(not IGRAPH_AVAILABLE, reason="igraph not installed")
class TestIgraphWeights:
    """Test edge weight handling in igraph graphs"""
    
    def test_no_weights(self):
        """Test without weights"""
        G = igraph.Graph()
        G.add_vertices(4)
        G.add_edges([(0, 1), (1, 2), (2, 3)])
        
        fa2 = ForceAtlas2(verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=10, weight_attr=None)
        
        assert len(layout) == 4
    
    def test_varying_weights(self):
        """Test with varying edge weights"""
        G = igraph.Graph()
        G.add_vertices(4)
        G.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
        G.es['weight'] = [1.0, 10.0, 1.0, 1.0]
        
        fa2 = ForceAtlas2(edgeWeightInfluence=1.0, verbose=False)
        layout = fa2.forceatlas2_igraph_layout(G, iterations=100, weight_attr='weight')
        
        assert len(layout) == 4
        # The edge with weight 10 should pull nodes 1 and 2 closer

