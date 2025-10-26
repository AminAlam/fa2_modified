"""
Tests for NetworkX integration
"""
import pytest
import numpy as np
import numbers

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from fa2_modified import ForceAtlas2


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
class TestNetworkXIntegration:
    """Test ForceAtlas2 with NetworkX graphs"""
    
    def test_networkx_simple_graph(self):
        """Test with simple NetworkX graph"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
        
        assert len(pos) == 4
        assert all(node in pos for node in G.nodes())
        assert all(len(pos[node]) == 2 for node in G.nodes())
    
    def test_networkx_with_initial_positions(self):
        """Test NetworkX layout with initial positions"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        
        initial_pos = {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (2.0, 0.0)
        }
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, pos=initial_pos, iterations=10)
        
        assert len(pos) == 3
        assert all(node in pos for node in G.nodes())
    
    def test_networkx_weighted_graph(self):
        """Test with weighted NetworkX graph"""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=5.0)
        G.add_edge(2, 0, weight=1.0)
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10, weight_attr='weight')
        
        assert len(pos) == 3
    
    def test_networkx_star_graph(self):
        """Test with star graph (hub topology)"""
        G = nx.star_graph(5)
        
        fa2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        assert len(pos) == 6  # 1 hub + 5 leaves
    
    def test_networkx_complete_graph(self):
        """Test with complete graph"""
        G = nx.complete_graph(5)
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
        
        assert len(pos) == 5
    
    def test_networkx_path_graph(self):
        """Test with path graph"""
        G = nx.path_graph(10)
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        assert len(pos) == 10
    
    def test_networkx_cycle_graph(self):
        """Test with cycle graph"""
        G = nx.cycle_graph(8)
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        assert len(pos) == 8
    
    def test_networkx_grid_graph(self):
        """Test with 2D grid graph"""
        G = nx.grid_2d_graph(3, 3)
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        assert len(pos) == 9
    
    def test_networkx_random_geometric_graph(self):
        """Test with random geometric graph"""
        G = nx.random_geometric_graph(20, 0.3)
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        assert len(pos) == 20
    
    def test_networkx_karate_club(self):
        """Test with Karate Club graph (classic test graph)"""
        G = nx.karate_club_graph()
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=100)
        
        assert len(pos) == 34
    
    def test_networkx_single_node(self):
        """Test with single node"""
        G = nx.Graph()
        G.add_node(0)
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
        
        assert len(pos) == 1
        assert 0 in pos
    
    def test_networkx_disconnected_graph(self):
        """Test with disconnected components"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        G.add_edges_from([(3, 4), (4, 5)])
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        assert len(pos) == 6
    
    def test_networkx_string_node_labels(self):
        """Test with string node labels"""
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
        
        assert len(pos) == 3
        assert 'A' in pos
        assert 'B' in pos
        assert 'C' in pos
    
    def test_networkx_mixed_node_labels(self):
        """Test with mixed node labels"""
        G = nx.Graph()
        G.add_edges_from([(1, 'A'), ('A', 2.5), (2.5, 1)])
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
        
        assert len(pos) == 3
    
    def test_networkx_no_edges(self):
        """Test with nodes but no edges"""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
        
        assert len(pos) == 4
    
    def test_networkx_position_format(self):
        """Test that positions are in correct format"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
        
        assert isinstance(pos, dict)
        for node, position in pos.items():
            assert len(position) == 2
            assert isinstance(position[0], numbers.Real)
            assert isinstance(position[1], numbers.Real)
    
    def test_networkx_parameter_combinations(self):
        """Test various parameter combinations with NetworkX"""
        G = nx.karate_club_graph()
        
        params = [
            {'barnesHutOptimize': True, 'strongGravityMode': False},
            {'barnesHutOptimize': False, 'strongGravityMode': True},
            {'outboundAttractionDistribution': True, 'edgeWeightInfluence': 0.5},
            {'scalingRatio': 5.0, 'gravity': 2.0},
        ]
        
        for param in params:
            fa2 = ForceAtlas2(**param, verbose=False)
            pos = fa2.forceatlas2_networkx_layout(G, iterations=10)
            assert len(pos) == 34
    
    def test_networkx_invalid_input(self):
        """Test that non-NetworkX graph raises error"""
        fa2 = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            fa2.forceatlas2_networkx_layout([[0, 1], [1, 0]], iterations=10)
    
    def test_networkx_invalid_pos_type(self):
        """Test that invalid pos type raises error"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        
        fa2 = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            fa2.forceatlas2_networkx_layout(G, pos=[(0, 0), (1, 1)], iterations=10)


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
class TestNetworkXEdgeWeights:
    """Test edge weight handling in NetworkX graphs"""
    
    def test_default_weight_attribute(self):
        """Test with default weight attribute"""
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(1, 2, weight=1.0)
        
        fa2 = ForceAtlas2(edgeWeightInfluence=1.0, verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        assert len(pos) == 3
    
    def test_custom_weight_attribute(self):
        """Test with custom weight attribute"""
        G = nx.Graph()
        G.add_edge(0, 1, strength=3.0)
        G.add_edge(1, 2, strength=1.0)
        
        fa2 = ForceAtlas2(edgeWeightInfluence=1.0, verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50, weight_attr='strength')
        
        assert len(pos) == 3
    
    def test_no_weight_attribute(self):
        """Test with no weight attribute (unweighted)"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2_networkx_layout(G, iterations=50, weight_attr=None)
        
        assert len(pos) == 3


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
class TestNetworkXReproducibility:
    """Test reproducibility with NetworkX graphs"""
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same NetworkX layout"""
        G = nx.karate_club_graph()
        
        fa2 = ForceAtlas2(verbose=False)
        
        # First run
        np.random.seed(42)
        import random
        random.seed(42)
        pos1 = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        # Second run with same seed
        np.random.seed(42)
        random.seed(42)
        pos2 = fa2.forceatlas2_networkx_layout(G, iterations=50)
        
        # Results should be identical
        for node in G.nodes():
            assert abs(pos1[node][0] - pos2[node][0]) < 1e-10
            assert abs(pos1[node][1] - pos2[node][1]) < 1e-10

