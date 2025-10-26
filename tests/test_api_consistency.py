"""
API consistency and error handling tests
Tests inspired by external ForceAtlas2 implementations
"""
import pytest
import numpy as np
import scipy.sparse
import numbers

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from fa2_modified import ForceAtlas2


class TestForceAtlas2APIConsistency:
    """Test API consistency across different input types"""
    
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_api_consistency_random_vs_custom_positions(self):
        """
        Run ForceAtlas2 layout on the same graph twice:
        once with random initial positions and once with custom positions.
        Both should return valid dictionaries with proper structure.
        """
        G = nx.path_graph(10)
        forceatlas = ForceAtlas2(verbose=False)
        
        # Layout with random initial positions
        pos1 = forceatlas.forceatlas2_networkx_layout(G, pos=None, iterations=10)
        
        # Layout with user-defined positions
        pos_initial = {n: (0.1 * n, 0.1 * n) for n in G.nodes()}
        pos2 = forceatlas.forceatlas2_networkx_layout(G, pos=pos_initial, iterations=10)
        
        # Both outputs must be dictionaries with 10 items
        assert isinstance(pos1, dict)
        assert isinstance(pos2, dict)
        assert len(pos1) == 10
        assert len(pos2) == 10
        
        # Keys should match
        assert set(pos1.keys()) == set(pos2.keys())
        
        # All values should be valid (x,y) tuples
        for p in pos1.values():
            assert isinstance(p, tuple)
            assert len(p) == 2
            assert all(isinstance(coord, numbers.Real) for coord in p)
        
        for p in pos2.values():
            assert isinstance(p, tuple)
            assert len(p) == 2
            assert all(isinstance(coord, numbers.Real) for coord in p)
    
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_api_consistency_custom_parameters(self):
        """
        Test ForceAtlas2 with various custom parameters to ensure
        API consistency across different configurations.
        """
        G = nx.complete_graph(5)
        pos = {i: (float(i), float(i)) for i in G.nodes()}
        
        forceatlas = ForceAtlas2(
            outboundAttractionDistribution=True,
            edgeWeightInfluence=0.5,
            jitterTolerance=5.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.5,
            scalingRatio=1.0,
            strongGravityMode=True,
            gravity=0.5,
            verbose=False
        )
        
        new_pos = forceatlas.forceatlas2_networkx_layout(G, pos=pos, iterations=20)
        
        # Verify output structure
        assert isinstance(new_pos, dict)
        assert len(new_pos) == 5
        
        for key, value in new_pos.items():
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert all(isinstance(coord, numbers.Real) for coord in value)
    
    def test_api_consistency_numpy_sparse(self):
        """Test that numpy array and sparse matrix give compatible results"""
        # Create same graph as numpy array and sparse matrix
        G_numpy = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=float)
        
        G_sparse = scipy.sparse.lil_matrix(G_numpy)
        
        forceatlas = ForceAtlas2(verbose=False)
        
        # Use same seed for both
        np.random.seed(42)
        import random
        random.seed(42)
        pos_numpy = forceatlas.forceatlas2(G_numpy, iterations=10)
        
        np.random.seed(42)
        random.seed(42)
        pos_sparse = forceatlas.forceatlas2(G_sparse, iterations=10)
        
        # Both should return same number of positions
        assert len(pos_numpy) == len(pos_sparse) == 4
        
        # Positions should be very similar (identical with same seed)
        for i in range(4):
            assert pos_numpy[i][0] == pytest.approx(pos_sparse[i][0], abs=1e-10)
            assert pos_numpy[i][1] == pytest.approx(pos_sparse[i][1], abs=1e-10)


class TestForceAtlas2ErrorHandling:
    """Test error handling for invalid inputs"""
    
    def test_invalid_matrix_non_square(self):
        """Test that non-square matrix raises AssertionError"""
        forceatlas = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            G = np.array([[0, 1], [1, 0], [0, 1]], dtype=float)
            forceatlas.init(G, pos=None)
    
    def test_invalid_matrix_non_symmetric(self):
        """Test that non-symmetric matrix raises AssertionError"""
        forceatlas = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            G = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
            forceatlas.init(G, pos=None)
    
    def test_invalid_position_type(self):
        """Test that invalid position type raises AssertionError"""
        forceatlas = ForceAtlas2(verbose=False)
        G = np.array([[0, 1], [1, 0]], dtype=float)
        
        with pytest.raises(AssertionError):
            forceatlas.init(G, pos="invalid")
    
    def test_invalid_position_shape(self):
        """Test that position array with wrong shape raises error"""
        forceatlas = ForceAtlas2(verbose=False)
        G = np.array([[0, 1], [1, 0]], dtype=float)
        
        # Position array with wrong number of nodes
        with pytest.raises((AssertionError, IndexError, ValueError)):
            pos = np.array([[0.0, 0.0]])  # Only 1 position for 2 nodes
            forceatlas.forceatlas2(G, pos=pos, iterations=10)
    
    def test_invalid_graph_type(self):
        """Test that invalid graph type raises AssertionError"""
        forceatlas = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            forceatlas.init([[0, 1], [1, 0]], pos=None)
    
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_networkx_invalid_graph_type(self):
        """Test that non-NetworkX object raises AssertionError"""
        forceatlas = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            forceatlas.forceatlas2_networkx_layout(
                "not a NetworkX graph",
                pos=None,
                iterations=10
            )
    
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_networkx_invalid_position_type(self):
        """Test that invalid position type for NetworkX raises AssertionError"""
        G = nx.complete_graph(5)
        forceatlas = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            forceatlas.forceatlas2_networkx_layout(
                G,
                pos="invalid",
                iterations=10
            )
    
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_networkx_position_list_not_dict(self):
        """Test that position as list (not dict) raises AssertionError"""
        G = nx.complete_graph(3)
        forceatlas = ForceAtlas2(verbose=False)
        
        with pytest.raises(AssertionError):
            forceatlas.forceatlas2_networkx_layout(
                G,
                pos=[(0, 0), (1, 1), (2, 2)],
                iterations=10
            )


class TestForceAtlas2InitializationEdgeCases:
    """Test edge cases in ForceAtlas2 initialization"""
    
    def test_init_returns_correct_types(self):
        """Test that init returns lists of correct types"""
        G = np.array([[0, 1], [1, 0]], dtype=float)
        forceatlas = ForceAtlas2(verbose=False)
        nodes, edges = forceatlas.init(G, pos=None)
        
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        assert all(hasattr(n, 'mass') for n in nodes)
        assert all(hasattr(e, 'weight') for e in edges)
    
    def test_init_edge_count(self):
        """Test that edges are counted correctly (undirected)"""
        # Triangle graph: 3 edges
        G = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=float)
        
        forceatlas = ForceAtlas2(verbose=False)
        nodes, edges = forceatlas.init(G, pos=None)
        
        # Should have 3 edges (not 6, since we only count upper triangle)
        assert len(edges) == 3
    
    def test_init_node_mass_calculation(self):
        """Test that node mass is calculated as 1 + degree"""
        G = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=float)
        
        forceatlas = ForceAtlas2(verbose=False)
        nodes, edges = forceatlas.init(G, pos=None)
        
        # Node 0 has degree 3, so mass should be 4
        assert nodes[0].mass == 4.0
        
        # Nodes 1, 2, 3 each have degree 1, so mass should be 2
        assert nodes[1].mass == 2.0
        assert nodes[2].mass == 2.0
        assert nodes[3].mass == 2.0
    
    def test_init_with_weighted_edges(self):
        """Test that weighted edges preserve their weights"""
        G = np.array([
            [0.0, 2.5, 0.0],
            [2.5, 0.0, 3.7],
            [0.0, 3.7, 0.0]
        ], dtype=float)
        
        forceatlas = ForceAtlas2(verbose=False)
        nodes, edges = forceatlas.init(G, pos=None)
        
        # Check that edge weights are preserved
        weights = sorted([e.weight for e in edges])
        assert weights == pytest.approx([2.5, 3.7])
    
    def test_init_with_custom_positions(self):
        """Test that custom positions are used correctly"""
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[1.5, 2.5], [3.5, 4.5]])
        
        forceatlas = ForceAtlas2(verbose=False)
        nodes, edges = forceatlas.init(G, pos=pos)
        
        assert nodes[0].x == pytest.approx(1.5)
        assert nodes[0].y == pytest.approx(2.5)
        assert nodes[1].x == pytest.approx(3.5)
        assert nodes[1].y == pytest.approx(4.5)


class TestForceAtlas2OutputFormat:
    """Test output format consistency"""
    
    def test_output_is_list_of_tuples(self):
        """Test that forceatlas2 returns list of (x,y) tuples"""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        forceatlas = ForceAtlas2(verbose=False)
        positions = forceatlas.forceatlas2(G, iterations=10)
        
        assert isinstance(positions, list)
        assert len(positions) == 3
        
        for pos in positions:
            assert isinstance(pos, tuple)
            assert len(pos) == 2
            assert isinstance(pos[0], numbers.Real)
            assert isinstance(pos[1], numbers.Real)
    
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_networkx_output_is_dict(self):
        """Test that NetworkX layout returns dict"""
        G = nx.path_graph(5)
        forceatlas = ForceAtlas2(verbose=False)
        positions = forceatlas.forceatlas2_networkx_layout(G, iterations=10)
        
        assert isinstance(positions, dict)
        assert len(positions) == 5
        assert set(positions.keys()) == set(G.nodes())
        
        for node, pos in positions.items():
            assert isinstance(pos, tuple)
            assert len(pos) == 2
            assert isinstance(pos[0], numbers.Real)
            assert isinstance(pos[1], numbers.Real)
    
    def test_positions_are_finite(self):
        """Test that all positions are finite (not NaN or inf)"""
        G = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=float)
        
        forceatlas = ForceAtlas2(verbose=False)
        positions = forceatlas.forceatlas2(G, iterations=50)
        
        for pos in positions:
            assert np.isfinite(pos[0])
            assert np.isfinite(pos[1])
    
    def test_positions_are_not_all_same(self):
        """Test that positions are not all identical (layout has spread)"""
        G = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=float)
        
        forceatlas = ForceAtlas2(verbose=False)
        positions = forceatlas.forceatlas2(G, iterations=100)
        
        # Check that not all positions are identical
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Standard deviation should be non-zero
        assert np.std(x_coords) > 1e-6 or np.std(y_coords) > 1e-6


class TestForceAtlas2ParameterValidation:
    """Test parameter validation"""
    
    def test_unimplemented_linLogMode_raises_error(self):
        """Test that linLogMode=True raises AssertionError"""
        with pytest.raises(AssertionError):
            ForceAtlas2(linLogMode=True)
    
    def test_unimplemented_adjustSizes_raises_error(self):
        """Test that adjustSizes=True raises AssertionError"""
        with pytest.raises(AssertionError):
            ForceAtlas2(adjustSizes=True)
    
    def test_unimplemented_multiThreaded_raises_error(self):
        """Test that multiThreaded=True raises AssertionError"""
        with pytest.raises(AssertionError):
            ForceAtlas2(multiThreaded=True)
    
    def test_valid_parameters_accepted(self):
        """Test that all valid parameter combinations are accepted"""
        valid_configs = [
            {},
            {'barnesHutOptimize': True, 'barnesHutTheta': 1.5},
            {'strongGravityMode': True, 'gravity': 2.0},
            {'outboundAttractionDistribution': True, 'edgeWeightInfluence': 0.5},
            {'scalingRatio': 5.0, 'jitterTolerance': 2.0},
            {'verbose': False},
        ]
        
        for config in valid_configs:
            fa2 = ForceAtlas2(**config)
            assert fa2 is not None

