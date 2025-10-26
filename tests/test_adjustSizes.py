"""
Test suite for adjustSizes (overlap prevention) feature.

These tests verify that the anti-collision implementation works correctly
when adjustSizes=True is enabled.
"""

import numpy as np
import pytest

from fa2_modified import ForceAtlas2, fa2util


class TestAntiCollisionFunctions:
    """Test the anti-collision helper functions"""

    def test_linRepulsion_antiCollision_basic(self):
        """Test basic anti-collision repulsion between two nodes (not overlapping)"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0
        n1.dx = 0.0
        n1.dy = 0.0

        n2 = fa2util.Node()
        n2.x = 3.0  # distance=3.0, effective distance = 3.0 - 1.0 - 1.0 = 1.0 (not overlapping)
        n2.y = 0.0
        n2.mass = 1.0
        n2.size = 1.0
        n2.dx = 0.0
        n2.dy = 0.0

        # Apply anti-collision repulsion
        fa2util.linRepulsion_antiCollision(n1, n2, coefficient=1.0)

        # Nodes should be pushed apart (normal repulsion, not strong)
        assert n1.dx < 0  # n1 pushed left
        assert n2.dx > 0  # n2 pushed right
        assert abs(n1.dx) == abs(n2.dx)  # Equal and opposite forces
        # Should use normal repulsion (not 100x factor)
        assert abs(n1.dx) < 10  # Not using strong repulsion

    def test_linRepulsion_antiCollision_overlapping(self):
        """Test anti-collision when nodes are overlapping"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0
        n1.dx = 0.0
        n1.dy = 0.0

        n2 = fa2util.Node()
        n2.x = 0.5  # Overlapping (distance < n1.size + n2.size)
        n2.y = 0.0
        n2.mass = 1.0
        n2.size = 1.0
        n2.dx = 0.0
        n2.dy = 0.0

        # Apply anti-collision repulsion
        fa2util.linRepulsion_antiCollision(n1, n2, coefficient=1.0)

        # Strong repulsion should be applied
        assert n1.dx < 0  # n1 pushed left
        assert n2.dx > 0  # n2 pushed right
        # Force should be stronger than non-overlapping case
        assert abs(n1.dx) > 10  # Strong repulsion factor of 100

    def test_linAttraction_antiCollision_basic(self):
        """Test anti-collision attraction between connected nodes"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0
        n1.dx = 0.0
        n1.dy = 0.0

        n2 = fa2util.Node()
        n2.x = 5.0
        n2.y = 0.0
        n2.mass = 1.0
        n2.size = 1.0
        n2.dx = 0.0
        n2.dy = 0.0

        # Apply anti-collision attraction
        fa2util.linAttraction_antiCollision(
            n1, n2, e=1.0, distributedAttraction=False, coefficient=1.0
        )

        # Nodes should be pulled together
        assert n1.dx > 0  # n1 pulled right
        assert n2.dx < 0  # n2 pulled left

    def test_linAttraction_antiCollision_overlapping_no_force(self):
        """Test that overlapping nodes don't attract (to prevent collapse)"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0
        n1.dx = 0.0
        n1.dy = 0.0

        n2 = fa2util.Node()
        n2.x = 0.5  # Overlapping
        n2.y = 0.0
        n2.mass = 1.0
        n2.size = 1.0
        n2.dx = 0.0
        n2.dy = 0.0

        # Apply anti-collision attraction
        fa2util.linAttraction_antiCollision(
            n1, n2, e=1.0, distributedAttraction=False, coefficient=1.0
        )

        # No attraction force should be applied when overlapping
        assert n1.dx == 0.0
        assert n1.dy == 0.0
        assert n2.dx == 0.0
        assert n2.dy == 0.0


class TestAdjustSizesWithRepulsion:
    """Test repulsion with adjustSizes enabled"""

    def test_apply_repulsion_with_adjustSizes(self):
        """Test that apply_repulsion uses anti-collision when adjustSizes=True"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0

        n2 = fa2util.Node()
        n2.x = 1.5
        n2.y = 0.0
        n2.mass = 1.0
        n2.size = 1.0

        nodes = [n1, n2]

        # Apply repulsion with adjustSizes
        fa2util.apply_repulsion(nodes, coefficient=1.0, adjustSizes=True)

        # Verify forces were applied
        assert n1.dx != 0.0 or n1.dy != 0.0
        assert n2.dx != 0.0 or n2.dy != 0.0

    def test_apply_repulsion_without_adjustSizes(self):
        """Test that apply_repulsion uses normal repulsion when adjustSizes=False"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0

        n2 = fa2util.Node()
        n2.x = 1.5
        n2.y = 0.0
        n2.mass = 1.0
        n2.size = 1.0

        nodes = [n1, n2]

        # Apply repulsion without adjustSizes (default behavior)
        fa2util.apply_repulsion(nodes, coefficient=1.0, adjustSizes=False)

        # Verify forces were applied
        assert n1.dx != 0.0 or n1.dy != 0.0
        assert n2.dx != 0.0 or n2.dy != 0.0

    def test_adjustSizes_produces_different_forces(self):
        """Test that adjustSizes=True produces different forces than adjustSizes=False for overlapping nodes"""
        # Test with overlapping nodes
        n1_with = fa2util.Node()
        n1_with.x = 0.0
        n1_with.y = 0.0
        n1_with.mass = 1.0
        n1_with.size = 1.0

        n2_with = fa2util.Node()
        n2_with.x = 1.5  # Overlapping: distance 1.5 - size 2.0 = -0.5
        n2_with.y = 0.0
        n2_with.mass = 1.0
        n2_with.size = 1.0

        # Create identical setup for without adjustSizes
        n1_without = fa2util.Node()
        n1_without.x = 0.0
        n1_without.y = 0.0
        n1_without.mass = 1.0
        n1_without.size = 1.0

        n2_without = fa2util.Node()
        n2_without.x = 1.5
        n2_without.y = 0.0
        n2_without.mass = 1.0
        n2_without.size = 1.0

        # Apply with adjustSizes
        fa2util.apply_repulsion([n1_with, n2_with], coefficient=1.0, adjustSizes=True)

        # Apply without adjustSizes
        fa2util.apply_repulsion(
            [n1_without, n2_without], coefficient=1.0, adjustSizes=False
        )

        # For overlapping nodes, adjustSizes should produce stronger repulsion
        # The anti-collision uses 100x factor for overlapping
        assert abs(n1_with.dx) > abs(
            n1_without.dx
        ), "adjustSizes should produce stronger forces for overlapping nodes"


class TestAdjustSizesWithAttraction:
    """Test attraction with adjustSizes enabled"""

    def test_apply_attraction_with_adjustSizes(self):
        """Test that apply_attraction uses anti-collision when adjustSizes=True"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0
        n1.dx = 0.0
        n1.dy = 0.0

        n2 = fa2util.Node()
        n2.x = 5.0
        n2.y = 0.0
        n2.mass = 1.0
        n2.size = 1.0
        n2.dx = 0.0
        n2.dy = 0.0

        nodes = [n1, n2]

        edge = fa2util.Edge()
        edge.node1 = 0
        edge.node2 = 1
        edge.weight = 1.0

        edges = [edge]

        # Apply attraction with adjustSizes
        fa2util.apply_attraction(
            nodes,
            edges,
            distributedAttraction=False,
            coefficient=1.0,
            edgeWeightInfluence=1.0,
            adjustSizes=True,
        )

        # Verify attraction forces were applied
        assert n1.dx > 0  # n1 pulled toward n2
        assert n2.dx < 0  # n2 pulled toward n1


class TestRegionWithAdjustSizes:
    """Test Barnes-Hut region calculations with adjustSizes"""

    def test_region_applyForce_with_adjustSizes(self):
        """Test that Region.applyForce works with adjustSizes parameter"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0

        n2 = fa2util.Node()
        n2.x = 10.0
        n2.y = 10.0
        n2.mass = 1.0
        n2.size = 1.0

        nodes = [n2]
        region = fa2util.Region(nodes)

        # Apply force with adjustSizes
        region.applyForce(n1, theta=1.2, coefficient=1.0, adjustSizes=True)

        # Verify force was applied
        assert n1.dx != 0.0 or n1.dy != 0.0

    def test_region_applyForceOnNodes_with_adjustSizes(self):
        """Test that Region.applyForceOnNodes works with adjustSizes"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0

        n2 = fa2util.Node()
        n2.x = 10.0
        n2.y = 10.0
        n2.mass = 1.0
        n2.size = 1.0

        n3 = fa2util.Node()
        n3.x = 10.0
        n3.y = -10.0
        n3.mass = 1.0
        n3.size = 1.0

        region_nodes = [n2, n3]
        region = fa2util.Region(region_nodes)
        region.buildSubRegions()

        target_nodes = [n1]

        # Apply force on nodes with adjustSizes
        region.applyForceOnNodes(
            target_nodes, theta=1.2, coefficient=1.0, adjustSizes=True
        )

        # Verify force was applied to target node
        assert n1.dx != 0.0 or n1.dy != 0.0


class TestAdjustSpeedWithAdjustSizes:
    """Test speed adjustment and force application with adjustSizes"""

    def test_adjustSpeedAndApplyForces_with_adjustSizes(self):
        """Test that adjustSpeedAndApplyForces applies different logic with adjustSizes=True"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0
        n1.old_dx = 0.0
        n1.old_dy = 0.0
        n1.dx = 1.0
        n1.dy = 1.0

        nodes = [n1]

        result = fa2util.adjustSpeedAndApplyForces(
            nodes, speed=1.0, speedEfficiency=1.0, jitterTolerance=1.0, adjustSizes=True
        )

        # Verify position changed
        assert n1.x != 0.0 or n1.y != 0.0
        # Verify result dictionary
        assert "speed" in result
        assert "speedEfficiency" in result

    def test_adjustSpeedAndApplyForces_without_adjustSizes(self):
        """Test normal behavior when adjustSizes=False"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        n1.size = 1.0
        n1.old_dx = 0.0
        n1.old_dy = 0.0
        n1.dx = 1.0
        n1.dy = 1.0

        nodes = [n1]

        result = fa2util.adjustSpeedAndApplyForces(
            nodes,
            speed=1.0,
            speedEfficiency=1.0,
            jitterTolerance=1.0,
            adjustSizes=False,
        )

        # Verify position changed
        assert n1.x != 0.0 or n1.y != 0.0
        # Verify result dictionary
        assert "speed" in result
        assert "speedEfficiency" in result

    def test_adjustSizes_limits_movement(self):
        """Test that adjustSizes=True applies movement limiting (factor clamping)"""
        # Create two identical nodes with large forces
        n1_with = fa2util.Node()
        n1_with.x = 0.0
        n1_with.y = 0.0
        n1_with.mass = 1.0
        n1_with.size = 1.0
        n1_with.old_dx = 0.0
        n1_with.old_dy = 0.0
        n1_with.dx = 100.0  # Large force
        n1_with.dy = 0.0

        n1_without = fa2util.Node()
        n1_without.x = 0.0
        n1_without.y = 0.0
        n1_without.mass = 1.0
        n1_without.size = 1.0
        n1_without.old_dx = 0.0
        n1_without.old_dy = 0.0
        n1_without.dx = 100.0  # Large force
        n1_without.dy = 0.0

        # Apply with adjustSizes (should limit movement)
        fa2util.adjustSpeedAndApplyForces(
            [n1_with],
            speed=10.0,
            speedEfficiency=1.0,
            jitterTolerance=1.0,
            adjustSizes=True,
        )

        # Apply without adjustSizes (no limiting)
        fa2util.adjustSpeedAndApplyForces(
            [n1_without],
            speed=10.0,
            speedEfficiency=1.0,
            jitterTolerance=1.0,
            adjustSizes=False,
        )

        # With adjustSizes, movement should be more limited (factor capped at 10.0/df)
        # Without adjustSizes, movement can be larger
        # The factor with adjustSizes includes: min(factor * df, 10.0) / df
        assert (
            n1_with.x <= n1_without.x
        ), "adjustSizes should limit movement for large forces"


class TestForceAtlas2WithAdjustSizes:
    """Integration tests for ForceAtlas2 with adjustSizes enabled"""

    def test_forceatlas2_with_adjustSizes_simple_graph(self):
        """Test ForceAtlas2 layout with adjustSizes on a simple graph"""
        # Create a simple 3-node triangle graph
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)

        fa2 = ForceAtlas2(adjustSizes=True, nodeSize=2.0, verbose=False)

        # Run layout
        positions = fa2.forceatlas2(G, iterations=10)

        # Verify we got positions for all nodes
        assert len(positions) == 3
        for pos in positions:
            assert len(pos) == 2  # x, y coordinates

    def test_forceatlas2_adjustSizes_prevents_overlap(self):
        """Test that adjustSizes helps prevent node overlap"""
        # Create a dense graph where nodes might overlap
        G = np.ones((5, 5)) - np.eye(5)  # Fully connected graph

        fa2 = ForceAtlas2(
            adjustSizes=True,
            nodeSize=1.5,
            barnesHutOptimize=False,  # Use direct n-body for this test
            verbose=False,
        )

        # Run layout
        positions = fa2.forceatlas2(G, iterations=50)

        # Calculate minimum distance between any two nodes
        min_distance = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt(
                    (positions[i][0] - positions[j][0]) ** 2
                    + (positions[i][1] - positions[j][1]) ** 2
                )
                min_distance = min(min_distance, dist)

        # With adjustSizes and nodeSize=1.5, nodes should maintain some distance
        # The minimum distance should be greater than zero
        assert min_distance > 0, "Nodes should not overlap completely"

    def test_node_size_initialization(self):
        """Test that node size is properly initialized from ForceAtlas2 parameter"""
        G = np.array([[0, 1], [1, 0]], dtype=float)

        fa2 = ForceAtlas2(adjustSizes=True, nodeSize=3.5, verbose=False)
        nodes, edges = fa2.init(G)

        # All nodes should have the specified size
        for node in nodes:
            assert node.size == 3.5

    def test_adjustSizes_with_barnes_hut(self):
        """Test that adjustSizes works with Barnes-Hut optimization"""
        G = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=float
        )

        fa2 = ForceAtlas2(
            adjustSizes=True,
            nodeSize=2.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            verbose=False,
        )

        # Should run without errors
        positions = fa2.forceatlas2(G, iterations=20)

        assert len(positions) == 4
        for pos in positions:
            assert len(pos) == 2


class TestBackwardCompatibility:
    """Test that existing code without adjustSizes still works"""

    def test_default_adjustSizes_is_false(self):
        """Test that adjustSizes defaults to False for backward compatibility"""
        fa2 = ForceAtlas2()
        assert fa2.adjustSizes is False

    def test_existing_code_works_without_adjustSizes_parameter(self):
        """Test that code written before adjustSizes was added still works"""
        G = np.array([[0, 1], [1, 0]], dtype=float)

        fa2 = ForceAtlas2(verbose=False)
        positions = fa2.forceatlas2(G, iterations=5)

        assert len(positions) == 2
