"""
Precise numerical tests for fa2util module with exact value checking
Tests inspired by external ForceAtlas2 implementations to ensure numerical accuracy
"""

import math

import pytest

from fa2_modified import fa2util


class TestLinRepulsionPrecise:
    """Precise numerical tests for linear repulsion"""

    def test_lin_repulsion_exact_values(self):
        """Test linear repulsion with exact numerical expectations"""
        # Two nodes at (1,1) and (2,2) with mass 2 each
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = 2.0
        b.mass = 2.0
        a.x, a.y = 1.0, 1.0
        b.x, b.y = 2.0, 2.0

        # Reset forces
        a.dx = a.dy = b.dx = b.dy = 0.0

        fa2util.linRepulsion(a, b, coefficient=1.0)

        # dx = 1-2 = -1, dy = -1, dist_sq = 2
        # factor = (coefficient * mass_a * mass_b) / dist_sq = (1 * 2 * 2) / 2 = 2
        # a.dx += -1 * 2 = -2, a.dy += -1 * 2 = -2
        # b.dx -= -1 * 2 = 2, b.dy -= -1 * 2 = 2
        assert a.dx == pytest.approx(-2.0)
        assert a.dy == pytest.approx(-2.0)
        assert b.dx == pytest.approx(2.0)
        assert b.dy == pytest.approx(2.0)

    def test_lin_repulsion_asymmetric_masses(self):
        """Test repulsion with different masses"""
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = 3.0
        b.mass = 2.0
        a.x, a.y = 0.0, 0.0
        b.x, b.y = 1.0, 0.0
        a.dx = a.dy = b.dx = b.dy = 0.0

        fa2util.linRepulsion(a, b, coefficient=1.0)

        # dx = 0-1 = -1, dist_sq = 1
        # factor = (1 * 3 * 2) / 1 = 6
        # a.dx += -1 * 6 = -6
        # b.dx -= -1 * 6 = 6
        assert a.dx == pytest.approx(-6.0)
        assert a.dy == pytest.approx(0.0)
        assert b.dx == pytest.approx(6.0)
        assert b.dy == pytest.approx(0.0)


class TestLinRepulsionRegion:
    """Test linear repulsion between node and region"""

    def test_lin_repulsion_region_basic(self):
        """Test linear repulsion between a node and a region"""
        node = fa2util.Node()
        node.mass = 3.0
        node.x = 5.0
        node.y = 5.0
        node.dx = node.dy = 0.0

        # Create a region with TWO nodes (single node regions don't calculate mass)
        dummy1 = fa2util.Node()
        dummy1.mass = 1.0
        dummy1.x = 3.0
        dummy1.y = 3.0

        dummy2 = fa2util.Node()
        dummy2.mass = 1.0
        dummy2.x = 3.0
        dummy2.y = 3.0

        region = fa2util.Region([dummy1, dummy2])

        # The region should have mass from both nodes
        assert region.mass == pytest.approx(2.0)

        fa2util.linRepulsion_region(node, region, coefficient=1.0)

        # dx = 5-3 = 2, dy = 2, dist_sq = 8
        # factor = (1 * 3 * 2) / 8 = 0.75
        # node.dx += 2 * 0.75 = 1.5
        # node.dy += 2 * 0.75 = 1.5
        assert node.dx == pytest.approx(1.5)
        assert node.dy == pytest.approx(1.5)

    def test_lin_repulsion_region_no_update_zero_distance(self):
        """Test no update when node is at region center"""
        node = fa2util.Node()
        node.mass = 1.0
        node.x = 3.0
        node.y = 3.0
        node.dx = node.dy = 0.0

        dummy = fa2util.Node()
        dummy.mass = 1.0
        dummy.x = 3.0
        dummy.y = 3.0
        region = fa2util.Region([dummy])

        fa2util.linRepulsion_region(node, region, coefficient=1.0)

        # No force when at same position
        assert node.dx == pytest.approx(0.0)
        assert node.dy == pytest.approx(0.0)


class TestLinGravityPrecise:
    """Precise numerical tests for linear gravity"""

    def test_lin_gravity_exact_values(self):
        """Test linear gravity with exact numerical expectations"""
        node = fa2util.Node()
        node.mass = 2.0
        node.x = 3.0
        node.y = 4.0  # distance from origin = 5
        node.dx = node.dy = 0.0

        fa2util.linGravity(node, g=1.0)

        # distance = sqrt(9 + 16) = 5
        # factor = (mass * g) / distance = (2 * 1) / 5 = 0.4
        # dx -= 3 * 0.4 = -1.2
        # dy -= 4 * 0.4 = -1.6
        assert node.dx == pytest.approx(-1.2)
        assert node.dy == pytest.approx(-1.6)

    def test_lin_gravity_different_gravity_values(self):
        """Test linear gravity with different gravity constants"""
        node = fa2util.Node()
        node.mass = 1.0
        node.x = 3.0
        node.y = 4.0  # distance = 5
        node.dx = node.dy = 0.0

        fa2util.linGravity(node, g=2.0)

        # factor = (1 * 2) / 5 = 0.4
        # dx -= 3 * 0.4 = -1.2
        # dy -= 4 * 0.4 = -1.6
        assert node.dx == pytest.approx(-1.2)
        assert node.dy == pytest.approx(-1.6)


class TestStrongGravityPrecise:
    """Precise numerical tests for strong gravity"""

    def test_strong_gravity_exact_values(self):
        """Test strong gravity with exact numerical expectations"""
        node = fa2util.Node()
        node.mass = 2.0
        node.x = 3.0
        node.y = 4.0
        node.dx = node.dy = 0.0

        fa2util.strongGravity(node, g=1.0, coefficient=1.0)

        # factor = coefficient * mass * g = 1 * 2 * 1 = 2
        # dx -= 3 * 2 = -6
        # dy -= 4 * 2 = -8
        assert node.dx == pytest.approx(-6.0)
        assert node.dy == pytest.approx(-8.0)

    def test_strong_gravity_with_scaling(self):
        """Test strong gravity with different coefficient"""
        node = fa2util.Node()
        node.mass = 1.0
        node.x = 2.0
        node.y = 3.0
        node.dx = node.dy = 0.0

        fa2util.strongGravity(node, g=1.0, coefficient=2.0)

        # factor = 2 * 1 * 1 = 2
        # dx -= 2 * 2 = -4
        # dy -= 3 * 2 = -6
        assert node.dx == pytest.approx(-4.0)
        assert node.dy == pytest.approx(-6.0)


class TestLinAttractionPrecise:
    """Precise numerical tests for linear attraction"""

    def test_lin_attraction_exact_values(self):
        """Test linear attraction with exact numerical expectations"""
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = 2.0
        b.mass = 3.0
        a.x, a.y = 0.0, 0.0
        b.x, b.y = 3.0, 4.0  # distance = 5
        a.dx = a.dy = b.dx = b.dy = 0.0

        edge_weight = 1.0

        fa2util.linAttraction(
            a, b, edge_weight, distributedAttraction=False, coefficient=2.0
        )

        # dx = 0-3 = -3, dy = 0-4 = -4
        # factor = -coefficient * edge_weight = -2.0 * 1.0 = -2.0
        # a.dx += -3 * -2.0 = 6.0
        # a.dy += -4 * -2.0 = 8.0
        # b.dx -= -3 * -2.0 = -6.0
        # b.dy -= -4 * -2.0 = -8.0
        assert a.dx == pytest.approx(6.0)
        assert a.dy == pytest.approx(8.0)
        assert b.dx == pytest.approx(-6.0)
        assert b.dy == pytest.approx(-8.0)

    def test_lin_attraction_distributed(self):
        """Test distributed attraction (hub dissuasion)"""
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = 4.0  # Hub with higher mass
        b.mass = 1.0
        a.x, a.y = 0.0, 0.0
        b.x, b.y = 2.0, 0.0
        a.dx = a.dy = b.dx = b.dy = 0.0

        edge_weight = 1.0

        fa2util.linAttraction(
            a, b, edge_weight, distributedAttraction=True, coefficient=1.0
        )

        # dx = 0-2 = -2, dy = 0
        # factor = -coefficient * edge_weight / mass_a = -1.0 * 1.0 / 4.0 = -0.25
        # a.dx += -2 * -0.25 = 0.5
        # b.dx -= -2 * -0.25 = -0.5
        assert a.dx == pytest.approx(0.5)
        assert a.dy == pytest.approx(0.0)
        assert b.dx == pytest.approx(-0.5)
        assert b.dy == pytest.approx(0.0)


class TestApplyForcesPrecise:
    """Precise tests for force application functions"""

    def test_apply_repulsion_exact(self):
        """Test apply_repulsion with exact values"""
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = b.mass = 2.0
        a.x, a.y = 0.0, 0.0
        b.x, b.y = 1.0, 0.0
        a.dx = a.dy = b.dx = b.dy = 0.0

        fa2util.apply_repulsion([a, b], coefficient=2.0)

        # dx = 0-1 = -1, dist_sq = 1
        # factor = (2 * 2 * 2) / 1 = 8
        # a.dx += -1 * 8 = -8
        # b.dx -= -1 * 8 = 8
        assert a.dx == pytest.approx(-8.0)
        assert a.dy == pytest.approx(0.0)
        assert b.dx == pytest.approx(8.0)
        assert b.dy == pytest.approx(0.0)

    def test_apply_gravity_linear_exact(self):
        """Test apply_gravity with linear mode"""
        node = fa2util.Node()
        node.mass = 1.0
        node.x, node.y = 3.0, 4.0  # distance = 5
        node.dx = node.dy = 0.0

        fa2util.apply_gravity(
            [node], gravity=1.0, scalingRatio=1.0, useStrongGravity=False
        )

        # factor = (mass * gravity) / distance = (1 * 1) / 5 = 0.2
        # dx -= 3 * 0.2 = -0.6
        # dy -= 4 * 0.2 = -0.8
        assert node.dx == pytest.approx(-0.6)
        assert node.dy == pytest.approx(-0.8)

    def test_apply_gravity_strong_exact(self):
        """Test apply_gravity with strong gravity mode"""
        node = fa2util.Node()
        node.mass = 1.0
        node.x, node.y = 3.0, 4.0
        node.dx = node.dy = 0.0

        fa2util.apply_gravity(
            [node], gravity=1.0, scalingRatio=1.0, useStrongGravity=True
        )

        # factor = scaling_ratio * mass * gravity = 1 * 1 * 1 = 1
        # dx -= 3 * 1 = -3
        # dy -= 4 * 1 = -4
        assert node.dx == pytest.approx(-3.0)
        assert node.dy == pytest.approx(-4.0)

    def test_apply_attraction_exact(self):
        """Test apply_attraction with exact values"""
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = b.mass = 1.0
        a.x, a.y = 0.0, 0.0
        b.x, b.y = 1.0, 0.0
        a.dx = a.dy = b.dx = b.dy = 0.0

        edge = fa2util.Edge()
        edge.node1 = 0
        edge.node2 = 1
        edge.weight = 1.0
        nodes = [a, b]

        fa2util.apply_attraction(
            nodes,
            [edge],
            distributedAttraction=False,
            coefficient=1.0,
            edgeWeightInfluence=1.0,
        )

        # dx = 0-1 = -1, factor = -1.0 * 1.0 = -1.0
        # a.dx += -1 * -1.0 = 1.0
        # b.dx -= -1 * -1.0 = -1.0
        assert a.dx == pytest.approx(1.0)
        assert a.dy == pytest.approx(0.0)
        assert b.dx == pytest.approx(-1.0)
        assert b.dy == pytest.approx(0.0)


class TestRegionPrecise:
    """Precise tests for Region class methods"""

    def test_region_mass_center_calculation(self):
        """Test precise calculation of region mass center"""
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = 2.0
        b.mass = 4.0
        a.x, a.y = 0.0, 0.0
        b.x, b.y = 4.0, 0.0

        region = fa2util.Region([a, b])

        # Total mass = 6
        assert region.mass == pytest.approx(6.0)

        # Center of mass: ((0*2 + 4*4)/6, (0*2 + 0*4)/6) = (16/6, 0) = (8/3, 0)
        assert region.massCenterX == pytest.approx(8.0 / 3.0)
        assert region.massCenterY == pytest.approx(0.0)

    def test_region_size_calculation(self):
        """Test region size calculation"""
        # Create nodes at specific positions
        nodes = []
        positions = [(0, 0), (4, 0)]
        for x, y in positions:
            n = fa2util.Node()
            n.mass = 2.0
            n.x, n.y = float(x), float(y)
            nodes.append(n)

        region = fa2util.Region(nodes)

        # massSumX = 0*2 + 4*2 = 8, mass = 4
        # Center is at (8/4, 0) = (2, 0)
        # Distance from (0,0) to (2,0) = 2, from (4,0) to (2,0) = 2
        # Size should be 2 * max_distance = 2 * 2 = 4.0
        expected_size = 4.0
        assert region.size == pytest.approx(expected_size)

    def test_region_build_subregions_quadrants(self):
        """Test that subregions are built in correct quadrants"""
        # Create 4 nodes in different quadrants around (2, 2)
        nodes = []
        positions = [(1, 3), (1, 1), (3, 3), (3, 1)]
        for x, y in positions:
            n = fa2util.Node()
            n.mass = 1.0
            n.x, n.y = float(x), float(y)
            nodes.append(n)

        region = fa2util.Region(nodes)
        region.buildSubRegions()

        # Should create 4 subregions (one for each quadrant)
        assert len(region.subregions) == 4

        # Each subregion should have exactly 1 node
        for subregion in region.subregions:
            assert len(subregion.nodes) == 1

    def test_region_apply_force_single_node(self):
        """Test applying force from a single-node region"""
        target = fa2util.Node()
        target.mass = 1.0
        target.x = 5.0
        target.y = 5.0
        target.dx = target.dy = 0.0

        source = fa2util.Node()
        source.mass = 1.0
        source.x = 1.0
        source.y = 1.0

        region = fa2util.Region([source])

        # For a single-node region, apply_force calls linRepulsion
        region.applyForce(target, theta=100.0, coefficient=1.0)

        # dx = 5-1 = 4, dy = 4, dist_sq = 32
        # factor = (1 * 1 * 1) / 32 = 0.03125
        # target.dx += 4 * 0.03125 = 0.125
        # target.dy += 4 * 0.03125 = 0.125
        assert target.dx == pytest.approx(0.125)
        assert target.dy == pytest.approx(0.125)

    def test_region_apply_force_uses_approximation(self):
        """Test that region uses approximation when theta condition is met"""
        target = fa2util.Node()
        target.mass = 1.0
        target.x = 10.0
        target.y = 10.0
        target.dx = target.dy = 0.0

        # Create a region with multiple nodes close together
        nodes = []
        for i in range(4):
            n = fa2util.Node()
            n.mass = 1.0
            n.x = float(i)
            n.y = 0.0
            nodes.append(n)

        region = fa2util.Region(nodes)
        region.buildSubRegions()

        # With large theta and large distance, should use region approximation
        region.applyForce(target, theta=0.5, coefficient=1.0)

        # Force should have been applied (non-zero)
        assert target.dx != 0.0 or target.dy != 0.0


class TestAdjustSpeedPrecise:
    """Precise tests for speed adjustment"""

    def test_adjust_speed_modifies_positions(self):
        """Test that adjust_speed_and_apply_forces updates node positions"""
        a = fa2util.Node()
        b = fa2util.Node()
        a.mass = b.mass = 1.0
        a.x, a.y = 0.0, 0.0
        b.x, b.y = 1.0, 0.0

        # Set forces
        a.old_dx, a.old_dy = 0.0, 0.0
        a.dx, a.dy = 1.0, 0.0
        b.old_dx, b.old_dy = 0.0, 0.0
        b.dx, b.dy = -1.0, 0.0

        old_a_x = a.x
        old_b_x = b.x

        result = fa2util.adjustSpeedAndApplyForces(
            [a, b], speed=1.0, speedEfficiency=1.0, jitterTolerance=1.0
        )

        # Positions should have changed
        assert a.x != old_a_x
        assert b.x != old_b_x

        # Result should contain updated speed values
        assert "speed" in result
        assert "speedEfficiency" in result
        assert result["speed"] > 0
        assert result["speedEfficiency"] > 0

    def test_adjust_speed_returns_valid_dict(self):
        """Test that adjust_speed returns properly formatted dictionary"""
        nodes = []
        for i in range(3):
            n = fa2util.Node()
            n.mass = 1.0
            n.x = float(i)
            n.y = 0.0
            n.dx = 0.1
            n.dy = 0.0
            n.old_dx = 0.0
            n.old_dy = 0.0
            nodes.append(n)

        result = fa2util.adjustSpeedAndApplyForces(
            nodes, speed=1.0, speedEfficiency=1.0, jitterTolerance=1.0
        )

        assert isinstance(result, dict)
        assert "speed" in result
        assert "speedEfficiency" in result
        assert isinstance(result["speed"], float)
        assert isinstance(result["speedEfficiency"], float)
