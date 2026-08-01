"""Tests for turn-by-turn instruction generation."""

import pytest

from campus_nav.navigation.engine.instructions import generate_instructions
from campus_nav.navigation.routing.dijkstra import dijkstra_route

pytestmark = pytest.mark.unit


def test_empty_route_produces_no_instructions(simple_graph):
    route = dijkstra_route(simple_graph, "A", "A")
    assert generate_instructions(route, simple_graph) == []


def test_single_edge_route_has_head_toward_and_arrival(simple_graph):
    route = dijkstra_route(simple_graph, "A", "B")
    instructions = generate_instructions(route, simple_graph)
    assert len(instructions) == 2
    assert "Head toward Node B" in instructions[0].text
    assert instructions[-1].is_arrival is True
    assert "arrived" in instructions[-1].text.lower()


def test_instruction_count_matches_edges_plus_arrival(simple_graph):
    route = dijkstra_route(simple_graph, "A", "E")
    instructions = generate_instructions(route, simple_graph)
    assert len(instructions) == len(route.edge_ids) + 1


def test_arrival_instruction_targets_destination_node(simple_graph):
    route = dijkstra_route(simple_graph, "A", "E")
    instructions = generate_instructions(route, simple_graph)
    assert instructions[-1].target_node_id == "E"
    assert instructions[-1].distance_meters == 0.0


def test_stairs_path_type_produces_stairs_warning(pilot_graph):
    route = dijkstra_route(pilot_graph, "main_gate", "hostel_h1")
    instructions = generate_instructions(route, pilot_graph)
    assert any("Stairs ahead" in instr.text for instr in instructions)


def test_ramp_path_type_produces_ramp_notice(pilot_graph):
    from campus_nav.navigation.routing.types import AccessibilityMode

    route = dijkstra_route(pilot_graph, "main_gate", "hostel_h1", AccessibilityMode.STRICT)
    instructions = generate_instructions(route, pilot_graph)
    assert any("Ramp ahead" in instr.text for instr in instructions)


def test_each_non_arrival_instruction_mentions_distance(simple_graph):
    route = dijkstra_route(simple_graph, "A", "E")
    instructions = generate_instructions(route, simple_graph)
    for instr in instructions[:-1]:
        assert "meters" in instr.text


def test_all_instructions_are_non_empty_strings(pilot_graph):
    route = dijkstra_route(pilot_graph, "main_gate", "biomedical")
    for instr in generate_instructions(route, pilot_graph):
        assert isinstance(instr.text, str)
        assert len(instr.text) > 0
