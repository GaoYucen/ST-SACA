"""Shared, checkpoint-free fleet and served-order accounting.

A state describes the start of a time slot. Only buses idle at that boundary
may depart. All trips, including new departures, advance by one slot duration.
Demand generation, pricing, routing objectives and learning losses live in the
original method modules and are deliberately not changed here.
"""
from collections import Counter
from dataclasses import dataclass
import math
from numbers import Integral


@dataclass
class DispatchAccounting:
    routes: dict
    orders: dict
    served_counts: list
    cost: float
    distance: float
    bus_profit: dict


def _fleet_snapshot(buses):
    snapshot = {}
    for bus_id, state in buses.items():
        if isinstance(bus_id, bool) or not isinstance(bus_id, Integral):
            raise ValueError("Bus IDs must be integers, not bool/float aliases")
        remaining, capacity = float(state[0]), int(state[1])
        if not math.isfinite(remaining) or remaining < 0:
            raise ValueError("Bus remaining time must be finite and nonnegative")
        if capacity != state[1] or capacity < 0:
            raise ValueError("Bus capacity must be a nonnegative integer")
        snapshot[bus_id] = [remaining, capacity]
    return snapshot


def available_seats(buses):
    """Physical capacity contributes to supply only while its bus is idle."""
    return sum(capacity for remaining, capacity in _fleet_snapshot(buses).values()
               if remaining == 0.0)


def dispatch_and_account(env, proposed_counts):
    """Count only valid, actually assigned passengers; never mutate the fleet."""
    raw_counts = list(proposed_counts)
    proposed = [int(value) for value in raw_counts]
    if (len(proposed) != env.config.num_destinations
            or any(count < 0 or count != value
                   for count, value in zip(proposed, raw_counts))):
        raise ValueError("Proposed orders must be nonnegative per-station integers")
    fleet = _fleet_snapshot(env.buses)
    orders = [station for station, count in enumerate(proposed)
              for _ in range(count)]
    routes, assignments = env.dispatcher.dispatch(orders, fleet) if orders else ({}, {})
    if set(routes) != set(assignments):
        raise ValueError("Dispatch route and passenger bus IDs must agree")
    served = [0] * env.config.num_destinations
    total_cost = total_distance = 0.0
    bus_profit = {}
    safe_routes, safe_assignments = {}, {}
    for bus_id, route in routes.items():
        if (isinstance(bus_id, bool) or not isinstance(bus_id, Integral)
                or bus_id not in env.buses or env.buses[bus_id][0] != 0.0):
            raise ValueError("A route can only be assigned to a start-of-slot idle bus")
        passengers = list(assignments[bus_id])
        route = list(route)
        if not passengers or len(passengers) > env.buses[bus_id][1]:
            raise ValueError("A dispatched bus must carry one to capacity passengers")
        for destination in route + passengers:
            if (isinstance(destination, bool) or not isinstance(destination, Integral)
                    or not 0 <= destination < len(served)):
                raise ValueError("Dispatch destinations must be valid integer station IDs")
        route = [int(destination) for destination in route]
        passengers = [int(destination) for destination in passengers]
        if not set(passengers).issubset(set(route)):
            raise ValueError("Every served passenger destination must appear in its route")
        for station, count in Counter(passengers).items():
            served[int(station)] += count
        distance = float(env.calculate_route_distance(route))
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Route distance must be finite and nonnegative")
        cost = float(env.config.beta_d) * distance
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Route cost must be finite and nonnegative")
        bus_profit[bus_id] = sum(float(env.current_p[int(k)]) for k in passengers) - cost
        total_distance += distance
        total_cost += cost
        safe_routes[bus_id], safe_assignments[bus_id] = route, passengers
    if any(actual > requested for actual, requested in zip(served, proposed)):
        raise ValueError("Dispatch served more orders than were proposed")
    return DispatchAccounting(safe_routes, safe_assignments, served,
                              total_cost, total_distance, bus_profit)


def advance_fleet(buses, bus_routes, route_distance, speed, slot_duration):
    """Preserve bus IDs and atomically advance old and new trips to t+1."""
    snapshot = _fleet_snapshot(buses)
    speed, slot_duration = float(speed), float(slot_duration)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Bus speed must be finite and positive")
    if not math.isfinite(slot_duration) or slot_duration <= 0:
        raise ValueError("Slot duration must be finite and positive")
    for bus_id, route in bus_routes.items():
        if (isinstance(bus_id, bool) or not isinstance(bus_id, Integral)
                or bus_id not in snapshot or snapshot[bus_id][0] != 0.0):
            raise ValueError("Only a start-of-slot idle bus may receive a new route")
        distance = float(route_distance(route))
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Route distance must be finite and nonnegative")
        travel_time = distance / speed
        if not math.isfinite(travel_time):
            raise ValueError("Travel time must be finite")
        snapshot[bus_id][0] = travel_time
    for bus_id, (remaining, capacity) in snapshot.items():
        buses[bus_id] = [max(0.0, remaining - slot_duration), capacity]
