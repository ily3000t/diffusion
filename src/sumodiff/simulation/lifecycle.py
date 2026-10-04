from __future__ import annotations

from collections import Counter

SIMULATION_EVENTS = {
    "loaded": "getLoadedIDList", "entered": "getDepartedIDList", "arrived": "getArrivedIDList",
    "teleport_started": "getStartingTeleportIDList", "teleport_ended": "getEndingTeleportIDList",
    "stop_started": "getStopStartingVehiclesIDList", "stop_ended": "getStopEndingVehiclesIDList",
    "parking_started": "getParkingStartingVehiclesIDList", "parking_ended": "getParkingEndingVehiclesIDList",
    "emergency_stop": "getEmergencyStoppingVehiclesIDList",
}


class Lifecycle:
    def __init__(self):
        self.previous = set()
        self.teleporting = set()
        self.entered = set()
        self.arrived = set()
        self.counts = Counter()

    def observe(self, tick: int, time_seconds: float, active: set[str], observed: dict[str, list[str]]) -> list[dict]:
        events = []
        def emit(kind, vehicle_id, source, **details):
            self.counts[kind] += 1
            events.append({"tick": tick, "time_seconds": time_seconds, "kind": kind,
                           "vehicle_id": vehicle_id, "source": source, **details})
        for kind, ids in observed.items():
            for vehicle_id in sorted(ids):
                emit(kind, vehicle_id, "traci.simulation." + SIMULATION_EVENTS[kind])
        departed = set(observed.get("entered", []))
        arrived = set(observed.get("arrived", []))
        starting = set(observed.get("teleport_started", []))
        ending = set(observed.get("teleport_ended", []))
        self.entered.update(departed)
        self.arrived.update(arrived)
        self.teleporting.update(starting)
        for vehicle_id in sorted(self.previous - active):
            if vehicle_id in arrived:
                emit("exited", vehicle_id, "active_id_difference+arrived_event", reason="arrived")
            elif vehicle_id in self.teleporting or vehicle_id in starting:
                emit("exited", vehicle_id, "active_id_difference+teleport_event", reason="teleport")
            else:
                emit("unexplained_disappearance", vehicle_id, "active_id_difference", reason="unknown")
        for vehicle_id in sorted(active - self.previous - departed - ending - self.teleporting):
            emit("unexplained_appearance", vehicle_id, "active_id_difference", reason="unknown")
        self.teleporting.difference_update(ending | arrived)
        self.previous = active.copy()
        return events
