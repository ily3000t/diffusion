"""Network topology, exact lane corridors, and raster representation.

SUMO lane shapes are treated as piecewise-linear centerlines. Corridors use
half-width buffers with flat caps and round joins (16 segments per quarter
circle); exact here means this explicit vector geometry, not a raster oracle.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import LineString, Polygon, GeometryCollection, mapping, box
from shapely.ops import unary_union


def _shape(value):
    rows = [list(map(float, p.split(','))) for p in value.split()]
    if rows and any(len(p) != 2 for p in rows):
        raise NotImplementedError('Only planar SUMO networks are supported')
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def _local_geometry(geometry, frame):
    r = frame.rotation
    offset = -r @ frame.origin
    return affinity.affine_transform(geometry, [r[0, 0], r[0, 1], r[1, 0], r[1, 1], *offset])


def resample_polyline(points, count):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, lengths > 1e-10]
    points = points[keep]
    if len(points) < 2:
        raise ValueError('Degenerate lane centerline')
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(0., distance[-1], count)
    xy = np.column_stack([np.interp(samples, distance, points[:, d]) for d in range(2)])
    tangent = np.gradient(xy, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    return xy, tangent


@dataclass
class RoadMap:
    lanes: list
    connections: list
    junctions: list
    lane_areas: dict
    drivable: object

    @classmethod
    def read(cls, path):
        root = ET.parse(path).getroot()
        if root.find('tlLogic') is not None:
            raise NotImplementedError('Signal-controlled networks are outside v1')
        lanes, edge_ids, lane_ids, areas = [], {}, {}, {}
        for edge in root.findall('edge'):
            edge_ids[edge.get('id')] = []
            for element in edge.findall('lane'):
                allowed = element.get('allow', '').split()
                forbidden = element.get('disallow', '').split()
                if (allowed and 'passenger' not in allowed and 'all' not in allowed) or 'passenger' in forbidden or 'all' in forbidden:
                    continue
                points = _shape(element.get('shape', ''))
                width = float(element.get('width', '3.2'))
                speed = float(element.get('speed', '0'))
                if len(points) < 2 or width <= 0 or speed <= 0 or not np.isfinite(points).all():
                    raise ValueError('Invalid lane geometry')
                record = dict(id=element.get('id'), edge_id=edge.get('id'), index=int(element.get('index')),
                              width_m=width, speed_limit_mps=speed, internal=edge.get('function') == 'internal',
                              priority=int(edge.get('priority', '0')), points_world_m=points.tolist(),
                              sumo_length_m=float(element.get('length', 'nan')),
                              zero_geometry_length=bool(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() < 1e-10),
                              allow=allowed, disallow=forbidden)
                if not np.isfinite(record['sumo_length_m']) or record['sumo_length_m'] <= 0:
                    raise ValueError('Invalid declared lane length')
                if record['zero_geometry_length'] and not record['internal']:
                    raise ValueError('Degenerate external lane')
                lanes.append(record)
                edge_ids[edge.get('id')].append(record['id'])
                lane_ids[(record['edge_id'], record['index'])] = record['id']
                areas[record['id']] = LineString(points).buffer(width / 2, cap_style=2, join_style=1, quad_segs=16)
        if not lanes:
            raise ValueError('No passenger lanes')
        connections = []
        for element in root.findall('connection'):
            attrs = dict(element.attrib)
            source = lane_ids.get((attrs['from'], int(attrs['fromLane'])))
            target = lane_ids.get((attrs['to'], int(attrs['toLane'])))
            if source is None or target is None:
                raise NotImplementedError('Connections to excluded vehicle classes are unsupported')
            via = attrs.get('via')
            if via and via not in areas:
                raise ValueError('Connection references missing internal lane')
            connections.append(dict(source_lane=source, target_lane=target, via_lane=via, sumo_raw=attrs))
        junctions, road_parts = [], list(areas.values())
        for element in root.findall('junction'):
            attrs = dict(element.attrib)
            points = _shape(attrs.get('shape', ''))
            requests = [dict(r.attrib) for r in element.findall('request')]
            incoming = attrs.get('incLanes', '').split()
            normal = [c for lane in incoming for c in connections
                      if c['source_lane'] == lane and not c['sumo_raw']['from'].startswith(':')]
            # netconvert orders each incoming lane's links by turn direction.
            rank = {'r': 0, 'R': 1, 's': 2, 'L': 3, 'l': 4, 't': 5}
            normal.sort(key=lambda c: (incoming.index(c['source_lane']), rank[c['sumo_raw']['dir']],
                                       int(c['sumo_raw']['toLane'])))
            if requests and len(normal) != len(requests):
                raise ValueError('Cannot associate junction requests with legal connections')
            for index, connection in enumerate(normal):
                connection['junction_id'] = attrs['id']
                connection['request_index'] = index
            for request in requests:
                index = int(request['index'])
                if index >= len(normal) or any(len(request[key]) != len(normal) for key in ('response', 'foes')):
                    raise ValueError('Invalid junction request bitset')
                request['yield_to_indices'] = [i for i, bit in enumerate(reversed(request['response'])) if bit == '1']
                request['foe_indices'] = [i for i, bit in enumerate(reversed(request['foes'])) if bit == '1']
            junctions.append(dict(sumo_raw=attrs, requests=requests, points_world_m=points.tolist()))
            if len(points) >= 3:
                polygon = Polygon(points)
                # Dead-end SUMO shapes may be degenerate lines; do not buffer them.
                if polygon.area > 1e-9:
                    if not polygon.is_valid:
                        raise ValueError('Invalid junction polygon')
                    road_parts.append(polygon)
        return cls(lanes, connections, junctions, areas, unary_union(road_parts))

    def route_lane_ids(self, route_edges):
        if not route_edges:
            raise ValueError('Empty planned route')
        edges = {lane['edge_id'] for lane in self.lanes if not lane['internal']}
        if any(edge not in edges for edge in route_edges):
            raise ValueError('Unknown planned route edge')
        permitted = {lane['id'] for lane in self.lanes if lane['edge_id'] in route_edges}
        by_source = {}
        for c in self.connections:
            by_source.setdefault(c['source_lane'], []).append(c)
        for start, end in zip(route_edges[:-1], route_edges[1:]):
            links = [c for c in self.connections if c['sumo_raw']['from'] == start and c['sumo_raw']['to'] == end]
            if not links:
                raise ValueError('Disconnected planned route')
            for link in links:
                cursor, visited = link, set()
                while cursor['via_lane']:
                    lane = cursor['via_lane']
                    if lane in visited:
                        raise ValueError('Internal lane cycle')
                    visited.add(lane)
                    permitted.add(lane)
                    next_links = [c for c in by_source.get(lane, []) if c['target_lane'] == link['target_lane']]
                    if len(next_links) != 1:
                        raise ValueError('Ambiguous or missing internal connection')
                    cursor = next_links[0]
        # Includes every legal passenger lane of route edges, permitting lane changes.
        return permitted

    def route_area(self, route_edges):
        return unary_union([self.lane_areas[lane] for lane in self.route_lane_ids(route_edges)])

    def local(self, frame, routes, points_per_lane=64):
        lane_ids = [lane['id'] for lane in self.lanes]
        features = np.zeros((len(lane_ids), points_per_lane, 8), dtype=np.float32)
        original = []
        for i, lane in enumerate(self.lanes):
            points = frame.positions(lane['points_world_m'])
            if lane['zero_geometry_length']:
                targets = [c['target_lane'] for c in self.connections if c['source_lane'] == lane['id']]
                neighbors = [l for l in self.lanes if l['id'] in targets and not l['zero_geometry_length']]
                if not neighbors:
                    raise ValueError('No direction available for zero-length internal lane')
                neighbor = frame.positions(neighbors[0]['points_world_m'])
                direction = neighbor[-1] - neighbor[0]
                direction /= np.linalg.norm(direction)
                xy, tangent = np.repeat(points[:1], points_per_lane, axis=0), np.repeat(direction[None], points_per_lane, axis=0)
            else:
                xy, tangent = resample_polyline(points, points_per_lane)
            features[i, :, :2], features[i, :, 2:4] = xy, tangent
            features[i, :, 4:] = [lane['width_m'], lane['speed_limit_mps'], lane['internal'], lane['priority']]
            original.append({**{k: v for k, v in lane.items() if k != 'points_world_m'}, 'points_local_m': points.tolist()})
        adjacency = np.zeros((len(lane_ids), len(lane_ids)), dtype=bool)
        for c in self.connections:
            adjacency[lane_ids.index(c['source_lane']), lane_ids.index(c['via_lane'] or c['target_lane'])] = True
        route_mask = np.zeros((len(routes), len(lane_ids)), dtype=bool)
        corridors = []
        for i, route in enumerate(routes):
            permitted = self.route_lane_ids(route)
            route_mask[i] = [lane in permitted for lane in lane_ids]
            corridors.append(mapping(_local_geometry(self.route_area(route), frame)))
        exact = dict(schema_version='sumodiff.local.map.v1', lanes=original, connections=self.connections,
                     junctions=[{**{k: v for k, v in j.items() if k != 'points_world_m'},
                                 'points_local_m': frame.positions(np.asarray(j['points_world_m']).reshape(-1, 2)).tolist()} for j in self.junctions],
                     drivable=mapping(_local_geometry(self.drivable, frame)), route_corridors=corridors,
                     geometry_convention='lane half-width flat-cap round-join buffer, quad_segs=16; junction shape union')
        # Remove world coordinates from raw junction attributes, keeping only local position.
        for j in exact['junctions']:
            raw = dict(j['sumo_raw'])
            position = frame.positions([float(raw.pop('x')), float(raw.pop('y'))]).tolist()
            raw.pop('shape', None)
            j['sumo_raw'] = raw
            j['position_local_m'] = position
        return features, adjacency, route_mask, exact


def rasterize(exact, extent, size=256):
    """Binary channels at fixed pixel centers: drivable, boundary, centerline.

    Rows go from +y to -y; columns go from -x to +x. Vector polygons remain
    available separately. Raster line width is one pixel, never a road test.
    """
    from shapely.geometry import shape
    xmin, ymin, xmax, ymax = extent
    if size < 2 or xmax <= xmin or ymax <= ymin:
        raise ValueError('Invalid raster bounds')
    def pixels(coords):
        return [((x - xmin) * size / (xmax - xmin) - .5,
                 (ymax - y) * size / (ymax - ymin) - .5) for x, y in coords]
    road = shape(exact['drivable'])
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    polygons = [road] if road.geom_type == 'Polygon' else list(road.geoms)
    for polygon in polygons:
        draw.polygon(pixels(polygon.exterior.coords), fill=1)
        for hole in polygon.interiors:
            draw.polygon(pixels(hole.coords), fill=0)
    boundary = Image.new('L', (size, size), 0)
    lines = ImageDraw.Draw(boundary)
    for polygon in polygons:
        lines.line(pixels(polygon.exterior.coords), fill=1, width=1)
        for hole in polygon.interiors:
            lines.line(pixels(hole.coords), fill=1, width=1)
    centerline = Image.new('L', (size, size), 0)
    centers = ImageDraw.Draw(centerline)
    for lane in exact['lanes']:
        centers.line(pixels(lane['points_local_m']), fill=1, width=1)
    return np.stack([np.asarray(i, dtype=np.uint8) for i in (image, boundary, centerline)])
