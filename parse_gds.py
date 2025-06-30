import numpy as np
import drawsvg as draw
import gdspy
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection, box
from shapely.strtree import STRtree
from IPython.display import HTML, display, SVG
import requests
import io
from pathlib import Path
from collections import Counter, defaultdict
from collections import namedtuple
import base64
from typing import Any
from graphviz import Digraph
import json

# https://github.com/mbalestrini/GDS2glTF/blob/main/gds2gltf.py


layerstack = {    
    (235,4): {'name':'substrate', 'zmin':-2, 'zmax':0, 'color':[ 0.2, 0.2, 0.2, 1.0]},
    (64,20): {'name':'nwell', 'zmin':-0.5, 'zmax':0.01, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    # (65,44): {'name':'tap', 'zmin':0, 'zmax':0.1, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    (65,20): {'name':'diff', 'zmin':-0.12, 'zmax':0.02, 'color':[ 0.9, 0.9, 0.9, 1.0]},    
    (66,20): {'name':'poly', 'zmin':0, 'zmax':0.18, 'color':[ 0.75, 0.35, 0.46, 1.0]},    
    (66,44): {'name':'licon', 'zmin':0, 'zmax':0.936, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    (67,20): {'name':'li1', 'zmin':0.936, 'zmax':1.136, 'color':[ 1.0, 0.81, 0.55, 1.0]},    

    (67,44): {'name':'mcon', 'zmin':1.011, 'zmax':1.376, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    (68,20): {'name':'met1', 'zmin':1.376, 'zmax':1.736, 'color':[ 0.16, 0.38, 0.83, 1.0]},    
    (68,44): {'name':'via', 'zmin':1.736,'zmax':2, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    (69,20): {'name':'met2', 'zmin':2, 'zmax':2.36, 'color':[ 0.65, 0.75, 0.9, 1.0]},    
    (69,44): {'name':'via2', 'zmin':2.36, 'zmax':2.786, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    (70,20): {'name':'met3', 'zmin':2.786, 'zmax':3.631, 'color':[ 0.2, 0.62, 0.86, 1.0]},    
    (70,44): {'name':'via3', 'zmin':3.631, 'zmax':4.0211, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    (71,20): {'name':'met4', 'zmin':4.0211, 'zmax':4.8661, 'color':[ 0.15, 0.11, 0.38, 1.0]},    
    (71,44): {'name':'via4', 'zmin':4.8661, 'zmax':5.371, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    (72,20): {'name':'met5', 'zmin':5.371, 'zmax':6.6311, 'color':[ 0.4, 0.4, 0.4, 1.0]},
    # (83,44): { 'zmin':0, 'zmax':0.1, 'name':'text'},
}
name2layerid = {v['name']:k for k, v in layerstack.items()}


def draw_polys(polys=[], pad=0.3, fill='#88e', svg=None, bbox=None, scale=30):
    if hasattr(polys, 'geometries'):
        polys = list(polys.geometries)
    if svg is None:
        if bbox is None:
            bbox = GeometryCollection(polys).bounds
        x1, y1, x2, y2 = np.ravel(bbox)
        svg = draw.Drawing(x2-x1+pad*2, y2-y1+pad*2, origin=(x1-pad,y1-pad), transform='scale(1 -1)')
        svg.append(draw.Rectangle(x1, y1, x2-x1, y2-y1, fill='#eee'))
        svg.set_pixel_scale(scale)
    for p in polys:
        p = np.array(p.exterior.coords).ravel()
        svg.append(draw.Lines(*p, close=True,
            fill=fill, stroke='#000', stroke_width=0.05, opacity=0.5))
    return svg


class DisjointSets:
    def __init__(self):
        self.nodes = {}

    def add(self, a):
        return self.nodes.setdefault(a, a)

    def get_root(self, a):
        if self.add(a) == a:
            return a
        root = self.get_root(self.nodes[a])
        self.nodes[a] = root
        return root
    
    def merge(self, a, b):
        root_a, root_b = self.get_root(a), self.get_root(b)
        self.nodes[root_b] = root_a

    def label_components(self, known_ids: dict[Any, int]={}):
        node2id = {self.get_root(node): i for node, i in known_ids.items()}
        next_id = max(list(node2id.values()) + [-1]) + 1
        id2nodes = {}
        for node in self.nodes:
            root = self.get_root(node)
            if root not in node2id:
                node2id[root] = next_id
                next_id += 1
            i = node2id[node] = node2id[root]
            id2nodes.setdefault(i, []).append(node)
        return id2nodes, node2id


Part = namedtuple('Part', 'layer idx')
FET = namedtuple('FET', 'type gate a b')
LayerKey = str | tuple[int,int]
Layers = dict[LayerKey, STRtree]

def extract_layers(cell : gdspy.Cell) -> Layers:
    layers = {}
    for lid, polys in cell.get_polygons(by_spec=True, depth=0).items():
        if isinstance(lid, str):
            continue
        lid = (int(lid[0]), int(lid[1]))
        if lid in layerstack:
            lid = layerstack[lid]['name']
        polys = [Polygon(p) for p in polys]
        polys = shapely.union_all(polys)
        polys = polys if isinstance(polys, MultiPolygon) else MultiPolygon([polys])
        layers[lid] = polys
    if 'diff' in layers and 'poly' in layers:
        layers['channel'] = layers['diff'] & layers['poly']
        layers['sd'] = layers['diff'] - layers['poly'] # source/drain
    return {k:STRtree(v.geoms) for k, v in layers.items()}

def connect_layers(layers: Layers) -> DisjointSets:
    parts = DisjointSets()
    def connect(a, via, b):
        if (a not in layers) or (via not in layers) or (b not in layers):
            return
        for via_idx, p in enumerate(layers[via].geometries):
            c = p.centroid
            part_a = layers[a].query(c, 'within')
            part_b = layers[b].query(c, 'within')
            if len(part_a)==0 or len(part_b)==0:
                continue
            a_idx, b_idx = int(part_a[0]), int(part_b[0])
            parts.merge(Part(a, a_idx), Part(b, b_idx))
            parts.merge(Part(a, a_idx), Part(via, via_idx))

    wires = [layerstack[i, 20]['name'] for i in range(66, 72+1)]
    vias  = [layerstack[i, 44]['name'] for i in range(66, 72)]
    if 'sd' in layers:
        for i in range(len(layers['sd'].geometries)):
            parts.add(('sd', i))
    connect('li1', 'licon', 'sd')    # connect source/drain
    for lo, via, hi in zip(wires[:-1], vias, wires[1:]):
        connect(lo, via, hi)
    return parts


def find_pins(layers: Layers, gds_cell: gdspy.Cell) -> dict[str, Part]:
    pin2part = {}
    for lab in gds_cell.get_labels(depth=0):
        if lab.text in ['VNB', 'VPB']:
            continue
        layer = (lab.layer, 20)
        if layer in layerstack:
            layer = layerstack[layer]['name']
        if layer not in layers:
            continue
        node = layers[layer].query(Point(lab.position), 'within')
        if len(node) != 1:
            print('missing pin:', lab.text, layer)
            continue
        pin2part[lab.text] = Part(layer, int(node[0]))
    return pin2part

def extract_fets(part2wire: dict[Part,int], layers: Layers) -> set[FET]:
    fets = set()
    if 'channel' not in layers:
        return fets
    for c in layers['channel'].geometries:
        sd = layers['sd'].query(c, 'touches')
        assert len(sd) == 2, 'Channel must touch 2 diffusion regions'
        center = c.centroid
        gate = layers['poly'].query(center, 'within')
        assert len(gate) == 1, 'Channel must touch one gate'
        nwell = layers['nwell'].query(center, 'within')
        fet_type = ('N', 'P')[len(nwell)]
        gate = part2wire[('poly', gate[0])]
        a = part2wire[('sd', sd[0])]
        b = part2wire[('sd', sd[1])]
        a, b = (b, a) if a>b else (a, b)
        fets.add(FET(fet_type, gate, a, b))
    return fets

def draw_net(fets, pin2wire, with_gates=True):
    dot = Digraph(graph_attr={'rankdir':'LR'}, node_attr={'shape':'square'}, engine='dot') #neato dot
    colors = {'N':'lightgreen', 'P':'coral'}
    power_wires = set([0,1])
    for pin, wire in pin2wire.items():
        if wire not in power_wires:
            dot.node(f'{abs(wire)}', label='~'*(wire<0)+pin, style='filled')
    for i, fet in enumerate(fets):
        dot.node(f'fet{i}', label='', style='filled', shape='circle', 
                fillcolor=colors[fet.type])
        if fet.a not in power_wires:
            dot.edge(f'fet{i}', f'{abs(fet.a)}', arrowhead='none' if fet.a >= 0 else 'odot')
        if fet.b not in power_wires:
            dot.edge(f'fet{i}', f'{abs(fet.b)}', arrowhead='none' if fet.b >= 0 else 'odot')
        if with_gates:
            gates = fet.gate if type(fet.gate) == tuple else (fet.gate,)
            for gate in gates:
                dot.edge(f'{abs(gate)}', f'fet{i}', dir='both', arrowhead='tee',
                        arrowtail='none' if gate >= 0 else 'odot', color='lightgrey')
    return dot


power_pins = {'VGND':0, 'VPWR':1, 'VNB':0, 'VPB':1, 
              'KAPWR':1, 'VPWRIN':1, 'LOWLVPWR':1}


def get_wires(fets):
    wires = {}
    for fet in fets:
        for wire in [fet.gate, fet.a, fet.b]:
            wires.setdefault(wire, []).append(fet)
    return wires

def classify_wires(fets, wires2fets):
    wires = {'N':set(), 'P':set(), 'G':set()}
    for fet in fets:
        wires['G'].add(fet.gate)
        wires[fet.type].update([fet.a, fet.b])
    in_wires = wires['G'] - (wires['N'] | wires['P'])
    out_wires = (wires['N'] & wires['P']) | {0,1}
    state_wires = set()
    visited = {0:2, 1:2}  # power wires
    def dfs(i):
        if i in visited:
            if visited[i]==1 and i in out_wires:
                state_wires.add(i)
            return
        visited[i] = 1
        for fet in wires2fets[i]:
            if fet.gate == i:
                if fet.a in out_wires or fet.b <= 1:
                    dfs(fet.a)
                if fet.b in out_wires or fet.a <= 1:
                    dfs(fet.b)
            elif i not in out_wires:
                dfs(fet.a if fet.a != i else fet.b)
        visited[i] = 2
    for i in in_wires:
        dfs(i)
    return {'in':in_wires, 'state':state_wires,
            'out':out_wires, 'gate':wires['G']}


def build_signals(wire2fets: dict[int, set[FET]],
                  inputs: set[int], to_resolve: set[int]):
    '''
    return: 
        resolved: bool
        signals: [wire_n,2,case_n]
    '''
    input_n = len(inputs)
    wire_n = max(wire2fets.keys())+1 if wire2fets else 2
    case_n = 1<<input_n
    signals = np.zeros([wire_n,2,case_n], np.uint8)
    signals[0,0] = signals[1,1] = 1
    I = np.arange(case_n)
    inputs = {wire : np.array([1-(I>>i)&1, (I>>i)&1], np.uint8)
                  for i, wire in enumerate(inputs)}
    queue = set(inputs)
    while queue:
        wire = queue.pop()
        for fet in wire2fets[wire]:
            gate_level = int(fet.type=='N')
            gate_open = inputs.get(fet.gate, signals[fet.gate])[gate_level]
            level = signals[:,1-gate_level]
            a, b = level[fet.a], level[fet.b]
            c = (a|b)&gate_open
            a1, b1 = a|c, b|c
            if (a!=a1).any():
                queue.add(fet.a)
                a[:] = a1
            if (b!=b1).any():
                queue.add(fet.b)
                b[:] = b1
    for wire, input_bits in inputs.items():
        undef_mask = ~signals[wire].any(0)
        signals[wire][:,undef_mask] = input_bits[:,undef_mask]
    resolved = signals[list(to_resolve)].any(1).all()
    return resolved, signals

class Cell:
    def __init__(self, gds_cell, cell_library):
        self.gds_cell = gds_cell
        self.cell_library = cell_library
        self.name = gds_cell.name
        self.short_name = gds_cell.name.split('__')[-1]
        self.bbox = gds_cell.get_bounding_box()

        # process geometry
        self.layers = extract_layers(gds_cell)
        self.pin2part = find_pins(self.layers, gds_cell)
        self.part_sets = connect_layers(self.layers)
        for part in self.pin2part.values():
            self.part_sets.add(part)
        known_wires = {self.pin2part[name]:i for name, i in power_pins.items() 
                       if name in self.pin2part}
        self.wire2parts, self.part2wire = self.part_sets.label_components(known_wires)
        self.pin2wire = {pin:self.part2wire[part] for pin, part in self.pin2part.items()}

        # analyse circuit
        self.fets = extract_fets(self.part2wire, self.layers)
        self.wires2fets = get_wires(self.fets)
    
        wire_by_type = classify_wires(self.fets, self.wires2fets)
        to_resolve = wire_by_type['out'] & wire_by_type['gate']
        resolved, signals = build_signals(
            self.wires2fets, wire_by_type['in'], to_resolve)
        if resolved and wire_by_type['state']:
            # Full-Adder cell is the only case where my cycle detection heuristics
            # gives a false positive. We can rule out this case by checking if
            # cell is resolved without knowing the state of the detected wire
            wire_by_type['state'] = {}
            print('false stateful:', self.short_name, ' - resolved')   
        if not resolved:
            wire_by_type['in'] |= wire_by_type['state']
            resolved, signals = build_signals(
                self.wires2fets, wire_by_type['in'], to_resolve)
        
        self.wire_by_type = wire_by_type
        self.signals = signals
        self.resolved = resolved
        self.lut_cache = {}

    def get_lut(self, out_wire):
        if out_wire in self.lut_cache:
            return self.lut_cache[out_wire]
        inputs = list(self.wire_by_type['in'])
        lut = self.signals[out_wire][1].reshape((2,)*len(inputs)).T
        active_inputs = []
        for axis in range(lut.ndim):
            if not np.diff(lut, axis=axis).any():
                lut = lut.take([0], axis)
            else:
                active_inputs.append(inputs[axis])
        lut = lut.T.ravel()
        lut_bits = (np.uint64(lut)<<np.arange(len(lut), dtype=np.uint64)).sum()
        self.lut_cache[out_wire] = active_inputs, lut, lut_bits
        return active_inputs, lut, lut_bits

    def __repr__(self):
        return f'[{self.short_name}]'

    def get_part(self, part):
        layer, idx = part
        return self.layers[layer].geometries[idx]
    
    def print_lut(self, out_wire=None):
        wire2pin = {v:k for k, v in self.pin2wire.items()}
        if out_wire is None:
            outs = (self.wire_by_type['out']-{0,1}) & set(wire2pin)
            assert outs, "Can't find the output wire"
            out_wire = outs.pop()
        inputs, lut, _ = self.get_lut(out_wire)
        n = len(inputs)
        label = lambda wire:wire2pin.get(wire, str(wire))
        print(" ".join(map(label, inputs)), '|', label(out_wire))
        for i, o in zip((np.c_[:1<<n]>>np.r_[:n])&1, lut):
            print(i, '|', o)

    def connect_child(self, ref):
        ref_box = box(*ref.get_bounding_box().ravel())
        ox, oy = ref.origin
        cell = self.cell_library[ref.ref_cell.name]
        cell2top = {}
        for layer in ['li1', 'met1']:
            top_layer = self.layers[layer]
            for top_part in top_layer.query(ref_box, 'intersects'):
                top_wire = self.part2wire[layer, top_part]
                if top_wire <= 1:
                    continue  # skip power wires
                top_geom = top_layer.geometries[top_part]
                top_geom = shapely.affinity.translate(top_geom, -ox, -oy)
                if ref.x_reflection:
                    top_geom = shapely.affinity.scale(top_geom, 1, -1, origin=(0,0))
                if ref.rotation:
                    top_geom = shapely.affinity.rotate(top_geom, ref.rotation, origin=(0,0))
                cell_parts = cell.layers[layer].query(top_geom, 'intersects')
                if len(cell_parts) == 0:
                    continue
                cell_wires = set(cell.part2wire[layer, cell_part] for cell_part in cell_parts)
                assert len(cell_wires) == 1
                cell2top[cell_wires.pop()] = top_wire
        return cell2top


def analyse_cells(gds):
    stateless, stateful, unresolved = [], [], []
    cells = {}
    for cell_name, gds_cell in gds.cells.items():
        short = cell_name.split('__')[-1]
        print(f'\r{short.ljust(30)}', end='')
        if len(gds_cell.references) > 0:
            print('- skipping non-leaf cell')
            continue
        cells[cell_name] = cell = Cell(gds_cell, cells)
        if not cell.resolved:
            unresolved.append(short)
        elif cell.wire_by_type['state']:
            stateful.append(short)
        else:
            stateless.append(short)
    print('\r', end='')
    def format_stat(names):
        n = len(names)
        names = Counter([s.rsplit('_', 1)[0] for s in names])
        names = [f'{s}*{n}' if n>1 else s for s, n in names.items()]
        return f'({n}) : ' + ', '.join(names)
    print('stateless', format_stat(stateless), '\n')
    print('stateful', format_stat(stateful), '\n')
    print('unresolved', format_stat(unresolved), '\n')
    return cells

LAYER_IDX_0 = 16
LAYER_CLOCK_BUF = 1
LAYER_CLOCK_GATE = 2
LAYER_MEMORY = 3

def export_wires(top_cell):
    wire_rects = []
    wire_infos = []
    layer_z = {lid:i + LAYER_IDX_0 for  i, lid in enumerate(layerstack)}
    for lid, quads in top_cell.gds_cell.get_polygons(by_spec=True, depth=0).items():
        if lid not in layerstack:
            continue
        layer_name = layerstack[lid]['name']
        if layer_name not in top_cell.layers:
            continue
        z = layer_z[lid]
        for q in quads:
            assert len(q) == 4
            center = q.mean(0)
            idx = top_cell.layers[layer_name].query(Point(center), 'intersects')
            assert(len(idx) == 1)
            part = (layer_name, idx[0])
            wire = top_cell.part2wire.get(part, -1)
            if wire <= 1:  # skip power grid
                continue
            (x0, y0), (x1, y1) = q.min(0), q.max(0)
            wire_rects.append([x0, y0, x1, y1])
            wire_infos.append([wire, z])
    return wire_rects, wire_infos

def export_circuit(top_cell, out_fn):
    wire_rects, wire_infos = export_wires(top_cell)

    gate_luts, gate_inputs, gate_outputs = {}, {}, {}
    input_n_stats = Counter()
    def set_gate(inputs, output, lut):
        output = int(output)
        inputs = [int(a) for a in inputs]
        gate_luts[output] = int(lut)
        gate_inputs[output] = inputs
        for i in inputs:
            gate_outputs.setdefault(i, []).append(output)
        input_n_stats[len(inputs)] += 1

    next_wire = len(top_cell.wire2parts)
    for wire in top_cell.pin2wire.values():
        set_gate([wire], wire, 0b10)

    for ref in top_cell.gds_cell.references:
        cell = top_cell.cell_library[ref.ref_cell.name]
        cell2top = top_cell.connect_child(ref)
        outputs = set(cell2top) & cell.wire_by_type['out']
        if len(outputs) == 0:
            continue
        
        hidden_state = cell.wire_by_type['state'] - set(cell2top)
        for out_wire in hidden_state:
            cell2top[out_wire] = next_wire
            next_wire += 1
        for out_wire in hidden_state | outputs:
            out_wire_global = cell2top[out_wire]
            active_inputs, _, lut_bits = cell.get_lut(out_wire)
            active_inputs = [cell2top.get(i) for i in active_inputs]
            assert None not in active_inputs, "missing input wire"
            assert len(active_inputs) <= 6, f"too many inputs ({cell.short_name})"
            set_gate(active_inputs, out_wire_global, lut_bits)
        
        px, py = 0.3, 0.3
        (x0, y0), (x1, y1) = ref.get_bounding_box()
        wire_rects.append([x0+px, y0+py, x1-px, y1-py])
        cell_type = 0
        has_clk = 'clk' in cell.short_name
        if hidden_state and has_clk:
            cell_type = LAYER_CLOCK_GATE
        elif has_clk:
            cell_type = LAYER_CLOCK_BUF
        elif hidden_state:
            cell_type = LAYER_MEMORY
        wire_infos.append([cell2top[outputs.pop()], cell_type])

    export = defaultdict(list)
    for i in range(next_wire+1):
        export['inputs_start'].append(len(export['inputs']))
        export['outputs_start'].append(len(export['outputs']))
        if i<next_wire:
            export['luts'].append(gate_luts.get(i, 0))
            export['inputs'].extend(gate_inputs.get(i, []))
            export['outputs'].extend(gate_outputs.get(i, []))

    meta = dict(bbox=top_cell.bbox.ravel().tolist())
    meta_reserved = 1024*4
    offset = [meta_reserved]
    export_arrays = []
    def add_array(name, a):
        meta[name] = dict(offset=offset[0], shape=a.shape, dtype=a.dtype.name)
        export_arrays.append((offset[0], a))
        size = a.itemsize * a.size
        align = 8
        offset[0] += (size+align-1) // align * align
    rects = np.array(wire_rects, dtype=np.float32).reshape(-1, 2)
    (x0,y0), (x1,y1) = top_cell.bbox
    rects = (rects-(x0,y0)) / (x1-x0, y1-y0)
    rects = rects.reshape(-1,4).clip(0.0, 1.0)
    wire_infos = np.uint16(wire_infos)
    z_order = wire_infos[:,1].argsort()
    add_array('wire_rects', np.uint16(rects[z_order]*((1<<16)-1)))
    add_array('wire_infos', np.uint16(wire_infos[z_order]))
    for k, arr in export.items():
        arr = (np.uint64 if k=='luts' else np.uint16)(arr)
        add_array(k, arr)

    pin_names, pin_wires, pin_pos = [], [], []
    for lab in top_cell.gds_cell.get_labels(depth=0):
        if not (lab.text in ['clk', 'rst_n', 'ena'] or lab.text[0]=='u'):
            continue
        wire = top_cell.pin2wire.get(lab.text)
        if not wire:
            continue
        pin_names.append(lab.text)
        pin_wires.append(wire)
        pin_pos.append(lab.position)
    meta['pins'] = pin_names
    add_array('pin_wires', np.uint16(pin_wires))
    add_array('pin_pos', np.float32(pin_pos))

    meta_json = json.dumps(meta).encode('utf8')
    print(meta_json)
    assert len(meta_json) < meta_reserved

    with open(out_fn.with_suffix('.bin'), 'wb') as f:
        f.write(meta_json)
        f.seek(meta_reserved)
        for ofs, a in export_arrays:
            f.seek(ofs)
            a.tofile(f)

    print('chip bbox:', top_cell.bbox.ravel().tolist())

def fetch_gds(project, cache_dir=Path('gds')):
    tt_index, project = project.split('_', 1)
    cache_dir.mkdir(exist_ok=True)
    name = f'{tt_index}_{project}.gds'
    cache_path = cache_dir / name
    if cache_path.exists():
        return cache_path
    url = f'https://github.com/TinyTapeout/tinytapeout-{tt_index}/raw/refs/heads/main/projects/{project}/{project}.gds'
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    loaded = 0
    tmp_path = cache_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
       for chunk in response.iter_content(chunk_size=1024):
           loaded += f.write(chunk)
           print(f"\rfetching {name} ... {loaded//1024} / {total//1024} kB", end='')
    print()
    tmp_path.rename(cache_path)
    return cache_path

pdk_root = '/Users/moralex/ttsetup/pdk/volare/sky130/versions/bdc9412b3e468c102d01b7cf6337be06ec6e9c9a/'
pdk_gds = pdk_root+'sky130A/libs.ref/sky130_fd_sc_hd/gds/sky130_fd_sc_hd.gds'

projects = [
    '05_tt_um_dinogame',
    #'07_tt_um_pongsagon_tiniest_gpu',
    '09_tt_um_znah_vga_ca',         # 😎
    '07_tt_um_algofoogle_raybox_zero',
    '08_tt_um_a1k0n_vgadonut',
    '09_tt_um_rejunity_vga_test01', # drop
    '08_tt_um_top',                 # 🔥
    '09_tt_um_2048_vga_game',       # shows a grid and 2048
    '08_tt_um_a1k0n_nyancat',       # 08/😎,  09/ doesn't work (only draws one line)
    '09_tt_um_vga_clock',           # 00:00:00
    
    #'09_tt_um_cdc_test',            # airplane
    #'08_tt_um_johshoff_metaballs',  # I see one ball
    #'09_tt_um_warp',                # not sure if vsync is always correct
    #'08_tt_um_a1k0n_vgadonut',      # 🍩, but TODO 2x clock
    #'09_tt_um_toivoh_demo',         # TODO 2x clock
]


if __name__ == '__main__':
    gds = gdspy.GdsLibrary().read_gds('gds/ihp/tt_um_znah_vga_ca.gds')
    cells = analyse_cells(gds)
    print('Top cell analysis ...')
    top_cell = Cell(gds.top_level()[0], cells)
    print('Export ...')
    export_circuit(top_cell, 'ihp.json')

    # for project in projects[:1]:
    #     gds_fn = fetch_gds(project)
    #     print(f'processing {gds_fn}...')
    #     gds = gdspy.GdsLibrary().read_gds(gds_fn)
    #     cells = analyse_cells(gds)
    #     print('Top cell analysis ...')
    #     top_cell = Cell(gds.top_level()[0], cells)
    #     print('Export ...')
    #     export_circuit(top_cell, gds_fn.with_suffix('.json'))
