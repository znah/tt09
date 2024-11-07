import numpy as np
import drawsvg as draw
import gdspy
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.strtree import STRtree
from IPython.display import HTML, display, SVG
import io
from collections import Counter, defaultdict
from collections import namedtuple
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
        for p in layers[via].geometries:
            c = p.centroid
            part_a = layers[a].query(c, 'within')
            part_b = layers[b].query(c, 'within')
            if len(part_a)==0 or len(part_b)==0:
                continue
            a_idx, b_idx = int(part_a[0]), int(part_b[0])
            parts.merge(Part(a, a_idx), Part(b, b_idx))

    wires = [layerstack[i, 20]['name'] for i in range(66, 72+1)]
    vias = [layerstack[i, 44]['name'] for i in range(66, 72)]
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

def merge_inverters(fets: set[FET], pin2wire: dict[str,int]):
    rename, remove = {}, []
    for fet in fets:
        if fet.type != 'N' or fet.a != 0:
            continue
        pfet = FET('P', fet.gate, 1, fet.b)
        if pfet not in fets:
            continue
        remove += [fet, pfet]
        rename[fet.b] = -fet.gate
    for k in rename:
        k1 = rename[k]
        while abs(k1) in rename:
            k1 = -rename[abs(k1)]
        rename[k] = k1
    ren = lambda a:rename.get(a,a)
    fets = set(FET(f.type, ren(f.gate), ren(f.a), ren(f.b)) 
               for f in (fets-set(remove)))
    pin2wire = {k:ren(v) for k, v in pin2wire.items()}
    return fets, pin2wire

def merge_gates(fets: set[FET], pin2wire: dict[str,int]):
    pinned_wires = set(pin2wire.values())
    wire2fets = defaultdict(list)
    for fet in fets:
        for i in [fet.gate, fet.a, fet.b]:
            wire2fets[i].append(fet)
    merges = DisjointSets()
    for wire in list(wire2fets):
        if len(wire2fets[wire]) != 2 or wire in pinned_wires:
            continue
        f1, f2 = wire2fets[wire]
        if f1.type != f2.type or wire in [f1.gate, f2.gate]:
            continue
        merges.merge(f1, f2)
    fets = fets - set(merges.nodes)
    for merged in merges.label_components()[0].values():
        gates = tuple(sorted([f.gate for f in merged]))
        cnt = Counter(sum([(f.a, f.b) for f in merged], ()))
        a, b = sorted([wire for wire in cnt if cnt[wire] == 1])
        fet = FET(merged[0].type, gate=gates, a=a, b=b)
        fets.add(fet)
    return fets

def simplify(fets, pin2wire, inverters=True, gates=True):
    if inverters:
        fets, pin2wire = merge_inverters(fets, pin2wire)
    if gates:
        fets = merge_gates(fets, pin2wire)
    return fets, pin2wire

def draw_net(fets, pin2wire, with_gates=True):
    dot = Digraph(graph_attr={'rankdir':'LR'}, node_attr={'shape':'square'}, engine='dot') #neato dot
    colors = {'N':'lightgreen', 'P':'coral'}
    power_wires = set([0,1])
    for pin, wire in pin2wire.items():
        if wire not in power_wires:
            dot.node(f'{abs(wire)}', label='~'*(wire<0)+pin)
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

def guess_inouts(fets, pin2wire):
    has_gate, has_diff = set(), set()
    for fet in fets:
        has_gate.add(fet.gate)
        has_diff.update((fet.a, fet.b))
    inputs, outputs = {}, {}
    for pin, wire in sorted(pin2wire.items()):
        if pin in power_pins:
            continue
        if wire in has_gate and wire not in has_diff:
            inputs[pin] = wire
        elif wire in has_diff or wire <= 1: # VGND and VPWR are outputs
            outputs[pin] = wire
    return inputs, outputs

def get_wires(fets):
    wires = {}
    for fet in fets:
        for wire in [fet.gate, fet.a, fet.b]:
            wires.setdefault(wire, []).append(fet)
    return wires

WIRE_0 = 0
WIRE_1 = 1
WIRE_FLOAT = 2
WIRE_UNDEFINED = 3
WIRE_SHORT = 4

def build_truth_table(fets: set[FET], inputs, outputs):
    '''
    inputs, outputs: dict[pin, wire_idx]
    '''
    input_n = len(inputs)
    assert input_n <= 6
    wires = get_wires(fets)
    wire_n = max(wires.keys())+1 if wires else 2

    case_n = 2**input_n
    case_i = np.arange(case_n, dtype=np.uint64)
    input_cases = (case_i >> np.arange(input_n,  dtype=np.uint64)[:,None])&1
    input_bits = np.uint64(input_cases*2**case_i).sum(1)

    mask = np.uint64(2**case_n-1)
    signals = np.zeros([2, wire_n], np.uint64)
    input_idx = list(inputs.values())
    signals[0, 0] = signals[1, 1] = mask  # VGND VPWR
    signals[1, input_idx] = input_bits
    signals[0, input_idx] = (~input_bits)&mask

    queue = list(input_idx)
    while queue:
        wire = queue.pop()
        for fet in wires[wire]:
            gate_level = int(fet.type=='N')
            level = signals[1-gate_level]
            gate = signals[gate_level][fet.gate]
            a, b = level[fet.a], level[fet.b]
            c = (a|b)&gate
            a1, b1 = a|c, b|c
            if a!=a1:
                queue.append(fet.a)
                level[fet.a] = a1
            if b!=b1:
                queue.append(fet.b)
                level[fet.b] = b1
    
    output_idx = list(outputs.values())
    out_signals = (signals[:,output_idx,None] >> case_i)&1
    lut_bits = signals[1, output_idx]
    results = out_signals[1]

    defined = signals[0] ^ signals[1]
    # check if all outputs are defined
    if (defined[output_idx] == mask).all():
        return input_cases, results, lut_bits
    
    results[(out_signals[0]==0) & (out_signals[1]==0)] = WIRE_UNDEFINED
    results[(out_signals[0]==1) & (out_signals[1]==1)] = WIRE_SHORT
    # undefined wires that are connected to defined (but closed) FETs
    # are set to the 'floating' state (tri-state buffers)
    for i, out_wire in enumerate(output_idx):
        src_gates = [fet.gate for fet in wires[out_wire] if fet.gate != out_wire]
        gates_defined = np.bitwise_and.reduce(defined[src_gates])
        gates_defined = ((gates_defined >> case_i)&1) == 1
        results[i, (results[i]==WIRE_UNDEFINED) & gates_defined] = WIRE_FLOAT
    lut_bits = signals[1, output_idx]
    return input_cases, results, lut_bits

class Cell:
    def __init__(self, gds_cell, cell_library):
        self.gds_cell = gds_cell
        self.cell_library = cell_library
        self.name = gds_cell.name
        self.short_name = gds_cell.name.split('__')[-1]
        self.bbox = gds_cell.get_bounding_box()

        self.layers = extract_layers(gds_cell)
        self.pin2part = find_pins(self.layers, gds_cell)
        self.part2pin = {v:k for k,v in self.pin2part.items()}  # TODO: multiple pins? 
        self.part_sets = connect_layers(self.layers)
        for part in self.pin2part.values():
            self.part_sets.add(part)
        known_wires = {self.pin2part[name]:i for name, i in [('VGND', 0), ('VPWR', 1)]}
        self.wire2parts, self.part2wire = self.part_sets.label_components(known_wires)
        self.pin2wire = {pin:self.part2wire[part] for pin, part in self.pin2part.items()}
                
        self.fets = extract_fets(self.part2wire, self.layers)

        #self.fets1, self.pin2wire1 = simplify(self.fets, self.pin2wire)

        self.input_wires, self.output_wires = guess_inouts(
            self.fets, self.pin2wire)
        self.lut_in, self.lut_out, self.lut_bits = build_truth_table(
            self.fets, self.input_wires, self.output_wires)
        self.resolved = (self.lut_out<=WIRE_FLOAT).all()

    def __repr__(self):
        return f'[{self.short_name}]'

    def get_part(self, part):
        layer, idx = part
        return self.layers[layer].geometries[idx]
    
    def print_table(self):
        print(f'-- {self.short_name} --')
        print(f'[{" ".join(self.input_wires.keys())}] [{" ".join(self.output_wires.keys())}]')
        for i, o in zip(self.lut_in.T, self.lut_out.T):
            print(i, o)

    def connect_pins(self, ref, pin_wires):
        cell = self.cell_library[ref.ref_cell.name]
        parent_wires = [None] * len(pin_wires)
        for i, wire in enumerate(pin_wires):
            for part in cell.wire2parts[wire]:
                lid, part_idx = part
                if lid not in ['li1', 'met1']:
                    continue
                geom = cell.layers[lid].geometries[part_idx]
                if ref.rotation:
                    geom = shapely.affinity.rotate(geom, ref.rotation, origin=(0,0))
                if ref.x_reflection:
                    geom = shapely.affinity.scale(geom, 1, -1, origin=(0,0))
                geom = shapely.affinity.translate(geom, *ref.origin)
                match_idx = self.layers[lid].query(geom, "intersects")
                if len(match_idx) != 1:
                    continue
                parent_wires[i] = self.part2wire[lid, match_idx[0]]
        return parent_wires


def analyse_cells(gds):
    resolved, stateful = [], []
    cells = {}
    for cell_name, gds_cell in gds.cells.items():
        short = cell_name.split('__')[-1]
        print(f'\r{short.ljust(30)}', end='')
        if len(gds_cell.references) > 0:
            print('- skipping')
            continue
        cells[cell_name] = cell = Cell(gds_cell, cells)
        (resolved if cell.resolved else stateful).append(short)
    print('\r', end='')
    def format_stat(names):
        n = len(names)
        names = Counter([s.rsplit('_', 1)[0] for s in names])
        names = [f'{s}*{n}' if n>1 else s for s, n in names.items()]
        return f'({n}) : ' + ', '.join(names)
    print('resolved', format_stat(resolved), '\n')
    print('stateful?', format_stat(stateful), '\n')
    return cells

def export_wires(top_cell):
    wire_rects = []
    wire_infos = []
    for lid, quads in top_cell.gds_cell.get_polygons(by_spec=True, depth=0).items():
        if lid not in layerstack:
            continue
        layer_name = layerstack[lid]['name']
        if layer_name not in top_cell.layers:
            continue
        z = lid[0]-66
        for q in quads:
            assert len(q) == 4
            center = q.mean(0)
            idx = top_cell.layers[layer_name].query(Point(center), 'intersects')
            assert(len(idx) == 1)
            part = (layer_name, idx[0])
            wire = top_cell.part2wire.get(part, -1)
            if wire <= 1:
                continue
            (x0, y0), (x1, y1) = q.min(0), q.max(0)
            wire_rects += [x0, y0, x1, y1]
            wire_infos += [wire, z]
    return wire_rects, wire_infos

def export_circuit(top_cell, out_f):
    wire_rects, wire_infos = export_wires(top_cell)

    gate_luts, gate_inputs, gate_outputs = {}, {}, {}
    def set_gate(inputs, output, lut):
        output = int(output)
        inputs = [int(a) for a in inputs]
        gate_luts[output] = int(lut)
        gate_inputs[output] = inputs
        for i in inputs:
            gate_outputs.setdefault(i, []).append(output)

    LUT_DLATCH_PE = 0b_10111000
    LUT_DLATCH_NE = 0b_11100010

    for wire in top_cell.pin2wire.values():
        set_gate([wire], wire, 0b10)

    next_wire = len(top_cell.wire2parts)
    for ref in top_cell.gds_cell.references:
        cell = top_cell.cell_library[ref.ref_cell.name]
        inputs = top_cell.connect_pins(ref, cell.input_wires.values())
        outputs = top_cell.connect_pins(ref, cell.output_wires.values())
        if not any(outputs):
            continue
        assert None not in inputs
        output_wire = outputs[0] or outputs[1]
        out_i = 0 if outputs[0] else 1
        
        if output_wire == 573:
            print(ref.properties[61], cell, output_wire)
            print(cell.lut_bits[out_i])
        px, py = 0.3, 0.3
        (x0, y0), (x1, y1) = ref.get_bounding_box()
        wire_rects += [x0+px, y0+py, x1-px, y1-py]
        wire_infos += [output_wire, 0]

        if 'dfxtp' in cell.name:  # d flip-flop
            pin2wire = (dict(zip(cell.input_wires.keys(), inputs)))
            CLK, D, Q = pin2wire['CLK'], pin2wire['D'], output_wire
            D1 = next_wire
            next_wire += 1
            set_gate([D, CLK, D1], D1, LUT_DLATCH_NE)
            set_gate([D1, CLK, Q], Q, LUT_DLATCH_PE)
        elif 'dlclkp' in cell.name:  # clock gate
            pin2wire = (dict(zip(cell.input_wires.keys(), inputs)))
            CLK, GATE, GCLK = pin2wire['CLK'], pin2wire['GATE'], output_wire
            E = next_wire
            next_wire += 1
            set_gate([GATE, CLK, E], E, LUT_DLATCH_NE)
            set_gate([E, CLK], GCLK, 0b1000) # and
        else:
            assert cell.resolved and len(inputs) <=6
            set_gate(inputs, output_wire, cell.lut_bits[out_i])

    export = defaultdict(list)
    for i in range(next_wire+1):
        export['inputs_start'].append(len(export['inputs']))
        export['outputs_start'].append(len(export['outputs']))
        if i<next_wire:
            export['luts'].append(gate_luts.get(i, 0))
            export['inputs'].extend(gate_inputs.get(i, []))
            export['outputs'].extend(gate_outputs.get(i, []))

    out_f.write('pins:%s,\n' % str(top_cell.pin2wire))
    out_f.write('wire_rects:[%s],\n' % (','.join('%.3f'%v for v in wire_rects)))
    out_f.write('wire_infos:[%s],\n' % (','.join('%d'%v for v in wire_infos)))
    out_f.write('gates:%s,\n'%str(dict(export)))


pdk_root = '/Users/moralex/ttsetup/pdk/volare/sky130/versions/bdc9412b3e468c102d01b7cf6337be06ec6e9c9a/'
pdk_gds = pdk_root+'sky130A/libs.ref/sky130_fd_sc_hd/gds/sky130_fd_sc_hd.gds'    

vga_gds = 'GDS_logs/runs/wokwi/final/gds/tt_um_znah_vga_ca.gds'

if __name__ == '__main__':
    gds = gdspy.GdsLibrary().read_gds(vga_gds)
    cells = analyse_cells(gds)
    cells['sky130_fd_sc_hd__and3_1'].print_table()
    cells['sky130_fd_sc_hd__conb_1'].print_table()
    print('Top cell analysis ...')
    top_cell = Cell(gds.top_level()[0], cells)
    print('Export ...')
    with open('circuit.js', 'w') as f:
        f.write('const CIRCUIT = {\n')
        export_circuit(top_cell, f)
        f.write('};\n')

    #print(f'connected wire n: {len(top_cell.wire2refs)} of {len(top_cell.wire2parts)}')
