from collections import defaultdict, Counter
from parse_gds import FET, DisjointSets


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