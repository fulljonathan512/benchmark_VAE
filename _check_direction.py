import json

root = r"d:\02_Studium\03_Bachelor_Informatik_KIT\01_Module\Semester_VIII\Bachelorarbeit\benchmark_VAE"
d = json.load(open(root + r"\dependency_graph.json", encoding="utf-8"))

root_node = d["root"]
edges = [tuple(e) for e in d["edges"]]
nodes = set(d["nodes"])

incoming = {}
for a, b in edges:
    incoming.setdefault(b, set()).add(a)

# requirement: only root has NO incoming edge
roots = sorted({n for n in nodes if not incoming.get(n)})
print("nodes with NO incoming edge:", roots)
ok_root = roots == [root_node]
print("=> only dvae_pipeline has no incoming edge:", ok_root)

# nodes with no edges at all (isolated)
conn = {n for e in edges for n in e}
isolated = sorted(nodes - conn)
print("isolated nodes (no edges):", isolated)

# is root source of all? count reachable from root
succ = {n: set() for n in nodes}
for a, b in edges:
    succ[a].add(b)
seen = set()
stack = [root_node]
while stack:
    n = stack.pop()
    if n in seen:
        continue
    seen.add(n)
    stack.extend(succ.get(n, ()))
print("nodes reachable from root:", len(seen), "of", len(nodes))
print("unreachable from root:", sorted(nodes - seen)[:20])
