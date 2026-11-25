"""
generate_dependency_graph.py
=============================

Builds a dependency graph of the run of ``mnist_models/dvae_pipeline.py`` and
writes it as a TikZ figure to ``abhaengigkeitsgraph.tex``.

Data sources
------------
1. ``tracing_output.txt`` (created by ``trace_files.py``):
   the *set* of files that are actually used (imported modules + OS opens).
   Filtered to files inside the project root, excluding ``Lib\\``.
   ``__pycache__/X.cpython-313.pyc`` is normalised back to the human-readable
   ``X.py`` source file.

2. Static ``ast`` analysis of those ``.py`` sources:
   ``trace_files.py`` only records a *flat set* - it does NOT record edges
   ("who imports whom"). Therefore the call hierarchy is reconstructed by
   resolving every ``import`` / ``from ... import`` statement to one of the
   files that participated in the run.

Output
------
``abhaengigkeitsgraph.tex``: a ``tikzpicture`` with one node per ``.py`` file
(a unique generated key + the relative path as label) and one directed edge
importer -> imported per resolved dependency. Nodes are arranged in top-down
hierarchy layers by their distance from the root script so that the order and
nesting of the call chain becomes visible.

Usage
-----
    python generate_dependency_graph.py
"""
import ast
import json
import os
import zlib
from pathlib import Path


# --------------------------------------------------------------------------- #
# 1. Read + normalise the trace                                              #
# --------------------------------------------------------------------------- #
def load_trace_files() -> set:
    """Return the set of normalised ``*.py`` paths (posix-style relative)
    inside the project, excluding the venv ``Lib`` subtree."""
    nodes = set()

    with open(TRACE_FILE, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            norm = line.replace("\\", os.sep)
            proj = str(PROJECT_ROOT).replace("\\", os.sep) + os.sep

            if not norm.startswith(proj):
                continue
            rel = norm[len(proj):]
            if rel.startswith("Lib" + os.sep):
                continue

            if rel.endswith(".pyc"):
                rel = relpyc_to_py(rel)
            if not rel.endswith(".py"):
                continue

            if rel in IGNORED_FILES:
                continue

            nodes.add(rel.replace(os.sep, "/"))

    return nodes


def relpyc_to_py(rel: str) -> str:
    """Map ``...\\__pycache__\\X.cpython-313.pyc`` -> ``...\\X.py``."""
    parts = rel.split(os.sep)
    new_parts = []
    skip_next = False
    for p in parts:
        if skip_next:
            skip_next = False
            continue
        if p == "__pycache__":
            skip_next = True
            continue
        new_parts.append(p)
    if new_parts:
        last = new_parts[-1]
        stem = last.split(".cpython-3")[0]
        if last.endswith(".pyc"):
            new_parts[-1] = stem + ".py"
    return os.sep.join(new_parts)

PROJECT_ROOT = Path(__file__).resolve().parent
TRACE_FILE = PROJECT_ROOT / "tracing_output.txt"
OUTPUT_TEX = PROJECT_ROOT / "abhaengigkeitsgraph.tex"
OUTPUT_JSON = PROJECT_ROOT / "dependency_graph.json"
ROOT_SCRIPT = "mnist_models/dvae_pipeline.py"

PYTHAE_ROOT = os.path.join("src", "pythae")
LIB_DIR = PROJECT_ROOT / "Lib"

# Helper that started the trace is not part of the analysed run itself.
IGNORED_FILES = {"trace_files.py", "generate_dependency_graph.py", "_validate_graph.py"}


# --------------------------------------------------------------------------- #
# 2. Build module-name index for resolving imports                            #
# --------------------------------------------------------------------------- #
def module_index(nodes: set) -> dict:
    """Map importable module names to (posix relative) file paths.

    ``src/pythae/__init__.py``            -> ``pythae``
    ``src/pythae/models/__init__.py``     -> ``pythae.models``
    ``src/pythae/models/base/base_model.py`` -> ``pythae.models.base.base_model``
    ``mnist_models/dvae_pipeline.py``     -> ``dvae_pipeline``
    """
    idx = {}
    prefix = PYTHAE_ROOT.replace(os.sep, "/") + "/"

    for rel in nodes:
        if rel == ROOT_SCRIPT or (rel.startswith("mnist_models/")
                                  and rel == ROOT_SCRIPT):
            idx[Path(rel).stem] = rel
            continue

        if rel.startswith(prefix):
            inner = rel[len(prefix):]
            parts = inner.split("/")
            if parts[-1] == "__init__.py":
                idx["pythae." + ".".join(parts[:-1])] = rel
            else:
                stem = parts[-1][:-3]
                idx["pythae." + ".".join(parts[:-1] + [stem])] = rel
    return idx


def package_of_file(rel: str) -> str:
    """Return the dotted package containing ``rel`` (directory path).

    ``src/pythae/models/dvae/dvae_model.py`` -> ``pythae.models.dvae``
    ``src/pythae/models/__init__.py``        -> ``pythae.models``
    ``mnist_models/dvae_pipeline.py``        -> ``mnist_models`` (fallback)
    """
    prefix = PYTHAE_ROOT.replace(os.sep, "/") + "/"
    rel = rel.replace(os.sep, "/")
    d = rel.rsplit("/", 1)[0]   # directory part
    if rel.startswith(prefix):
        inner = d[len(prefix):]
        if inner:
            return "pythae." + inner.replace("/", ".")
        return "pythae"
    return d.replace("/", ".")


def _resolve_import(module: str, names: list, idx: dict) -> set:
    """Files targeted by ``from <module> import <names>`` (module may be '')."""
    cands = set()
    if module:
        cands.add(module)
    for n in names:
        cands.add((module + "." + n) if module else n)
    return {idx[m] for m in cands if m in idx}




# --------------------------------------------------------------------------- #
# 3. Resolve imports of one file                                              #
# --------------------------------------------------------------------------- #




def file_imports(rel: str, nodes: set, idx: dict) -> set:
    """Return set of destination files imported by the file ``rel``.

    Edge direction: the current file is the *caller*, imported files are the
    *callees* (arrow importer -> imported).
    """
    path = PROJECT_ROOT / rel
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=str(path))
    except (OSError, SyntaxError):
        return set()

    pkg = package_of_file(rel)
    targets = set()

    for node_ast in ast.walk(tree):
        if isinstance(node_ast, ast.Import):
            for alias in node_ast.names:
                # `import pythae.x` -> direct module name
                if alias.name in idx:
                    targets.add(idx[alias.name])

        elif isinstance(node_ast, ast.ImportFrom):
            level = node_ast.level or 0     # 0 = absolute, >=1 = relative
            mod = node_ast.module

            if level == 0:
                # absolute import: use the module string as-is
                target_module = mod or ""
                target_module = target_module.lstrip(".")
            else:
                # relative import: combine the package base with module part
                base = pkg
                base_parts = base.split(".") if base else []
                # `from .   ` stays in the package; `from ..` climbs one level
                for _ in range(level - 1):
                    if len(base_parts) > 1:
                        base_parts.pop()
                if mod:
                    target_module = ".".join(base_parts + [mod]) if base_parts \
                        else mod
                else:
                    target_module = ".".join(base_parts)

            names = [a.name for a in node_ast.names] or []
            targets |= _resolve_import(target_module, names, idx)

    targets.discard(rel)
    return targets


def add_parent_package_edges(edges: set, nodes: set) -> set:
    """Return the *ancestor package-__init__* edges to add.

    Importing a submodule executes the ``__init__.py`` of every parent package
    too (e.g. importing ``pythae.data.datasets`` also runs ``pythae/__init__.py``
    and ``pythae/data/__init__.py``). These are runtime ancestors, not direct
    import statements, so they are kept apart from the direct import edges and
    rendered as dotted/light edges later.
    """
    init_dir = {}
    for n in nodes:
        if n.endswith("/__init__.py"):
            init_dir[n.rsplit("/", 1)[0]] = n

    ancestor = set()
    for a, b in edges:
        parts = b.split("/")
        for i in range(len(parts) - 1, 0, -1):
            d = "/".join(parts[:i])
            if d == "src":
                break  # the source dir is not a module
            if d in init_dir:
                pkg = init_dir[d]
                if pkg != b and pkg != a:
                    ancestor.add((a, pkg))
    return ancestor




# --------------------------------------------------------------------------- #
# 4. Layout - hierarchy levels                                                #
# --------------------------------------------------------------------------- #
def compute_layout(nodes: set, edges: set, root: str) -> dict:
    """Return node -> level (BFS depth) starting from ``root``.

    Edges go importer -> imported. We walk root -> imported -> imported ...
    so that mirror files at level 1 are what the root imports directly.
    """
    succ = {n: set() for n in nodes}
    for a, b in edges:
        if a in succ:
            succ[a].add(b)

    level = {}
    if root in nodes:
        level[root] = 0
    frontier = [root] if root in nodes else []
    while frontier:
        nxt = []
        for n in frontier:
            for s in succ.get(n, ()):
                if s not in level:
                    level[s] = level[n] + 1
                    nxt.append(s)
        frontier = nxt

    mx = max(level.values()) if level else 0
    for n in sorted(nodes):
        if n not in level:
            level[n] = mx + 1
    return level


def tree_layout(nodes: set, edges: set, root: str):
    """Build a two-dimensional tree layout.

    We take a spanning tree of the (importer -> imported) relation rooted at
    ``root`` via BFS. Nodes are then arranged so that every parent is centred
    above its direct tree children (a Reingold-Tilford-like layout), which
    makes the hierarchical call structure clearly visible as a 2D tree:
        - y : depth in the tree (root at 0, children below)
        - x : position of the subtree within its own layer

    Returns a dict node -> (x, y).
    """
    # --- successor graph --------------------------------------------------
    succ = {n: set() for n in nodes}
    for a, b in edges:
        if a in succ:
            succ[a].add(b)

    # --- spanning tree: assign each node one BFS parent -------------------
    parent = {}
    children = {n: [] for n in nodes}
    depth = {}

    if root not in nodes:
        return {n: (0.0, 0.0) for n in nodes}, 0

    parent[root] = None
    depth[root] = 0
    frontier = [root]
    while frontier:
        nxt = []
        for n in frontier:
            for s in sorted(succ.get(n, ())):
                if s not in parent:
                    parent[s] = n
                    depth[s] = depth[n] + 1
                    children[n].append(s)
                    nxt.append(s)
        frontier = nxt

    # --- attach any remaining (disconnected) nodes under the root ---------
    base = max(depth.values()) if depth else 0
    for n in sorted(nodes):
        if n not in parent:
            parent[n] = root
            depth[n] = base + 1
            children[root].append(n)

    # --- Reingold-Tilford style x-assignment ------------------------------
    # leaves get successive integer slots; parents are centred over children.
    slot = {"count": 0}
    xpos = {}

    def place(n):
        kids = children.get(n, [])
        if not kids:
            xpos[n] = slot["count"]
            slot["count"] += 1
            return
        for c in kids:
            place(c)
        xpos[n] = sum(xpos[c] for c in kids) / len(kids)

    place(root)

    # --- scale into coordinates -------------------------------------------
    node_spacing = 2.0     # horizontal distance between leaf slots
    level_spacing = 1.6    # vertical distance between depth levels
    # Anchor the ROOT node at the origin (0,0): subtract its horizontal
    # position so the root sits exactly at x=0, while children spread out
    # below it (negative y) and to the right/left around the root.
    x_root = xpos.get(root, 0.0)
    positions = {}
    for n in nodes:
        positions[n] = ((xpos[n] - x_root) * node_spacing, -depth[n] * level_spacing)

    max_level = max(depth.values())
    return positions, max_level



# --------------------------------------------------------------------------- #
# 5. TikZ rendering helpers                                                   #
# --------------------------------------------------------------------------- #
def sanitise_key(rel: str) -> str:
    """Stable, unique, TikZ-safe node key (deterministic across runs)."""
    return "n" + str(zlib.crc32(rel.encode("utf-8")) & 0xFFFFFFFF)


def escape_label(rel: str) -> str:
    """Escape TeX-special characters; normalise slashes to ``/``."""
    repl = {
        "_": r"\_",
        "#": r"\#",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "{": r"\{",
        "}": r"\}",
    }
    out = []
    for ch in rel:
        if ch == "\\":
            out.append("/")
        else:
            out.append(repl.get(ch, ch))
    return "".join(out)


def short_label(rel: str) -> str:
    """Compact label: strip the common ``src/pythae/`` prefix when possible."""
    prefix = PYTHAE_ROOT.replace(os.sep, "/") + "/"
    if rel.startswith(prefix):
        return "pythae." + rel[len(prefix):]
    return rel


def main():
    nodes = load_trace_files()
    idx = module_index(nodes)

    print(f"Loaded {len(nodes)} python source files from the trace.")

    if ROOT_SCRIPT not in nodes:
        p = PROJECT_ROOT / ROOT_SCRIPT
        if p.exists():
            nodes.add(ROOT_SCRIPT)
            idx = module_index(nodes)

    import_edges = set()
    for rel in nodes:
        for dest in file_imports(rel, nodes, idx):
            if dest in nodes and dest != rel:
                import_edges.add((rel, dest))

    print(f"Resolved {len(import_edges)} direct import edges.")
    ancestor_edges = add_parent_package_edges(import_edges, nodes)
    print(f"  +{len(ancestor_edges)} ancestor package-__init__ edges (dotted).")
    edges = import_edges | ancestor_edges
    print(f"Total edges: {len(edges)}")

    positions, max_level = tree_layout(nodes, edges, ROOT_SCRIPT)

    keys = {n: sanitise_key(n) for n in nodes}
    lines = []
    lines.append(r"% Autogenerated by generate_dependency_graph.py")
    lines.append(r"% Dependency graph of the run of mnist_models/dvae_pipeline.py")
    lines.append(r"% Fine-grained; files in the project, excluding Lib\.")
    lines.append(r"\documentclass[border=5pt]{standalone}")
    lines.append(r"\usepackage{tikz}")
    lines.append(r"\usetikzlibrary{arrows.meta}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{tikzpicture}[")
    lines.append(r"  nstyle/.style={draw, rounded corners=2pt, align=center,")
    lines.append(r"    font=\tiny, inner sep=2pt},")
    lines.append(r"  ROOT/.style={fill=red!20},")
    lines.append(r"  MNIST/.style={fill=blue!12},")
    lines.append(r"]")
    lines.append(r"  % --- nodes (sorted by x-position) ---")

    for n in sorted(nodes, key=lambda n: (positions[n][0], positions[n][1], n)):
        style = []
        if n == ROOT_SCRIPT:
            style.append("ROOT")
        elif n.startswith("mnist_models/"):
            style.append("MNIST")
        opts = (",").join(["nstyle"] + style)
        x, y = positions[n]
        lines.append(
            f"  \\node[{opts}] ({keys[n]}) at ({x:.2f},{y:.2f}) "
            f"{{{escape_label(short_label(n))}}};"
        )

    lines.append(r"  % --- edges (direct imports: solid; ancestor package-__init__: dotted/gray) ---")
    for a, b in sorted(import_edges):
        lines.append(f"  \\draw[->,>={{stealth}}, thin] ({keys[a]}) -- ({keys[b]});")
    for a, b in sorted(ancestor_edges):
        lines.append(
            f"  \\draw[->,>={{stealth}}, thin, dotted, gray!70] ({keys[a]}) -- ({keys[b]});"
        )

    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")

    tex = "\n".join(lines) + "\n"
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")

    # structural verification
    used = {n for side in edges for n in side}
    missing = [n for n in nodes if keys[n] not in tex]
    duplicate = len(keys) != len(set(keys.values()))
    print("Verify: all node keys present =", not missing)
    print("Verify: node keys unique =", not duplicate)
    if missing:
        print("  missing nodes:", missing[:10])

    payload = {
        "root": ROOT_SCRIPT,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_direct_edges": len(import_edges),
        "num_ancestor_edges": len(ancestor_edges),
        "max_level": max_level,
        "nodes": sorted(nodes),
        "edges": [list(e) for e in sorted(edges)],
        "direct_edges": [list(e) for e in sorted(import_edges)],
        "ancestor_edges": [list(e) for e in sorted(ancestor_edges)],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

