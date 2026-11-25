"""
trace_files.py
==============

Launcher that runs a target Python script while recording every file that is
touched during execution, and writes the results to an output file instead of
the console.

It relies on Python's audit hooks (PEP 578, Python 3.8+) to capture file opens
without requiring changes to the traced script, plus a check of ``sys.modules``
for imported modules:

1. IMPORTS
   After the script finishes, every module in ``sys.modules`` contributes its
   on-disk ``__file__`` path (``.py``/``.pyc``/``.so``/...).

2. OS-LEVEL FILE OPENS
   Audit event ``"open"`` fires for every ``open()`` call, which catches plain
   files too (configs, datasets, databases, model artifacts, ...).

Usage:
    python trace_files.py <script.py> [args...]
    python trace_files.py -- <script.py> [args...]

Output:
    Results are written (UTF-8) to the file given by ``TRACE_OUTPUT`` if set,
    otherwise to ``tracing_output.txt`` in the current directory.
"""

import os
import sys
import time


def _default_out():
    return os.environ.get("TRACE_OUTPUT", os.path.join(os.getcwd(), "tracing_output.txt"))


def _write_all(handle, lines):
    """Write lines (list of str) to handle, joining properly."""
    handle.write("\n".join(lines))
    handle.write("\n")


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        script = argv[i + 1]
        argv = argv[i + 1:]
    elif argv:
        script = argv[0]
        argv = argv[1:]
    else:
        print("Usage: python trace_files.py <script.py> [args...]")
        sys.exit(2)

    out_path = _default_out()
    start = time.time()

    # ------------------------------------------------------------------ #
    # Sources of recorded files
    # ------------------------------------------------------------------ #
    loaded_sources = set()      # module origin (e.g. .../module.py)
    opened_files = set()        # absolute/relative path from open()

    # A single audit hook (PEP 578, Python 3.8+) captures OS-level opens.
    # Imported modules are collected from ``sys.modules`` afterwards, which is
    # simpler and complete.
    def audit(event, args):
        if event == "open":
            path = args[0]
            if path not in opened_files:
                opened_files.add(path)

    sys.addaudithook(audit)

    # ------------------------------------------------------------------ #
    # Run the target script in this process
    # ------------------------------------------------------------------ #
    sys.argv = argv

    # Build the same `__main__` global namespace that `python script.py` uses,
    # so the script can rely on `__file__`, `__spec__`, `__package__`, etc.
    main_globals = {
        "__name__": "__main__",
        "__file__": script,
        "__cached__": None,
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
        "__builtins__": __builtins__,
    }

    with open(script, encoding="utf-8") as f:
        source = f.read()
    compiled = compile(source, script, "exec")
    exec(compiled, main_globals)

    elapsed = time.time() - start

    # Collect every imported module's on-disk file from sys.modules.
    for name, module in list(sys.modules.items()):
        mod_file = getattr(module, "__file__", None)
        if mod_file and mod_file not in loaded_sources:
            loaded_sources.add(mod_file)

    # ------------------------------------------------------------------ #
    # Write results
    # ------------------------------------------------------------------ #
    output_lines = []
    output_lines.append(f"# Trace finished in {elapsed:.2f}s")
    output_lines.append(f"# Script: {script}")
    output_lines.append("")

    output_lines.append(f"## Imported modules ({len(loaded_sources)}):")
    for path in sorted(loaded_sources):
        output_lines.append(path)

    output_lines.append("")
    output_lines.append(f"## Files opened by the OS ({len(opened_files)}):")
    for path in sorted(opened_files):
        output_lines.append(path)

    with open(out_path, "w", encoding="utf-8") as f:
        _write_all(f, output_lines)

    # Keep the console output minimal (summary only).
    print(f"Done. Wrote {len(loaded_sources)} modules and "
          f"{len(opened_files)} opened files to: {out_path}")


if __name__ == "__main__":
    main()
