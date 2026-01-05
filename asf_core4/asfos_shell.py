"""
Minimal asfOS shell over the clockless kernel and shape FS.
"""

import argparse
import shlex
from pathlib import Path

from asfos_kernel import Kernel


def run_script(kernel: Kernel, path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            dispatch(kernel, line)
            kernel.run()


def dispatch(kernel: Kernel, line: str):
    parts = shlex.split(line)
    if not parts:
        return
    cmd, args = parts[0], parts[1:]

    if cmd in ("exit", "quit"):
        raise SystemExit(0)
    if cmd == "help":
        print("Commands:")
        print("  put_dyck <dyck> [label] [namespace]")
        print("  put_python <code> [label] [namespace]")
        print("  put_file <path> [label] [namespace]")
        print("  show <key>")
        print("  label <key> <label> [namespace]")
        print("  labels <key>")
        print("  view <label> [namespace]")
        print("  flow_push <name>")
        print("  flow_pop")
        print("  flow_decompose <phase1> <phase2> ...")
        print("  flow_complete")
        print("  flow_status")
        print("  help")
        print("  exit")
        return

    if cmd == "put_dyck":
        if not args:
            print("usage: put_dyck <dyck> [label] [namespace]")
            return
        dyck = args[0]
        label = args[1] if len(args) > 1 else None
        namespace = args[2] if len(args) > 2 else "fs"
        kernel.emit("store_dyck", {"dyck": dyck, "label": label, "namespace": namespace})
        return

    if cmd == "put_python":
        if not args:
            print("usage: put_python <code> [label] [namespace]")
            return
        code = args[0]
        label = args[1] if len(args) > 1 else None
        namespace = args[2] if len(args) > 2 else "program"
        kernel.emit("store_python", {"source": code, "label": label, "namespace": namespace})
        return

    if cmd == "put_file":
        if not args:
            print("usage: put_file <path> [label] [namespace]")
            return
        path = args[0]
        code = Path(path).read_text(encoding="utf-8")
        label = args[1] if len(args) > 1 else None
        namespace = args[2] if len(args) > 2 else "program"
        kernel.emit("store_python", {"source": code, "label": label, "namespace": namespace})
        return

    if cmd == "show":
        if not args:
            print("usage: show <key>")
            return
        dyck = kernel.fs.get_dyck(args[0])
        print(dyck if dyck else "shape not found")
        return

    if cmd == "label":
        if len(args) < 2:
            print("usage: label <key> <label> [namespace]")
            return
        namespace = args[2] if len(args) > 2 else "fs"
        kernel.emit("label", {"key": args[0], "label": args[1], "namespace": namespace})
        return

    if cmd == "labels":
        if not args:
            print("usage: labels <key>")
            return
        labels = kernel.fs.labels(args[0])
        print(labels)
        return

    if cmd == "view":
        if not args:
            print("usage: view <label> [namespace]")
            return
        namespace = args[1] if len(args) > 1 else "fs"
        keys = kernel.fs.view_label(args[0], namespace=namespace)
        print(keys)
        return

    if cmd == "flow_push":
        if not args:
            print("usage: flow_push <name>")
            return
        kernel.emit("flow_push", {"name": args[0]})
        return

    if cmd == "flow_pop":
        kernel.emit("flow_pop", {})
        return

    if cmd == "flow_decompose":
        if not args:
            print("usage: flow_decompose <phase1> <phase2> ...")
            return
        kernel.emit("flow_decompose", {"phases": args})
        return

    if cmd == "flow_complete":
        kernel.emit("flow_complete", {})
        return

    if cmd == "flow_status":
        kernel.emit("flow_status", {})
        return

    print(f"unknown command: {cmd}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="asfos.db", help="SQLite DB path")
    parser.add_argument("--script", help="Run commands from script file")
    args = parser.parse_args()

    kernel = Kernel(args.db)
    kernel.install_default_handlers()

    if args.script:
        run_script(kernel, args.script)
        kernel.close()
        return

    print("asfOS shell (type 'help')")
    while True:
        try:
            line = input("asfOS> ")
            dispatch(kernel, line)
            kernel.run()
        except SystemExit:
            kernel.close()
            return
        except Exception as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
