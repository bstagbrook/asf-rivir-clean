"""
Clockless kernel: event-driven processing over shapes.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from shape_fs import ShapeFS


@dataclass
class Event:
    type: str
    payload: dict


class Kernel:
    def __init__(self, db_path: str):
        self.fs = ShapeFS(db_path)
        self.queue: List[Event] = []
        self.handlers: Dict[str, Callable[[Event], None]] = {}
        self.flow_stack: List[dict] = []

    def close(self):
        self.fs.close()

    def on(self, event_type: str, handler: Callable[[Event], None]):
        self.handlers[event_type] = handler

    def emit(self, event_type: str, payload: dict):
        self.queue.append(Event(event_type, payload))

    def run(self):
        while self.queue:
            event = self.queue.pop(0)
            handler = self.handlers.get(event.type)
            if handler:
                handler(event)

    def install_default_handlers(self):
        self.on("store_dyck", self._handle_store_dyck)
        self.on("store_python", self._handle_store_python)
        self.on("label", self._handle_label)
        self.on("print", self._handle_print)
        self.on("flow_push", self._handle_flow_push)
        self.on("flow_pop", self._handle_flow_pop)
        self.on("flow_decompose", self._handle_flow_decompose)
        self.on("flow_complete", self._handle_flow_complete)
        self.on("flow_status", self._handle_flow_status)

    def _handle_store_dyck(self, event: Event):
        dyck = event.payload["dyck"]
        label = event.payload.get("label")
        namespace = event.payload.get("namespace", "fs")
        key = self.fs.put_dyck(dyck, label=label, namespace=namespace)
        self.emit("print", {"text": f"stored {key}"})

    def _handle_store_python(self, event: Event):
        source = event.payload["source"]
        label = event.payload.get("label")
        namespace = event.payload.get("namespace", "program")
        key = self.fs.put_python(source, label=label, namespace=namespace)
        self.emit("print", {"text": f"stored {key}"})

    def _handle_label(self, event: Event):
        key = event.payload["key"]
        label = event.payload["label"]
        namespace = event.payload.get("namespace", "fs")
        shape = self.fs.get_shape(key)
        if not shape:
            self.emit("print", {"text": "shape not found"})
            return
        self.fs.catalog.set_label(shape, label, namespace=namespace, confidence=1.0, notes="kernel")
        self.emit("print", {"text": f"labeled {key} as {namespace}:{label}"})

    def _handle_print(self, event: Event):
        print(event.payload["text"])

    def _handle_flow_push(self, event: Event):
        name = event.payload["name"]
        self.flow_stack.append({"name": name, "complete": False, "milestones": []})
        self.emit("print", {"text": f"pushed {name}"})

    def _handle_flow_pop(self, event: Event):
        if not self.flow_stack:
            self.emit("print", {"text": "flow stack empty"})
            return
        node = self.flow_stack.pop()
        self.emit("print", {"text": f"popped {node['name']}"})

    def _handle_flow_decompose(self, event: Event):
        phases = event.payload["phases"]
        if not phases:
            self.emit("print", {"text": "no phases provided"})
            return
        current = phases[0]
        rest = phases[1:]
        for name in reversed(rest):
            self.flow_stack.append({"name": name, "complete": False, "milestones": []})
        self.flow_stack.append({"name": current, "complete": False, "milestones": []})
        self.emit("print", {"text": f"current {current}; pushed {len(rest)} phases"})

    def _handle_flow_complete(self, event: Event):
        if not self.flow_stack:
            self.emit("print", {"text": "flow stack empty"})
            return
        node = self.flow_stack.pop()
        node["complete"] = True
        self.emit("print", {"text": f"completed {node['name']}"})
        if self.flow_stack:
            self.emit("print", {"text": f"current {self.flow_stack[-1]['name']}"})

    def _handle_flow_status(self, event: Event):
        if not self.flow_stack:
            self.emit("print", {"text": "flow stack empty"})
            return
        current = self.flow_stack[-1]["name"]
        stack = [n["name"] for n in self.flow_stack]
        self.emit("print", {"text": f"current {current} stack={stack}"})
