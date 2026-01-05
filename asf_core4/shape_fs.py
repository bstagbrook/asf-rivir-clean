"""
Shape-based storage and views on top of the ASF catalog.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import importlib.util

from py_to_dyck import compile_source


def load_asf_core2():
    path = Path("/Volumes/StagbrookField/stagbrook_field/.asf_core2.py")
    spec = importlib.util.spec_from_file_location("asf_core2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


@dataclass
class ShapeFS:
    db_path: str

    def __post_init__(self):
        self.core2 = load_asf_core2()
        self.catalog = self.core2.PersistentCatalog(self.db_path)

    def close(self):
        self.catalog.close()

    def put_dyck(self, dyck: str, label: Optional[str] = None, namespace: str = "fs") -> str:
        shape = self.core2.parse_dyck(dyck)
        self.catalog.put(shape)
        if label:
            self.catalog.set_label(shape, label, namespace=namespace, confidence=1.0, notes="shape_fs")
        return self.core2.key(shape)

    def put_python(self, source: str, label: Optional[str] = None, namespace: str = "program") -> str:
        source = source.replace("\\n", "\n")
        dyck = compile_source(source)
        return self.put_dyck(dyck, label=label, namespace=namespace)

    def get_shape(self, key: str):
        row = self.catalog._conn.execute(
            "SELECT canonical_bytes FROM shapes WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return self.core2.parse(row["canonical_bytes"].decode())

    def get_dyck(self, key: str) -> Optional[str]:
        shape = self.get_shape(key)
        if not shape:
            return None
        return self.core2.serialize_dyck(shape).decode()

    def labels(self, key: str):
        shape = self.get_shape(key)
        if not shape:
            return []
        return self.catalog.get_labels(shape)

    def view_label(self, label: str, namespace: str = "fs") -> List[str]:
        return self.catalog.find_by_label(label, namespace=namespace)
