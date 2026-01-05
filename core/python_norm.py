"""
Python AST normalization for equivalence keys.
"""

import ast
import hashlib


class _Normalizer(ast.NodeTransformer):
    def __init__(self):
        self._stack = []

    def _push_scope(self, mapping):
        self._stack.append(mapping)

    def _pop_scope(self):
        self._stack.pop()

    def _resolve(self, name):
        for scope in reversed(self._stack):
            if name in scope:
                return scope[name]
        return name

    def visit_Lambda(self, node):
        self.generic_visit(node)
        if len(node.args.args) != 1:
            return node
        old = node.args.args[0].arg
        new = f"v{len(self._stack)}"
        mapping = {old: new}
        self._push_scope(mapping)
        node.body = self.visit(node.body)
        self._pop_scope()
        node.args.args[0].arg = new
        return node

    def visit_FunctionDef(self, node):
        mapping = {}
        for i, arg in enumerate(node.args.args):
            mapping[arg.arg] = f"v{i}"
        self._push_scope(mapping)
        node.body = [self.visit(n) for n in node.body]
        self._pop_scope()
        for i, arg in enumerate(node.args.args):
            arg.arg = f"v{i}"
        return node

    def visit_Name(self, node):
        node.id = self._resolve(node.id)
        return node

    def visit_BinOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            if isinstance(node.left.value, int) and isinstance(node.right.value, int):
                if isinstance(node.op, ast.Add):
                    return ast.Constant(node.left.value + node.right.value)
                if isinstance(node.op, ast.Sub):
                    return ast.Constant(node.left.value - node.right.value)
                if isinstance(node.op, ast.Mult):
                    return ast.Constant(node.left.value * node.right.value)
        return node

    def visit_UnaryOp(self, node):
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, int):
                return ast.Constant(-node.operand.value)
        return node

    def visit_Compare(self, node):
        node.left = self.visit(node.left)
        node.comparators = [self.visit(c) for c in node.comparators]
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            if isinstance(node.left, ast.Constant) and isinstance(node.comparators[0], ast.Constant):
                return ast.Constant(node.left.value == node.comparators[0].value)
        return node

    def visit_IfExp(self, node):
        node.test = self.visit(node.test)
        node.body = self.visit(node.body)
        node.orelse = self.visit(node.orelse)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node


def python_equiv_key(source: str, mode: str = "python_ast") -> str:
    tree = ast.parse(source)
    if mode == "python_norm":
        tree = _Normalizer().visit(tree)
        ast.fix_missing_locations(tree)
    dump = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()
