"""Replay interprocedural counterexamples from trusted project source, not quotes.

The certificate supplies only an exported C entry and concrete int/null inputs.
Calls, argument positions, returns, assignments and taken guards are reconstructed
from ASTs. Supported semantics: signed 32-bit int, null int pointers, direct calls,
locals, if/else, return, integer comparisons/arithmetic. Unsupported operations,
ambiguous linkage, macros, loops, casts and preceding undefined behavior reject.
This proves a conditional API counterexample, not whole-program reachability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from pathlib import PurePosixPath

from tree_sitter_languages import get_parser


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Value:
    number: int
    kind: str = "int"
    edges: frozenset[str] = frozenset()


class Fault(Exception):
    def __init__(self, path, function, node, kind, edges):
        self.path, self.function, self.node, self.kind = path, function, node, kind
        self.edges = edges


class Returned(Exception):
    def __init__(self, value):
        self.value = value


def _path(path):
    # Resolve only against the supplied snapshot, never read a model-selected file.
    return str(PurePosixPath(str(path).replace("\\", "/")))


class Replay:
    def __init__(self, sources, target):
        if not isinstance(sources, dict) or not sources:
            raise Unsupported("no trusted project source snapshot was supplied")
        if len(sources) > 2048 or sum(len(s) for s in sources.values() if isinstance(s, str)) > 16_000_000:
            raise Unsupported("the source snapshot exceeds the replay resource limit")
        self.sources = {_path(p): s for p, s in sources.items()}
        if len(self.sources) != len(sources):
            raise Unsupported("conflicting source paths in the snapshot")
        self.functions = {}
        self.trace = []
        self.steps = 0
        self.depth = 0
        self.target = target
        for path, source in self.sources.items():
            if not isinstance(source, str):
                raise Unsupported("source must be a string")
            if not path.endswith(".c"):
                continue
            raw = source.encode()
            root = get_parser("c").parse(raw).root_node
            if root.has_error:
                continue  # An unrelated broken file cannot supply a definition.
            for node in root.named_children:
                if node.type != "function_definition":
                    continue
                dec = node.child_by_field_name("declarator")
                if dec.type != "function_declarator":
                    continue
                name_node = dec.child_by_field_name("declarator")
                if name_node.type != "identifier":
                    continue
                name = raw[name_node.start_byte:name_node.end_byte].decode()
                self.functions.setdefault(name, []).append((path, raw, root, node))

    def tick(self):
        self.steps += 1
        if self.steps > 2000:
            raise Unsupported("the certificate exceeds the execution-step limit")

    def text(self, frame, node):
        return frame[1][node.start_byte:node.end_byte].decode()

    def record(self, frame, node, event, **fields):
        self.trace.append(dict(event=event, file=frame[0],
                               line=node.start_point[0] + 1,
                               text=self.text(frame, node), **fields))

    def coerce(self, value, kind):
        if not isinstance(value, Value):
            raise Unsupported("missing return value")
        if kind == "ptr":
            if value.number != 0:
                raise Unsupported("non-null pointers, objects and aliases are unsupported")
            return Value(0, "ptr", value.edges)
        if value.kind != "int" or not -(2**31) <= value.number < 2**31:
            raise Unsupported("integer type mismatch or a preceding overflow")
        return value

    def decl(self, frame, node):
        typ = node.child_by_field_name("type")
        if typ is None or self.text(frame, typ) not in {"int", "void"}:
            raise Unsupported("only int/void with null int* parameters are supported")
        if any(c.type in {"type_qualifier", "storage_class_specifier"}
               for c in node.named_children):
            # Function static is handled separately; local statics/volatile reject.
            raise Unsupported("qualifiers or local storage duration are unsupported")
        dec = node.child_by_field_name("declarator")
        if dec is None:
            raise Unsupported("declaration has no declarator")
        kind = "int"
        if dec.type == "pointer_declarator":
            if self.text(frame, typ) != "int":
                raise Unsupported("only null int* pointers are supported")
            if any(c.type == "type_qualifier" for c in dec.named_children):
                raise Unsupported("pointer qualifiers are unsupported")
            kind = "ptr"
            dec = dec.child_by_field_name("declarator")
        if dec.type != "identifier":
            raise Unsupported("arrays, function pointers and complex declarators are unsupported")
        return self.text(frame, dec), kind

    def resolve(self, name, caller=None, entry=None):
        defs = self.functions.get(name, [])
        local = [f for f in defs if f[0] == caller and self.is_static(f)]
        visible = local or [f for f in defs if not self.is_static(f)]
        if entry is not None:
            visible = [f for f in visible if f[0] == entry]
            # Reject duplicate exported definitions even if the certificate names one.
            if len([f for f in defs if not self.is_static(f)]) != 1:
                raise Unsupported("the entry symbol does not link uniquely")
        if len(visible) != 1:
            raise Unsupported(f"function {name} is missing, indirect, or not uniquely defined")
        frame = visible[0]
        root = frame[2]
        for n in root.named_children:
            if n.type not in {"function_definition", "declaration", "comment"}:
                raise Unsupported("macros, conditional compilation and unknown top-level constructs are unsupported")
            if n.type == "declaration":
                # Only direct function prototypes; no globals/typedef contracts.
                d = n.child_by_field_name("declarator")
                if d is None or d.type != "function_declarator":
                    raise Unsupported("global variables and complex declarations are unsupported")
        return frame

    def signature(self, frame, node):
        typ = node.child_by_field_name("type")
        dec = node.child_by_field_name("declarator")
        if typ is None or self.text(frame, typ) not in {"int", "void"} or dec.type != "function_declarator":
            raise Unsupported("this function prototype type is unsupported")
        params = dec.child_by_field_name("parameters").named_children
        if len(params) == 1 and self.text(frame, params[0]) == "void":
            params = []
        kinds = []
        for param in params:
            ptype = param.child_by_field_name("type")
            pdec = param.child_by_field_name("declarator")
            if param.type != "parameter_declaration" or ptype is None or self.text(frame, ptype) != "int":
                raise Unsupported("this function prototype parameter type is unsupported")
            if any(c.type == "type_qualifier" for c in param.named_children):
                raise Unsupported("function prototype qualifiers are unsupported")
            if pdec is not None and pdec.type not in {"identifier", "pointer_declarator", "abstract_pointer_declarator"}:
                raise Unsupported("complex function prototypes are unsupported")
            kinds.append("ptr" if pdec is not None and pdec.type in {"pointer_declarator", "abstract_pointer_declarator"} else "int")
        return self.text(frame, typ), kinds

    def is_static(self, frame):
        return any(c.type == "storage_class_specifier" and self.text(frame, c) == "static"
                   for c in frame[3].named_children)

    def call(self, frame, args):
        if self.depth >= 16:
            raise Unsupported("the call depth exceeds the limit")
        function = frame[3]
        if any(c.type not in {"primitive_type", "function_declarator", "compound_statement",
                              "storage_class_specifier", "comment"} for c in function.named_children):
            raise Unsupported("function type qualifiers or extended attributes are unsupported")
        dec = function.child_by_field_name("declarator")
        params = dec.child_by_field_name("parameters").named_children
        if len(params) == 1 and self.text(frame, params[0]) == "void":
            params = []
        if len(params) != len(args):
            raise Unsupported("the argument count does not match the parameter count")
        env, types = {}, {}
        for param, value in zip(params, args):
            if param.type != "parameter_declaration":
                raise Unsupported("variadic parameters are unsupported")
            name, kind = self.decl(frame, param)
            if name in types:
                raise Unsupported("duplicate parameter name")
            env[name], types[name] = self.coerce(value, kind), kind
            self.record(frame, param, "parameter", name=name, value=value.number)
        self.depth += 1
        try:
            self.statement(frame, function.child_by_field_name("body"), env, types)
            value = None
        except Returned as ret:
            value = ret.value
        finally:
            self.depth -= 1
        ret_type = self.text(frame, function.child_by_field_name("type"))
        if ret_type == "void" and value is None:
            return None
        if ret_type != "int":
            raise Unsupported("this return type is unsupported")
        value = self.coerce(value, "int")
        self.record(frame, function, "return_value", value=value.number)
        return value

    def statement(self, f, n, env, types):
        self.tick()
        if n.type == "compound_statement":
            previous = set(types)
            try:
                for child in n.named_children:
                    if child.type != "comment":
                        self.statement(f, child, env, types)
            finally:
                for name in set(types) - previous:
                    types.pop(name)
                    env.pop(name, None)
        elif n.type == "declaration":
            decs = n.children_by_field_name("declarator")
            if len(decs) != 1 or decs[0].type != "init_declarator":
                raise Unsupported("only a single initialized local declaration is supported")
            dec = decs[0]
            # Reuse parameter declaration validation with an explicit declarator.
            typ = n.child_by_field_name("type")
            if self.text(f, typ) != "int" or any(c.type in {"type_qualifier", "storage_class_specifier"} for c in n.named_children):
                raise Unsupported("this local variable type is unsupported")
            name_node = dec.child_by_field_name("declarator")
            if name_node.type != "identifier":
                raise Unsupported("local pointers/arrays are unsupported")
            name = self.text(f, name_node)
            if name in types:
                raise Unsupported("variable shadowing is unsupported")
            types[name] = "int"
            env[name] = self.coerce(self.expr(f, dec.child_by_field_name("value"), env, types), "int")
            self.record(f, n, "assignment", name=name, value=env[name].number)
        elif n.type == "expression_statement":
            if n.named_children:
                self.expr(f, n.named_children[0], env, types)
        elif n.type == "if_statement":
            take = bool(self.expr(f, n.child_by_field_name("condition"), env, types).number)
            self.record(f, n.child_by_field_name("condition"), "branch", take=take)
            branch = n.child_by_field_name("consequence" if take else "alternative")
            if branch is not None:
                if branch.type == "else_clause":
                    branch = branch.named_children[0]
                self.statement(f, branch, env, types)
        elif n.type == "return_statement":
            value = self.expr(f, n.named_children[0], env, types) if n.named_children else None
            self.record(f, n, "return", value=value.number if value else None)
            raise Returned(value)
        else:
            raise Unsupported(f"unsupported statement {n.type}")

    def expr(self, f, n, env, types):
        self.tick()
        if n is None:
            raise Unsupported("missing expression")
        t = n.type
        if t == "parenthesized_expression":
            return self.expr(f, n.named_children[0], env, types)
        if t == "number_literal":
            text = self.text(f, n)
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", text):
                raise Unsupported("only decimal int constants are supported")
            return self.coerce(Value(int(text)), "int")
        if t == "identifier":
            name = self.text(f, n)
            if name not in env:
                raise Unsupported(f"undefined variable or macro {name}")
            return env[name]
        if t == "assignment_expression":
            if n.parent is None or n.parent.type != "expression_statement":
                raise Unsupported("nested assignment and evaluation order are unsupported")
            left = n.child_by_field_name("left")
            op = self.text(f, n.child_by_field_name("operator"))
            name = self.text(f, left)
            if left.type != "identifier" or name not in types or op != "=":
                raise Unsupported("complex assignment is unsupported")
            value = self.coerce(self.expr(f, n.child_by_field_name("right"), env, types), types[name])
            env[name] = value
            self.record(f, n, "assignment", name=name, value=value.number)
            return value
        if t == "call_expression":
            name_node = n.child_by_field_name("function")
            if name_node.type != "identifier":
                raise Unsupported("indirect calls are unsupported")
            name = self.text(f, name_node)
            callee = self.resolve(name, caller=f[0])
            for declaration in f[2].named_children:
                if declaration.type != "declaration":
                    continue
                dec = declaration.child_by_field_name("declarator")
                ident = dec.child_by_field_name("declarator") if dec is not None else None
                if ident is None or self.text(f, ident) != name:
                    continue
                if self.signature(f, declaration) != self.signature(callee, callee[3]):
                    raise Unsupported("the call prototype disagrees with the real definition")
                if any(c.type == "storage_class_specifier" and self.text(f, c) == "static" for c in declaration.named_children) and callee[0] != f[0]:
                    raise Unsupported("a static prototype cannot link to a definition in another file")
            arg_nodes = n.child_by_field_name("arguments").named_children
            args = [self.expr(f, a, env, types) for a in arg_nodes]
            params = callee[3].child_by_field_name("declarator").child_by_field_name("parameters").named_children
            for param, arg, arg_node in zip(params, args, arg_nodes):
                if self.text(callee, param) == "void":
                    continue
                _, kind = self.decl(callee, param)
                if kind == "ptr" and arg.kind == "int" and self.text(f, arg_node) != "0":
                    raise Unsupported("int-to-pointer conversion only accepts the literal 0; a runtime zero value is not trusted")
            self.record(f, n, "call", callee=name, callee_file=callee[0], arguments=[a.number for a in args])
            edge = f"{f[0]}:{n.start_point[0] + 1}->{callee[0]}:{name}"
            crossing = f[0] != callee[0]
            if crossing:
                args = [Value(a.number, a.kind, a.edges | {edge}) for a in args]
            result = self.call(callee, args)
            if crossing and result is not None:
                result = Value(result.number, result.kind, result.edges | {edge})
            return result
        if t == "unary_expression" or t == "pointer_expression":
            op = self.text(f, n.child_by_field_name("operator"))
            value = self.expr(f, n.child_by_field_name("argument"), env, types)
            if op == "*" and value.kind == "ptr" and value.number == 0:
                raise Fault(f[0], self.text(f, f[3].child_by_field_name("declarator").child_by_field_name("declarator")), n, "null_guard", value.edges)
            if value.kind == "int" and op in {"-", "+", "!"}:
                result = -value.number if op == "-" else value.number if op == "+" else int(not value.number)
                return self.coerce(Value(result, edges=value.edges), "int")
            if op == "!":
                return Value(int(not value.number), edges=value.edges)
            raise Unsupported("unary operations are unsupported")
        if t == "binary_expression":
            op = self.text(f, n.child_by_field_name("operator"))
            a = self.expr(f, n.child_by_field_name("left"), env, types)
            if op == "&&" and not a.number:
                return Value(0, edges=a.edges)
            if op == "||" and a.number:
                return Value(1, edges=a.edges)
            b = self.expr(f, n.child_by_field_name("right"), env, types)
            if op in {"==", "!="}:
                if a.kind != b.kind:
                    integer_node = n.child_by_field_name("left" if a.kind == "int" else "right")
                    if self.text(f, integer_node) != "0":
                        raise Unsupported("pointer comparison only supports the null pointer literal 0")
                return Value(int((a.number == b.number) if op == "==" else (a.number != b.number)), edges=a.edges | b.edges)
            if a.kind != "int" or b.kind != "int":
                raise Unsupported("pointer arithmetic is unsupported")
            x, y = a.number, b.number
            if op in {"/", "%"} and y == 0:
                name = self.text(f, f[3].child_by_field_name("declarator").child_by_field_name("declarator"))
                raise Fault(f[0], name, n, "arithmetic_bound", b.edges)
            if op in {"/", "%"} and x == -(2**31) and y == -1:
                raise Unsupported("preceding signed division overflow")
            if op == "+": result = x + y
            elif op == "-": result = x - y
            elif op == "*": result = x * y
            elif op in {"/", "%"}:
                q = (abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1)
                result = q if op == "/" else x - q * y
            elif op == "<": result = int(x < y)
            elif op == "<=": result = int(x <= y)
            elif op == ">": result = int(x > y)
            elif op == ">=": result = int(x >= y)
            elif op in {"&&", "||"}: result = int(bool(y))
            else: raise Unsupported(f"unsupported operator {op}")
            return self.coerce(Value(result, edges=a.edges | b.edges), "int")
        raise Unsupported(f"unsupported expression {t}")


def check_certificate(certificate, sources, sink, target_path, obligation_id, direction):
    """Return accepted, real evidence, reason, and a replayable audit record."""
    audit = {"engine": "cross_file_replay_v1", "accepted": False, "trace": []}
    try:
        if direction != "VIOLATED" or obligation_id not in {"arithmetic_bound", "null_guard"}:
            raise Unsupported("no interprocedural semantic rule exists for this direction or obligation")
        if not isinstance(certificate, dict) or certificate.get("version") != 1:
            raise Unsupported("missing version 1 cross-file execution certificate")
        entry = certificate.get("entry", {})
        inputs = certificate.get("inputs", [])
        if not isinstance(entry, dict) or not isinstance(inputs, list) or len(inputs) > 32 or any(type(x) is not int for x in inputs):
            raise Unsupported("the entry or the concrete inputs are malformed")
        replay = Replay(sources, sink)
        audit["trace"] = replay.trace
        audit["assumptions"] = {"entry": entry, "inputs": inputs,
                                "data_model": "signed 32-bit int; null int pointers",
                                "scope": "conditional exported C API counterexample"}
        audit["source_hashes"] = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in replay.sources.items()}
        frame = replay.resolve(str(entry.get("function", "")), entry=_path(entry.get("file", "")))
        try:
            replay.call(frame, [Value(x) for x in inputs])
        except Fault as fault:
            line = fault.node.start_point[0] + 1
            expected_kind = "arith" if obligation_id == "arithmetic_bound" else "deref"
            if (fault.path != _path(target_path) or fault.function != sink.get("func_name")
                    or line != int(sink.get("line") or 0) or fault.kind != obligation_id
                    or sink.get("sink_type") != expected_kind):
                raise Unsupported("the actual first fault disagrees with the target file, function, location or obligation")
            # A line can contain several operations. Do not let an earlier fault
            # discharge the obligation of a different expression on that line.
            owner = fault.node
            while owner.parent is not None and owner.type != "function_definition":
                owner = owner.parent
            raw = replay.sources[fault.path].encode()
            candidates = []
            stack = [owner]
            while stack:
                node = stack.pop()
                stack.extend(node.named_children)
                op = node.child_by_field_name("operator")
                token = raw[op.start_byte:op.end_byte].decode() if op else ""
                relevant = (node.type == "binary_expression" and token in {"/", "%"}) if obligation_id == "arithmetic_bound" else (node.type in {"unary_expression", "pointer_expression"} and token == "*")
                if relevant and node.start_point[0] + 1 == line:
                    candidates.append(node)
            actual = raw[fault.node.start_byte:fault.node.end_byte].decode()
            if len(candidates) != 1 or re.sub(r"\s+", "", actual) not in re.sub(r"\s+", "", str(sink.get("snippet") or "")):
                raise Unsupported("the target line has several risky operations, or the source expression is not bound to the sink")
            edges = [e for e in replay.trace if e["event"] == "call" and e["file"] != e["callee_file"]]
            if not edges or not fault.edges:
                raise Unsupported("the risky operand was not passed through an actual cross-file argument or return value")
            replay.record((fault.path, replay.sources[fault.path].encode()), fault.node,
                          "unsafe_operation", obligation_id=obligation_id,
                          transfer_edges=sorted(fault.edges))
            audit["accepted"] = True
            audit["certificate"] = certificate
            refs = [f"{e['file']}:{e['line']} {e['event']}: {e['text']}" for e in replay.trace]
            return True, refs, ("the target unsafe operation was reached by an interprocedural replay over real source; actual arguments, formals, return values and the path were all checked"), audit
        raise Unsupported("the concrete inputs did not trigger the target unsafe operation (a guard blocked it, or execution ended normally)")
    except (Unsupported, ValueError, TypeError, KeyError, AttributeError) as exc:
        audit["reason"] = str(exc)
        return False, [], str(exc), audit
