#!/usr/bin/env python3
# Innesce (Slim) – Python Interpreter + Optional llvmlite "AOT" JIT
# Features:
#   - Lexer + scoped-indentation Linter (warnings only)
#   - Parser + AST for: funcs, enums (with payloads), namespaces, consts
#   - Expressions: arithmetic, comparisons (== != < <= > >=), boolean (and/or/not, && || !)
#   - Arrays [ ... ], Records { k: v, ... }, indexing a[i], field access r.k
#   - Truth enum + postfix "?" sugar -> (expr == Truth::True)
#   - Full match with enum tags and destructuring, plus constants & multiple patterns
#   - Gates (file, net) with compile-time declaration + runtime enable
#   - Packetized loop: step(dt_ns)
#   - Optional llvmlite compilation for simple numeric functions (pure arithmetic/if/return)
#
# Usage examples:
#   python innesce.py examples/hello.inn
#   python innesce.py examples/gates.inn --gates=file
#   python innesce.py examples/timer.inn --loop --tick=16ms
#
#   # With explicit capsule selection
#   python innesce.py examples/enum_match.inn --capsule Demo
#
from __future__ import annotations
import sys, re, time, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC

try:
    # Optional; we compile a subset if available
    from llvmlite import ir, binding as llvm
    LLVM_OK = True
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
except Exception:
    LLVM_OK = False

# ============================
# Utilities: durations & gates
# ============================
DUR_UNITS = {"ns":1, "us":1_000, "ms":1_000_000, "s":1_000_000_000, "min":60*1_000_000_000, "h":3600*1_000_000_000}
GATE_BITS = {"file":1<<0, "net":1<<1}

def parse_tick(s: str) -> int:
    m = re.fullmatch(r"(\\d+)(ns|us|ms|s|min|h)", s)
    if not m:
        raise SystemExit(f"--tick expects e.g. 16ms, 1s, 500us; got {s}")
    return int(m.group(1)) * DUR_UNITS[m.group(2)]

# ============================
# Linter: "scoped indentation"
# ============================
def lint_scoped_indentation(src: str) -> List[str]:
    """
    Heuristic linter that warns when indentation doesn't align with block structure.
    Rules:
      - Lines ending with ' is' (capsule/namespace/fn/etc.) should be followed by a deeper indent on next non-empty line (until 'end')
      - 'end' should return to previous indent level
    Only warnings; language is not off-side.
    """
    lines = src.splitlines()
    warnings = []
    indent_stack = [0]
    need_deeper_next = False

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    for i, raw in enumerate(lines, start=1):
        s = raw.rstrip()
        if not s or s.lstrip().startswith("--"):
            continue

        ind = indent_of(raw)

        # Check 'end'
        if re.match(r"\\s*end\\s*$", s):
            if len(indent_stack) > 1 and ind != indent_stack[-2]:
                warnings.append(f"{i}: 'end' indentation {ind} doesn't match parent level {indent_stack[-2]}")
            if len(indent_stack) > 1:
                indent_stack.pop()
            need_deeper_next = False
            continue

        # If previous line opened a block (need deeper indent now)
        if need_deeper_next and ind <= indent_stack[-1]:
            warnings.append(f"{i}: expected deeper indent after '... is' block opener")
        need_deeper_next = False

        # Detect openers: 'capsule X is', 'namespace X is', 'fn X(...) is', 'if ... then', 'match ... is', 'case ... =>'
        if re.search(r"\\b(capsule|namespace|fn|match)\\b.*\\bis\\s*$", s) or re.search(r"\\bif\\b.*\\bthen\\s*$", s) or re.search(r"\\bcase\\b.*=>\\s*$", s) or re.search(r"\\bdefault\\b\\s*=>\\s*$", s):
            indent_stack.append(ind)
            need_deeper_next = True

    return warnings

# ============
# Tokenization
# ============
Token = Tuple[str, str, int, int]  # (kind, text, line, col)

KEYWORDS = {
    "capsule","namespace","enum","const","fn","return","if","elsif","else",
    "match","is","case","default","end","let","gate","allow","true","false",
    "maybe","asm","then","and","or","not"
}

# Operators & punctuation
# Note: order matters, longer first
TOKEN_SPEC = [
    ("WS",        r"[ \\t]+"),
    ("NEWLINE",   r"\\r?\\n"),
    ("COMMENT",   r"--[^\\n]*"),
    ("NUMBER",    r"\\d+(_(ns|us|ms|s|min|h))?"),
    ("STRING",    r"\\\"([^\\\"\\\\]|\\\\.)*\\\""),
    ("OP",        r"::|->|=>|:=|==|!=|<=|>=|&&|\\|\\||\\[|\\]|\\{|\\}|\\(|\\)|\\.|,|:|;|\\+|\\-|\\*|/|<|>|!|\\?"),
    ("ID",        r"[A-Za-z_][A-Za-z0-9_]*"),
]

MASTER = re.compile("|".join(f"(?P<{k}>{p})" for k,p in TOKEN_SPEC))

def lex(src: str) -> List[Token]:
    out: List[Token] = []
    line = 1
    col = 1
    i = 0
    while i < len(src):
        m = MASTER.match(src, i)
        if not m:
            raise SyntaxError(f"Lex error at line {line}:{col}: {src[i:i+20]!r}")
        kind = m.lastgroup
        text = m.group(0)
        if kind == "WS" or kind == "COMMENT":
            pass  # skip
        elif kind == "NEWLINE":
            line += 1
            col  = 0
        elif kind == "ID" and text.lower() in KEYWORDS:
            out.append( (text.lower(), text, line, col) )
        elif kind == "OP":
            out.append( (text, text, line, col) )
        else:
            out.append( (kind, text, line, col) )
        i = m.end()
        col += len(text)
    out.append( ("EOF","", line, col) )
    return out

# =====
#  AST
# =====
@dataclass
class Node(ABC): line:int=0; col:int=0

@dataclass
class Program(Node):
    capsules: List['Capsule']=field(default_factory=list)

@dataclass
class Capsule(Node):
    name:str=""; decls:List['Decl']=field(default_factory=list)
    allowed_gates:int=0

@dataclass
class Decl(Node, ABC): pass

@dataclass
class Namespace(Decl):
    name:str=""; decls:List['Decl']=field(default_factory=list)

@dataclass
class EnumVariant:
    name:str
    fields: List[Tuple[str,str]]  # list of (field_name, type_name); may be empty

@dataclass
class EnumDecl(Decl):
    name:str=""; variants:List[EnumVariant]=field(default_factory=list)

@dataclass
class ConstDecl(Decl):
    name:str=""; type_name:str=""; expr:'Expr'=None

@dataclass
class FnDecl(Decl):
    name:str=""; params:List[Tuple[str,str]]=field(default_factory=list) # (name,type)
    ret_type:Optional[str]=None; body:List['Stmt']=field(default_factory=list)

@dataclass
class GateAllow(Decl):
    gates:List[str]=field(default_factory=list)

@dataclass
class Stmt(Node, ABC): pass

@dataclass
class LetStmt(Stmt):
    name:str=""; type_name:str=""; expr:'Expr'=None

@dataclass
class AssignStmt(Stmt):
    name:str=""; expr:'Expr'=None

@dataclass
class ReturnStmt(Stmt):
    expr:Optional['Expr']=None

@dataclass
class IfStmt(Stmt):
    branches:List[Tuple['Expr', List[Stmt]]]=field(default_factory=list)
    else_body:List[Stmt]=field(default_factory=list)

@dataclass
class MatchCasePattern(Node): pass

@dataclass
class PatternConst(MatchCasePattern):
    expr:'Expr'=None

@dataclass
class PatternEnum(MatchCasePattern):
    enum_name:str=""; variant:str=""; bind_names:List[str]=field(default_factory=list)

@dataclass
class MatchStmt(Stmt):
    target:'Expr'=None
    cases:List[Tuple[List[MatchCasePattern], List[Stmt]]]=field(default_factory=list)
    default:List[Stmt]=field(default_factory=list)

@dataclass
class ExprStmt(Stmt):
    expr:'Expr'=None

@dataclass
class Expr(Node, ABC): pass

@dataclass
class Name(Expr): ident:str=""

@dataclass
class Number(Expr): value:int=0

@dataclass
class String(Expr): value:str=""

@dataclass
class Call(Expr): name:str=""; args:List[Expr]=field(default_factory=list)

@dataclass
class CallQualified(Expr): enum_name:str=""; variant:str=""; args:List[Expr]=field(default_factory=list)

@dataclass
class BinOp(Expr): op:str=""; lhs:Expr=None; rhs:Expr=None

@dataclass
class UnaryOp(Expr): op:str=""; expr:Expr=None

@dataclass
class ArrayLit(Expr): items:List[Expr]=field(default_factory=list)

@dataclass
class RecordLit(Expr): items:List[Tuple[str,Expr]]=field(default_factory=list)

@dataclass
class IndexExpr(Expr): base:Expr=None; index:Expr=None

@dataclass
class FieldExpr(Expr): base:Expr=None; field:str=""

@dataclass
class TruthTest(Expr): expr:Expr=None  # postfix '?' sugar

# =========
#  Parser
# =========
class Parser:
    def __init__(self, toks: List[Token]):
        self.toks = toks; self.i = 0

    def cur(self) -> Token: return self.toks[self.i]
    def accept(self, kind_or_text: str) -> Optional[Token]:
        k,t,_,_ = self.cur()
        if k == kind_or_text or t == kind_or_text:
            self.i+=1; return (k,t,0,0)
        return None
    def expect(self, kind_or_text: str) -> Token:
        tok = self.accept(kind_or_text)
        if tok: return tok
        k,t,l,c = self.cur()
        raise SyntaxError(f"Expected {kind_or_text} at {l}:{c}, got {k}:{t}")

    def parse(self) -> Program:
        capsules=[]
        while self.cur()[0] != "EOF":
            capsules.append(self.capsule_decl())
        return Program(capsules=capsules)

    def capsule_decl(self) -> Capsule:
        self.expect("capsule"); name = self.expect("ID")[1]
        self.expect("is")
        decls=[]
        allowed=0
        while self.cur()[0] not in ("end","EOF"):
            k,_,_,_ = self.cur()
            if k=="namespace": decls.append(self.namespace_decl())
            elif k=="enum":    decls.append(self.enum_decl())
            elif k=="const":   decls.append(self.const_decl())
            elif k=="fn":      decls.append(self.fn_decl())
            elif k=="gate":    allowed |= self.gate_allow_decl()
            else:
                raise SyntaxError(f"Unexpected in capsule: {self.cur()}")
        self.expect("end")
        return Capsule(name=name, decls=decls, allowed_gates=allowed)

    def namespace_decl(self) -> Namespace:
        self.expect("namespace"); name=self.expect("ID")[1]; self.expect("is")
        decls=[]
        while self.cur()[0]!="end":
            k,_,_,_ = self.cur()
            if k=="enum":  decls.append(self.enum_decl())
            elif k=="const":decls.append(self.const_decl())
            elif k=="fn":  decls.append(self.fn_decl())
            else: raise SyntaxError("Only enum/const/fn inside namespace")
        self.expect("end"); return Namespace(name=name, decls=decls)

    def enum_decl(self) -> EnumDecl:
        # enum Option is
        #   None,
        #   Some(value:i32)
        # end
        self.expect("enum"); name=self.expect("ID")[1]; self.expect("is")
        variants: List[EnumVariant] = []
        while True:
            vname = self.expect("ID")[1]
            fields: List[Tuple[str,str]] = []
            if self.accept("("):
                if not self.accept(")"):
                    while True:
                        fname = self.expect("ID")[1]; self.expect(":"); ftype = self.type_name()
                        fields.append( (fname, ftype) )
                        if self.accept(")"): break
                        self.expect(",")
            variants.append(EnumVariant(vname, fields))
            if self.accept(","): continue
            break
        self.expect("end")
        return EnumDecl(name=name, variants=variants)

    def const_decl(self) -> ConstDecl:
        self.expect("const"); name=self.expect("ID")[1]
        self.expect(":"); ty=self.type_name(); self.expect(":=")
        expr=self.expr(); self.expect(";")
        return ConstDecl(name=name, type_name=ty, expr=expr)

    def type_name(self)->str:
        k,t,_,_=self.cur()
        if k in ("ID","i32","i64","f32","f64","bool","truth","str"):
            self.i+=1; return t
        raise SyntaxError(f"Type name expected, got {self.cur()}")

    def fn_decl(self)->FnDecl:
        self.expect("fn"); name=self.expect("ID")[1]
        self.expect("(")
        params=[]
        if not self.accept(")"):
            while True:
                pnam=self.expect("ID")[1]; self.expect(":"); pty=self.type_name()
                params.append((pnam,pty))
                if self.accept(")"): break
                self.expect(",")
        ret=None
        if self.accept("->"):
            ret=self.type_name()
        self.expect("is")
        body=self.block_stmts()
        self.expect("end")
        return FnDecl(name=name, params=params, ret_type=ret, body=body)

    def gate_allow_decl(self)->int:
        self.expect("gate"); self.expect("allow"); self.expect("(")
        bits=0
        name = self.expect("ID")[1]
        bits |= GATE_BITS.get(name,0)
        while self.accept(","):
            name=self.expect("ID")[1]
            bits |= GATE_BITS.get(name,0)
        self.expect(")"); self.expect(";")
        return bits

    def block_stmts(self)->List[Stmt]:
        out=[]
        while self.cur()[0] not in ("end","else","default") and not (self.cur()[0]=="EOF"):
            out.append(self.stmt())
        return out

    def stmt(self)->Stmt:
        k,_,_,_=self.cur()
        if k=="let": return self.let_stmt()
        if k=="return": return self.return_stmt()
        if k=="if": return self.if_stmt()
        if k=="match": return self.match_stmt()
        # assign or expr
        if self.peek_assign(): return self.assign_stmt()
        e=self.expr(); self.expect(";"); return ExprStmt(expr=e)

    def let_stmt(self)->LetStmt:
        self.expect("let"); name=self.expect("ID")[1]; self.expect(":"); ty=self.type_name()
        self.expect(":="); e=self.expr(); self.expect(";")
        return LetStmt(name=name, type_name=ty, expr=e)

    def peek_assign(self)->bool:
        if self.cur()[0]!="ID": return False
        j=self.i+1
        return j<len(self.toks) and self.toks[j][1]==":="

    def assign_stmt(self)->AssignStmt:
        name=self.expect("ID")[1]; self.expect(":="); e=self.expr(); self.expect(";")
        return AssignStmt(name=name, expr=e)

    def return_stmt(self)->ReturnStmt:
        self.expect("return")
        if self.cur()[0] != ";":
            e=self.expr(); self.expect(";"); return ReturnStmt(expr=e)
        self.expect(";"); return ReturnStmt()

    def if_stmt(self)->IfStmt:
        self.expect("if"); cond=self.expr(); self.expect("then")
        branches=[ (cond, self.block_stmts()) ]
        while self.accept("elsif"):
            c=self.expr(); self.expect("then")
            branches.append( (c, self.block_stmts()) )
        else_body=[]
        if self.accept("else"):
            else_body=self.block_stmts()
        self.expect("end")
        return IfStmt(branches=branches, else_body=else_body)

    def match_stmt(self)->MatchStmt:
        self.expect("match"); tgt=self.expr(); self.expect("is")
        cases=[]
        default=[]
        while True:
            if self.accept("default"):
                self.expect("=>"); default=self.block_stmts(); break
            if self.cur()[0]=="end": break
            self.expect("case")
            pats=[ self.case_pattern() ]
            while self.accept(","):
                pats.append(self.case_pattern())
            self.expect("=>")
            body=self.block_stmts()
            cases.append( (pats, body) )
        self.expect("end")
        return MatchStmt(target=tgt, cases=cases, default=default)

    def case_pattern(self)->MatchCasePattern:
        # Enum::Variant(x,y) | constant expr
        if self.cur()[0]=="ID":
            # lookahead for Enum::Variant
            enum_name = self.cur()[1]
            j = self.i+1
            if j < len(self.toks) and self.toks[j][0]=="::":
                # qualified
                self.i += 2
                variant = self.expect("ID")[1]
                bind_names=[]
                if self.accept("("):
                    if not self.accept(")"):
                        while True:
                            bind_names.append(self.expect("ID")[1])
                            if self.accept(")"): break
                            self.expect(",")
                return PatternEnum(enum_name, variant, bind_names)
        # else treat as constant
        e = self.expr_simple()
        return PatternConst(e)

    # ------- Expressions with precedence -------
    # or (||) / and (&&) / not (!)
    def expr(self)->Expr:
        return self.parse_or()

    def parse_or(self)->Expr:
        node = self.parse_and()
        while True:
            if self.accept("or") or self.accept("||"):
                node = BinOp("or", node, self.parse_and())
            else:
                break
        return node

    def parse_and(self)->Expr:
        node = self.parse_cmp()
        while True:
            if self.accept("and") or self.accept("&&"):
                node = BinOp("and", node, self.parse_cmp())
            else:
                break
        return node

    def parse_cmp(self)->Expr:
        node = self.parse_addsub()
        while True:
            if   self.accept("=="): node = BinOp("==", node, self.parse_addsub())
            elif self.accept("!="): node = BinOp("!=", node, self.parse_addsub())
            elif self.accept("<="): node = BinOp("<=", node, self.parse_addsub())
            elif self.accept(">="): node = BinOp(">=", node, self.parse_addsub())
            elif self.accept("<"):  node = BinOp("<",  node, self.parse_addsub())
            elif self.accept(">"):  node = BinOp(">",  node, self.parse_addsub())
            else: break
        return node

    def parse_addsub(self)->Expr:
        node=self.parse_muldiv()
        while True:
            if self.accept("+"): node=BinOp("+", node, self.parse_muldiv())
            elif self.accept("-"): node=BinOp("-", node, self.parse_muldiv())
            else: break
        return node

    def parse_muldiv(self)->Expr:
        node=self.parse_unary()
        while True:
            if self.accept("*"): node=BinOp("*", node, self.parse_unary())
            elif self.accept("/"): node=BinOp("/", node, self.parse_unary())
            else: break
        return node

    def parse_unary(self)->Expr:
        if self.accept("not") or self.accept("!"):
            return UnaryOp("not", self.parse_unary())
        if self.accept("-"):
            return UnaryOp("neg", self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self)->Expr:
        node = self.atom()
        while True:
            if self.accept("["):
                idx = self.expr()
                self.expect("]")
                node = IndexExpr(node, idx)
            elif self.accept("."):
                field = self.expect("ID")[1]
                node = FieldExpr(node, field)
            elif self.accept("?"):
                node = TruthTest(node)
            else:
                break
        return node

    def expr_simple(self)->Expr:
        # used for constant pattern; no commas
        return self.parse_or()

    def atom(self)->Expr:
        k,t,l,c = self.cur()
        if k=="NUMBER":
            self.i+=1
            if "_" in t:  # duration like 16_ms
                v,unit = t.split("_")
                return Number(value=int(v)*DUR_UNITS[unit])
            return Number(value=int(t))
        if k=="STRING":
            self.i+=1
            s = bytes(t[1:-1], "utf-8").decode("unicode_escape")
            return String(value=s)
        if t=="(":
            self.i+=1; e=self.expr(); self.expect(")"); return e
        if t=="[":
            self.i+=1
            items: List[Expr] = []
            if not self.accept("]"):
                while True:
                    items.append(self.expr())
                    if self.accept("]"): break
                    self.expect(",")
            return ArrayLit(items)
        if t=="{":
            self.i+=1
            pairs: List[Tuple[str,Expr]] = []
            if not self.accept("}"):
                while True:
                    key = self.expect("ID")[1]
                    self.expect(":")
                    val = self.expr()
                    pairs.append( (key,val) )
                    if self.accept("}"): break
                    self.expect(",")
            return RecordLit(pairs)
        if k=="ID":
            # qualified call: Enum::Variant(args)
            name = t
            j = self.i+1
            if j < len(self.toks) and self.toks[j][0]=="::":
                # Could be Enum::Variant ctor OR qualified name
                self.i += 2
                variant = self.expect("ID")[1]
                if self.accept("("):
                    args=[]
                    if not self.accept(")"):
                        while True:
                            args.append(self.expr())
                            if self.accept(")"): break
                            self.expect(",")
                    return CallQualified(enum_name=name, variant=variant, args=args)
                # No call -> treat as qualified name token backtrack? For simplicity, require call.
                return Name(ident=f"{name}::{variant}")
            # normal call or name
            self.i+=0  # already at ID
            self.i+=1
            if self.accept("("):
                args=[]
                if not self.accept(")"):
                    while True:
                        args.append(self.expr())
                        if self.accept(")"): break
                        self.expect(",")
                return Call(name=name, args=args)
            return Name(ident=name)
        raise SyntaxError(f"Bad expression at {l}:{c} near {self.cur()}")

# =============================
# Semantic types / environments
# =============================
class Type:
    def __init__(self,name:str): self.name=name
    def __repr__(self): return self.name

I32=Type("i32"); I64=Type("i64"); F32=Type("f32"); F64=Type("f64")
BOOL=Type("bool"); TRUTH=Type("truth"); STR=Type("str")
PRIMS = {"i32":I32,"i64":I64,"f32":F32,"f64":F64,"bool":BOOL,"truth":TRUTH,"str":STR}

@dataclass
class FnValue:
    decl: FnDecl
    capsule: 'CapsuleEnv'

@dataclass
class CapsuleEnv:
    name:str
    gates:int
    consts:Dict[str, Tuple[Type, Any]] = field(default_factory=dict)
    enums:Dict[str, Dict[str, EnumVariant]] = field(default_factory=dict) # EnumName -> VariantName->EnumVariant
    fns:  Dict[str, FnValue] = field(default_factory=dict)

class RuntimeErrorEx(Exception): pass
class ReturnSignal(Exception):
    def __init__(self,val): self.val=val

# ==========
# Interpreter
# ==========
class Interpreter:
    def __init__(self, prog: Program, enabled_gates:int):
        self.prog = prog
        self.enabled_gates = enabled_gates
        self.capsules: Dict[str,CapsuleEnv] = {}
        self._enum_type_aliases = set()

        # llvmlite state
        self._llvm_module = None
        self._llvm_engine = None
        self._compiled_fns = {}  # name -> callable

    # ---- Build environments & static checks ----
    def prepare(self):
        # First pass: create capsule envs and register enums/fns (+ flatten namespaces)
        for cap in self.prog.capsules:
            env=CapsuleEnv(cap.name, gates=cap.allowed_gates)
            self.capsules[cap.name]=env
            for d in cap.decls:
                if isinstance(d, EnumDecl):
                    # Map enum -> variant map
                    env.enums[d.name] = { ev.name: ev for ev in d.variants }
                elif isinstance(d, FnDecl):
                    env.fns[d.name]=FnValue(d, env)
                elif isinstance(d, Namespace):
                    for nd in d.decls:
                        if isinstance(nd, EnumDecl):
                            env.enums[f"{d.name}::{nd.name}"] = { ev.name: ev for ev in nd.variants }
                        elif isinstance(nd, FnDecl):
                            env.fns[f"{d.name}::{nd.name}"]=FnValue(nd, env)
                        elif isinstance(nd, ConstDecl):
                            pass
                elif isinstance(d, ConstDecl):
                    pass

        # Build enum "type aliases" (so consts can use enum names as types → i32 runtime)
        for env in self.capsules.values():
            for en in env.enums.keys():
                self._enum_type_aliases.add(en)

        # Second pass: evaluate consts (both top-level and namespaced)
        for cap in self.prog.capsules:
            env=self.capsules[cap.name]
            # top-level
            for d in cap.decls:
                if isinstance(d, ConstDecl):
                    ty=self.type_of_name(d.type_name)
                    if ty is None: raise RuntimeErrorEx(f"Unknown type '{d.type_name}' for const {d.name}")
                    val=self.eval_expr(d.expr, env, {})
                    self._check_const_type(d.name, ty, val)
                    env.consts[d.name]=(ty,val)
            # namespaced
            for d in cap.decls:
                if isinstance(d, Namespace):
                    for nd in d.decls:
                        if isinstance(nd, ConstDecl):
                            ty=self.type_of_name(nd.type_name)
                            if ty is None: raise RuntimeErrorEx(f"Unknown type '{nd.type_name}' for const {d.name}::{nd.name}")
                            val=self.eval_expr(nd.expr, env, {})
                            self._check_const_type(f"{d.name}::{nd.name}", ty, val)
                            env.consts[f"{d.name}::{nd.name}"]=(ty,val)

        # Optional: compile all functions with llvmlite if possible (only simple numeric subset)
        if LLVM_OK:
            self._compile_all_numeric_functions()

    def _check_const_type(self, name:str, ty:Type, val:Any):
        if ty in (I32, I64) and not isinstance(val, int):
            raise RuntimeErrorEx(f"const {name} expects integer")
        if ty is STR and not isinstance(val, str):
            raise RuntimeErrorEx(f"const {name} expects string")

    def type_of_name(self, n:str)->Optional[Type]:
        if n in self._enum_type_aliases: return I32
        return PRIMS.get(n, None)

    # ---- Runtime helpers (gated stdlib) ----
    def gate_check(self, needed:int, capsule:CapsuleEnv, opname:str):
        if (capsule.gates & needed)==0:
            raise RuntimeErrorEx(f"E3001: '{opname}' requires gate; add 'gate allow(...)' in capsule '{capsule.name}'")
        if (self.enabled_gates & needed)==0:
            raise RuntimeErrorEx(f"E3002: runtime policy denies '{opname}' (enable with --gates=...)")

    def std_fs_write(self, capsule:CapsuleEnv, path:str, data:str)->int:
        self.gate_check(GATE_BITS["file"], capsule, "fs_write")
        with open(path,"ab") as f:
            f.write(data.encode("utf-8"))
        return 0

    def std_net_send(self, capsule:CapsuleEnv, host:str, port:int, data:str)->int:
        self.gate_check(GATE_BITS["net"], capsule, "net_send")
        return 0

    # ---- Evaluate expressions ----
    def eval_expr(self, e:Expr, capsule:CapsuleEnv, locals:Dict[str, Any])->Any:
        if isinstance(e, Number): return e.value
        if isinstance(e, String): return e.value
        if isinstance(e, Name):
            if e.ident in locals: return locals[e.ident]
            if e.ident in capsule.consts: return capsule.consts[e.ident][1]
            if "::" in e.ident and e.ident in capsule.consts:
                return capsule.consts[e.ident][1]
            # enums Truth built-in
            if e.ident in ("true","false","maybe"):
                return {"true":1,"false":0,"maybe":2}[e.ident]
            # allow referencing enum names or variants? return None
            return self._lookup_enum_value_by_item(capsule, e.ident)
        if isinstance(e, TruthTest):
            val = self.eval_expr(e.expr, capsule, locals)
            return 1 if val == 1 else 0
        if isinstance(e, UnaryOp):
            v = self.eval_expr(e.expr, capsule, locals)
            if e.op=="not": return 0 if self._truthy(v)==1 else 1
            if e.op=="neg": return -v
            raise RuntimeErrorEx(f"Unsupported unary {e.op}")
        if isinstance(e, BinOp):
            l=self.eval_expr(e.lhs, capsule, locals)
            # short-circuit
            if e.op=="and":
                return 1 if (self._truthy(l)==1 and self._truthy(self.eval_expr(e.rhs,capsule,locals))==1) else 0
            if e.op=="or":
                return 1 if (self._truthy(l)==1 or self._truthy(self.eval_expr(e.rhs,capsule,locals))==1) else 0
            r=self.eval_expr(e.rhs, capsule, locals)
            if e.op=="+": return l+r
            if e.op=="-": return l-r
            if e.op=="*": return l*r
            if e.op=="/": return l//r if isinstance(l,int) and isinstance(r,int) else l/r
            if e.op=="==": return 1 if l==r else 0
            if e.op=="!=": return 1 if l!=r else 0
            if e.op=="<":  return 1 if l< r else 0
            if e.op==">":  return 1 if l> r else 0
            if e.op=="<=": return 1 if l<=r else 0
            if e.op==">=": return 1 if l>=r else 0
            raise RuntimeErrorEx(f"Unsupported op {e.op}")
        if isinstance(e, ArrayLit):
            return [ self.eval_expr(it, capsule, locals) for it in e.items ]
        if isinstance(e, RecordLit):
            return { k:self.eval_expr(v,capsule,locals) for (k,v) in e.items }
        if isinstance(e, IndexExpr):
            base = self.eval_expr(e.base, capsule, locals)
            idx  = self.eval_expr(e.index, capsule, locals)
            return base[idx]
        if isinstance(e, FieldExpr):
            base = self.eval_expr(e.base, capsule, locals)
            return base[e.field]
        if isinstance(e, CallQualified):
            # Enum::Variant(...) constructor -> runtime tag
            return self._construct_enum_value(capsule, e.enum_name, e.variant,
                                              [self.eval_expr(a,capsule,locals) for a in e.args])
        if isinstance(e, Call):
            # stdlib
            if e.name=="fs_write":
                path=self.eval_expr(e.args[0], capsule, locals)
                data=self.eval_expr(e.args[1], capsule, locals)
                return self.std_fs_write(capsule, path, data)
            if e.name=="net_send":
                host=self.eval_expr(e.args[0], capsule, locals)
                port=int(self.eval_expr(e.args[1], capsule, locals))
                data=self.eval_expr(e.args[2], capsule, locals)
                return self.std_net_send(capsule, host, port, data)
            if e.name=="print":
                vals=[self.eval_expr(a,capsule,locals) for a in e.args]
                print(*vals, end="")
                return 0
            if e.name=="println":
                vals=[self.eval_expr(a,capsule,locals) for a in e.args]
                print(*vals)
                return 0
            # user functions (maybe compiled)
            fn = capsule.fns.get(e.name)
            if not fn:
                raise RuntimeErrorEx(f"Unknown function '{e.name}'")
            # compiled callable?
            compiled = self._compiled_fns.get((capsule.name, e.name))
            if compiled is not None:
                # prepare numeric args
                arg_vals = [self.eval_expr(a,capsule,locals) for a in e.args]
                return int(compiled(*arg_vals))
            return self.call_fn(fn, capsule, e.args, locals)
        raise RuntimeErrorEx(f"Unsupported expr {e}")

    def _truthy(self, v: Any) -> int:
        # truth enum (1 true, 0 false, 2 maybe) + Python truth for others
        if isinstance(v, int):
            if v in (0,1,2): return 1 if v==1 else 0
            return 1 if v!=0 else 0
        if isinstance(v, (list, dict, str)):
            return 1 if len(v)>0 else 0
        return 1 if bool(v) else 0

    def _lookup_enum_value_by_item(self, capsule:CapsuleEnv, ident:str):
        # unqualified item name lookup (if unique); returns tag code (int)
        found=None
        for en, variants in capsule.enums.items():
            if ident in variants and len(variants[ident].fields)==0:
                code = list(variants.keys()).index(ident)
                if found is None:
                    found = code
                else:
                    raise RuntimeErrorEx(f"Ambiguous enum item '{ident}'")
        if found is not None: return found
        raise RuntimeErrorEx(f"Unknown name '{ident}'")

    def _construct_enum_value(self, capsule:CapsuleEnv, enum_name:str, variant:str, args:List[Any]):
        if enum_name not in capsule.enums or variant not in capsule.enums[enum_name]:
            raise RuntimeErrorEx(f"Unknown enum ctor {enum_name}::{variant}")
        ev = capsule.enums[enum_name][variant]
        if len(ev.fields) != len(args):
            raise RuntimeErrorEx(f"Arity mismatch for {enum_name}::{variant}")
        return {"_enum": enum_name, "_tag": variant, "_fields": args}

    # ---- Function calls & statements ----
    def call_fn(self, fnv:FnValue, capsule:CapsuleEnv, arg_exprs:List[Expr], caller_locals:Dict[str,Any])->Any:
        decl = fnv.decl
        if len(decl.params)!=len(arg_exprs):
            raise RuntimeErrorEx(f"Arity mismatch calling {decl.name}")
        locals: Dict[str,Any] = {}
        for (pname,pty), a in zip(decl.params, arg_exprs):
            val=self.eval_expr(a, capsule, caller_locals)
            # simple type checks
            if pty in ("i32","i64") and not isinstance(val,int):
                raise RuntimeErrorEx(f"Type mismatch: param {pname} expects int")
            if pty=="str" and not isinstance(val,str):
                raise RuntimeErrorEx(f"Type mismatch: param {pname} expects str")
            locals[pname]=val
        try:
            for s in decl.body:
                self.exec_stmt(s, capsule, locals)
        except ReturnSignal as rs:
            return rs.val
        return 0

    def exec_stmt(self, s:Stmt, capsule:CapsuleEnv, locals:Dict[str,Any]):
        if isinstance(s, LetStmt):
            v=self.eval_expr(s.expr, capsule, locals)
            if s.type_name in ("i32","i64") and not isinstance(v,int):
                raise RuntimeErrorEx(f"Type mismatch: let {s.name}:{s.type_name}")
            if s.type_name=="str" and not isinstance(v,str):
                raise RuntimeErrorEx(f"Type mismatch: let {s.name}:str")
            locals[s.name]=v; return
        if isinstance(s, AssignStmt):
            if s.name not in locals: raise RuntimeErrorEx(f"Unknown local '{s.name}'")
            locals[s.name]=self.eval_expr(s.expr, capsule, locals); return
        if isinstance(s, ReturnStmt):
            val=self.eval_expr(s.expr,capsule,locals) if s.expr else 0
            raise ReturnSignal(val)
        if isinstance(s, ExprStmt):
            _ = self.eval_expr(s.expr, capsule, locals); return
        if isinstance(s, IfStmt):
            for cond, body in s.branches:
                cv=self.eval_expr(cond, capsule, locals)
                if self._truthy(cv)==1:
                    for st in body: self.exec_stmt(st,capsule,locals)
                    return
            for st in s.else_body: self.exec_stmt(st,capsule,locals)
            return
        if isinstance(s, MatchStmt):
            tv=self.eval_expr(s.target, capsule, locals)
            matched=False
            for patterns, body in s.cases:
                for pat in patterns:
                    if self._match_pattern(pat, tv, capsule, locals):
                        for st in body: self.exec_stmt(st,capsule,locals)
                        matched=True; break
                if matched: break
            if not matched:
                for st in s.default: self.exec_stmt(st,capsule,locals)
            return
        raise RuntimeErrorEx(f"Unsupported stmt {s}")

    def _match_pattern(self, pat:MatchCasePattern, val:Any, capsule:CapsuleEnv, locals:Dict[str,Any])->bool:
        if isinstance(pat, PatternConst):
            pv = self.eval_expr(pat.expr, capsule, locals)
            return pv == val
        if isinstance(pat, PatternEnum):
            if isinstance(val, dict) and val.get("_enum")==pat.enum_name and val.get("_tag")==pat.variant:
                fields = val.get("_fields", [])
                if len(pat.bind_names) != len(fields):
                    return False
                for name, v in zip(pat.bind_names, fields):
                    locals[name] = v
                return True
            return False
        return False

    # ---- Optional llvmlite compilation ----
    def _compile_all_numeric_functions(self):
        # Build module/engine
        self._llvm_module = ir.Module(name="innesce_module")
        target = llvm.Target.from_default_triple()
        tm = target.create_target_machine()
        self._llvm_engine = llvm.create_mcjit_compiler(llvm.parse_assembly(str(self._llvm_module)), tm)

        # We compile simple numeric functions only: integer args/returns, numeric exprs/if/return
        for cap in self.capsules.values():
            for name, fnv in cap.fns.items():
                if self._is_llvm_eligible(fnv.decl):
                    try:
                        self._compile_fn(cap, fnv.decl)
                    except Exception:
                        # Ignore failures, fall back to interpreter
                        pass
        # Finalize & get ptrs
        llvm_mod = llvm.parse_assembly(str(self._llvm_module))
        llvm_mod.verify()
        self._llvm_engine.add_module(llvm_mod)
        self._llvm_engine.finalize_object()
        # Map symbols to Python callables (ctypes-like by address)
        for cap in self.capsules.values():
            for name, fnv in cap.fns.items():
                key = (cap.name, name)
                if key in self._compiled_fns:  # already installed
                    continue
                sym = self._llvm_engine.get_function_address(f"{cap.name}_{name}")
                if sym == 0: continue
                # Build a Python callable via ctypes.CFUNCTYPE with int signature up to 6 args
                import ctypes
                argc = len(fnv.decl.params)
                if argc>6: continue
                CFN = ctypes.CFUNCTYPE(ctypes.c_int64, *([ctypes.c_int64]*argc))
                self._compiled_fns[key] = CFN(sym)

    def _is_llvm_eligible(self, decl:FnDecl)->bool:
        # Only integer params/returns, and body contains only let/assign/return/if/exprStmt with + - * / comparisons and calls to other eligible fns (not implemented fully)
        ints = {"i32","i64"}
        if decl.ret_type and decl.ret_type not in ints: return False
        for _,ty in decl.params:
            if ty not in ints: return False
        # very shallow check
        return True

    def _compile_fn(self, cap:CapsuleEnv, decl:FnDecl):
        # Build function type
        i64 = ir.IntType(64)
        fn_type = ir.FunctionType(i64, [i64]*len(decl.params))
        fn = ir.Function(self._llvm_module, fn_type, name=f"{cap.name}_{decl.name}")
        block = fn.append_basic_block("entry")
        builder = ir.IRBuilder(block)

        # Locals map to alloca
        locals = {}
        for i, (pname, _) in enumerate(decl.params):
            ptr = builder.alloca(i64, name=pname)
            builder.store(fn.args[i], ptr)
            locals[pname] = ptr

        def emit_expr(e:Expr) -> ir.Value:
            if isinstance(e, Number):
                return i64(e.value)
            if isinstance(e, Name):
                if e.ident in locals:
                    return builder.load(locals[e.ident])
                # constants unsupported in compiled subset
                raise RuntimeError("Non-local name in compiled fn")
            if isinstance(e, BinOp):
                a = emit_expr(e.lhs); b = emit_expr(e.rhs)
                if e.op=="+": return builder.add(a,b)
                if e.op=="-": return builder.sub(a,b)
                if e.op=="*": return builder.mul(a,b)
                if e.op=="/": return builder.sdiv(a,b)
                if e.op in ("==","!=","<",">","<=",">="):
                    cmp_map = {
                        "==":"==","!=":"!=","<":"<",">":">","<=":"<=",">=":">="
                    }
                    pred = cmp_map[e.op]
                    res = builder.icmp_signed(pred, a, b)
                    return builder.zext(res, i64)
                raise RuntimeError(f"Unsupported op {e.op} in compiled fn")
            if isinstance(e, UnaryOp) and e.op=="neg":
                v = emit_expr(e.expr); return builder.sub(i64(0), v)
            raise RuntimeError(f"Unsupported expr in compiled fn: {e}")

        def emit_stmt_list(stmts:List[Stmt]):
            for s in stmts:
                if isinstance(s, LetStmt):
                    v = emit_expr(s.expr); ptr = builder.alloca(i64, name=s.name); builder.store(v, ptr); locals[s.name]=ptr
                elif isinstance(s, AssignStmt):
                    if s.name not in locals: raise RuntimeError("Assign to unknown local in compiled fn")
                    v = emit_expr(s.expr); builder.store(v, locals[s.name])
                elif isinstance(s, ExprStmt):
                    _ = emit_expr(s.expr)
                elif isinstance(s, ReturnStmt):
                    v = emit_expr(s.expr) if s.expr else i64(0); builder.ret(v); return True
                elif isinstance(s, IfStmt):
                    # compile as simple if-else chain with truthy test (non-zero)
                    after_bb = fn.append_basic_block("ifend")
                    def emit_branch(branches, else_body):
                        if not branches:
                            emit_stmt_list(else_body)
                            builder.branch(after_bb)
                            return
                        cond, body = branches[0]
                        cv = emit_expr(cond)
                        cmp0 = builder.icmp_signed("!=", cv, i64(0))
                        then_bb = fn.append_basic_block("then")
                        else_bb = fn.append_basic_block("else")
                        builder.cbranch(cmp0, then_bb, else_bb)
                        builder.position_at_end(then_bb)
                        if emit_stmt_list(body): return True
                        builder.branch(after_bb)
                        builder.position_at_end(else_bb)
                        return emit_branch(branches[1:], else_body)
                    if emit_branch(s.branches, s.else_body):
                        return True
                    builder.position_at_end(after_bb)
                else:
                    raise RuntimeError(f"Unsupported stmt in compiled fn: {s}")
            return False

        finished = emit_stmt_list(decl.body)
        if not finished:
            builder.ret(i64(0))

        # keep track for symbol resolution later
        self._compiled_fns[(cap.name, decl.name)] = None  # placeholder

# ==========
#   Driver
# ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help=".inn source")
    ap.add_argument("--capsule", default=None, help="capsule to run (defaults to first)")
    ap.add_argument("--gates", default="", help="comma list: file,net")
    ap.add_argument("--loop", action="store_true", help="run packetized loop (calls step(dt))")
    ap.add_argument("--tick", default="16ms", help="loop tick (e.g. 16ms, 1s)")
    ap.add_argument("--lint", action="store_true", help="run indentation linter and show warnings")
    args = ap.parse_args()

    src = open(args.file,"r",encoding="utf-8").read()

    if args.lint:
        warns = lint_scoped_indentation(src)
        if warns:
            print("Linter warnings:")
            for w in warns:
                print("  -", w)
            print()

    toks = lex(src)
    prog = Parser(toks).parse()

    enabled=0
    if args.gates:
        for g in args.gates.split(","):
            g=g.strip()
            if not g: continue
            if g not in GATE_BITS: raise SystemExit(f"Unknown gate '{g}'")
            enabled |= GATE_BITS[g]

    intr = Interpreter(prog, enabled_gates=enabled)
    intr.prepare()

    capname = args.capsule or (prog.capsules[0].name if prog.capsules else None)
    if not capname: raise SystemExit("No capsules found.")

    if args.loop:
        tick_ns = parse_tick(args.tick)
        rc = intr.run_loop(capname, tick_ns)
    else:
        rc = intr.run_capsule_main(capname)

    sys.exit(int(rc))

# Entry points for Interpreter
def run_capsule_main(self, capname:str)->int:
    cap=self.capsules.get(capname)
    if not cap: raise RuntimeErrorEx(f"Capsule '{capname}' not found")
    if "main" not in cap.fns:
        return 0
    return int(self.call_fn(cap.fns["main"], cap, [], {}))

def run_loop(self, capname:str, tick_ns:int)->int:
    cap=self.capsules.get(capname)
    if not cap: raise RuntimeErrorEx(f"Capsule '{capname}' not found")
    if "step" not in cap.fns:
        raise RuntimeErrorEx("Loop mode requires fn step(dt:i64)")
    step_fn = cap.fns["step"]
    last = time.perf_counter_ns()
    while True:
        now  = time.perf_counter_ns()
        dt   = now - last
        last = now
        rc = int(self.call_fn(step_fn, cap, [Number(value=tick_ns)], {}))
        if rc!=0: return rc
        if tick_ns>0:
            tts = max(0, (tick_ns - (time.perf_counter_ns()-now))/1e9)
            if tts>0: time.sleep(tts)

Interpreter.run_capsule_main = run_capsule_main
Interpreter.run_loop = run_loop

if __name__=="__main__":
    main()
