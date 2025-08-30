# Innesce-Slim-MVP-Redesign

This version of Innesce is intentionally small

Innesce (Slim) — MVP Redesign
1) Scope (only what’s real)

Front-end: .inn → tokenize → parse → AST → basic static checks (names, types, arity, simple control-flow rules).

Lowering: AST → LLVM IR.

Artifacts: object (.o/.obj) → link to native .exe (llc+lld or ORC JIT for dev).

Surface sugar: Ada-ish keywords + durations, qualifiers, enums, extended bools, extended conditionals → all lower to normal types/if/else.

Inline asm: route to LLVM InlineAsm.

Safety: strong types (no implicit widening), explicit casts only, early errors when resolvable.

Optimizations: let LLVM do it (O2/O3 pipeline).

Namespaces & “scoped indentation”: namespaces are real; indentation is a linter rule (warnings), not a parser offside rule (keeps MVP simple).

Modules/“capsules”: each capsule = compilation unit → one object file.

Permissions/Gates: compile-time annotation + tiny runtime that refuses blocked APIs unless the gate is granted.

Packetized execution: an optional runtime loop that calls exported functions per “frame” or “step”.
