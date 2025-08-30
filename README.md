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

## _____

Keywords (subset)

capsule, namespace, use, let, const, fn, return, if, elsif, else, match, end, enum, true, false, maybe, gate, asm

## _____

Durations (sugar → i64 nanoseconds)

Literals: 100_ns, 200_us, 300_ms, 4_s, 5_min, 2_h

Lowering: X_ms → duration_ms(X) → i64 (nanoseconds) via helper.

Operators: + - * / on durations produce i64 (or helper functions). MVP keeps it numeric with a type alias.

Extended bools

truth is an enum {True, False, Maybe}.

Sugar: if t? then lowers to if t == Truth.True then.

Interop: truth never implicitly converts to bool.

## _____

Type system (MVP)

Primitives: i32, i64, f32, f64, bool, truth, str.

Durations: sugar → i64 nanoseconds (helpers in runtime lib).

Enums: first-class (lower to i32 in IR).

No implicit casts: i32 → i64 requires cast<i64>(x).

Functions: first-class names; no closures in MVP.

Namespacing: namespace A::B lowers to mangled names _A_B_symbol.

## _____

Toolchain layout (C++23 + LLVM)

innesce/
  CMakeLists.txt
  src/
    main.cpp                 // CLI: innescec <file.inn> [-c|-S|--exe|--loop]
    lexer.hpp  lexer.cpp
    tokens.hpp
    parser.hpp parser.cpp
    ast.hpp
    sema.hpp   sema.cpp      // name + type checks, gate checks
    codegen.hpp codegen.cpp  // AST → LLVM IR
    diagnostics.hpp
    options.hpp
  rt/
    in_rt.hpp  in_rt.cpp     // duration helpers, fs/net stubs, gate mask, scheduler
  examples/
    hello.inn
    timer.inn
    gates.inn

## _____

Core passes (how it works)

Lexer

Tokenize identifiers, keywords, numbers, string literals, duration suffixes (_ms, _s, etc.).

Normalize durations: produce a TOK_DURATION(value, unit) token.

Parser → AST

Recursive-descent for the subset above.

Keep source spans on every node for diagnostics.

Sema (basic static checks)

Scope tables (capsule → namespace → locals).

Check: redeclaration, unknown symbol, argument count/types, return presence.

Gates: build a set of allowed gates for each capsule; if AST sees a gated API without allowance → error.

Lower “extended conditionals” to explicit branches in the AST (normalize phase).

Codegen → LLVM IR

Map types (truth/enum → i32, duration → i64).

Emit functions, control flow (if/else/match), calls into runtime (in_rt).

Inline asm via llvm::InlineAsm.

Optimize + Emit

Run PassBuilder with O2+Vectorize.

Emit object & link (lld) → .exe. (Dev mode can ORC-JIT to run immediately.)

## _____

Run with packet loop (CLI sets gate mask & loop):

innescec examples/timer.inn --exe --loop --tick=16ms --gates=file,net

## _____

