# Active scalar seed allocation across a retained reverse call

`repro_enzyme_active_seed_box.jl` compares two adapters for the same owned
custom rule. Both lease and return the same vector tape, compute the same
gradient, and receive a runtime scalar seed. One reverse adapter is inlined;
the other keeps its call boundary with `@noinline`. The script imports only
Enzyme and Julia standard libraries.

On Julia 1.13.1, Enzyme `d37eb44840cc3dd7f6fc013c57e04867110964de`, and its
native library `f25c881a0e56cfb39aabf818a4c31d5d27c3ca72` (LLVM 20), the initial
diagnostic passed eight changed-point value/gradient checks. Three warm calls
allocated zero bytes for the inline case and 48 bytes for the retained case.
Allocation profiling attributed those three objects to `Active{Float64}`.
The original profile also boxed its single scalar result; the preserved
script discards that result inside the profile expression.

Run in that pinned environment:

```sh
julia +1.13.1 --startup-file=no --project=<env> benchmark/repro_enzyme_active_seed_box.jl
```

For the actual differentiated LLVM (not the Julia wrapper around Enzyme's
compiled function pointer), append `--dump-llvm`. `SEED_BOX_IR_DIR` selects
the output directory; it defaults to Julia's temporary directory.

On that diagnostic revision, the public grouped-model consumer allocates one 16-byte
`EnzymeCore.Active{Float64}` object per call when its reusable tape is used.
This reproducer establishes that a retained reverse call can preserve the
same allocation. It does not yet establish that the entire public model has
exactly the same compiler cause. Enzyme's `src/rules/customrules.jl` at the
pin above constructs the active return annotation with `emit_allocobj!`
(line 2216). The exact preserved script also passed all eight checks. Its
specialized profile contains no allocation in the inline case and exactly
three 16-byte `Active{Float64}` objects for three retained calls. The
differentiated LLVM places `activeRT.Active{Float64}`'s `ijl_gc_small_alloc`
immediately before the retained reverse call. Whether that allocation
survives optimization depends on the call context. There is no source-math,
runtime-activity, or production inlining workaround in this diagnostic.

Enzyme repair `custom_reverse_seed`, landed in dogfood
`e7b6383e4d3132d7d97c1ac9fca432fca319b28c`, passes the immutable seed through
caller stack storage under the existing readonly, noncapturing ABI. Check it
with `Enzyme.has_dogfood_repair(:custom_reverse_seed)`. The upstream report is
[Enzyme.jl #4062](https://github.com/EnzymeAD/Enzyme.jl/issues/4062). The profile
above remains the historical baseline; it does not describe repaired Enzyme.
