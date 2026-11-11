# ReactiveKernels

ReactiveKernels prepares reusable Julia computations from the values you have to the values you want.

Install from a source checkout; this shell command resolves its pinned dependencies:

```sh
git clone https://github.com/nsiccha/ReactiveKernels.jl.git
cd ReactiveKernels.jl && julia --project=. deps/source_dependencies.jl
```

The prepared function evaluates the affine transformation at the supplied input, scale and offset:

```@example start
using ReactiveKernels

@kernel affine(x, scale, offset) = begin
    value = scale * x + offset
end
f = prepare(affine)
f(2.0, 3.0, 1.0)
```

Preparation chooses the computations needed for a requested output. Calling
`f` runs the prepared Julia function; the inputs remain ordinary runtime
values. The graph itself is pure. Optional state and buffer interfaces have
explicit ownership contracts.

[Graphs and queries](graphs.md) explains the input/output boundary.
[Bound regression](bound-regression.md) uses one graph for preparation,
prediction and residual evaluation.

For probabilistic programs, the nested package
[ReactiveKernelsPPL](rkppl.md) adds parameter declarations, transformations and
log-density queries. ReactiveKernels prepares computations; it does not choose
an inference method for you.
