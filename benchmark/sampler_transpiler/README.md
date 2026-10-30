# Reactive mathematical kernel transpiler

The experimental `prepare_transpiled` interface lowers captured `@kernel`
methods to native Julia. It fixes types, array shapes, the call graph
and callable authorities during preparation. Mutation must be visible to RK.

Start with [`prepared_examples.jl`](prepared_examples.jl) for scalar and vector
consumers, then [`prepared_hmc.jl`](prepared_hmc.jl) for multinomial HMC with the
bound Eight Schools density. Neither consumer assembles compiler metadata or
reads private slots.

```julia
program = prepare_transpiled(source, initial_value;
    method=:advance!, argument=2.0, outputs=(value=:value,))
state = initial_transpiled_state(program)
result = program(state, 2.0)
result.outputs.value
next = program(result.state, result.argument)
```

`iterations` fixes the number of method calls per batch. `kernel_kwargs` binds
construction keywords. Named outputs refer to a root field (`:value`) or child
field (`(:init, :pos)`); derived fields are refreshed after the batch. State and
runtime arguments are preserved, and outputs are independent snapshots. Each
state belongs to one prepared program. Its storage and preparation metadata are
private. Reprepare after changing types, shapes, controls or fixed authorities.

Random operations use ordinary RNGs. Pass the returned argument onward to
continue its random stream. This remains an experimental interface with finite
source support.
There is one runtime argument. Loop ranges are fixed at preparation; recursive
calls, helper keyword splats and nonterminal returns inside free helpers reject.

## Run the examples

```sh
julia benchmark/sampler_transpiler/setup.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/prepared_hmc.jl
```

`runtime_keyword_probe.jl`, `partial_transfer_probe.jl`,
`indexed_array_probe.jl` and `scalar_argument_probe.jl` cover compiler
mechanisms.

## Throughput and provenance

The [public walkthrough](https://nsiccha.github.io/ReactiveKernels.jl/dev/hmc-transpiler)
shows the mathematical sources and source-linked historical numerical results.
Compare multinomial HMC with AdvancedHMC `MultinomialTS`, and endpoint HMC with
AdvancedHMC `EndPointTS`:

```sh
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/position_multinomial_scaling.jl /absolute/multinomial.csv
timeout --signal=TERM --kill-after=15s 600s julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/matched_endpoint_comparison.jl /absolute/endpoint scaling
```

The receipts in this directory (`*.toml`, `*.csv`) are historical. Their Reactant
rows were measured with ReactiveKernels' removed Reactant path; the scripts that
produced them were removed with it.

The [experiment history](HISTORY.md) preserves earlier interfaces, probes,
receipts and superseded performance checkpoints.
