# Reactive mathematical kernel transpiler

The experimental `prepare_transpiled` interface lowers captured `@kernel`
methods to native Julia or Reactant. It fixes types, array shapes, the call graph
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

Load Reactant and pass `backend=:reactant` to compile the same source. Random
operations require `Reactant.ReactantRNG`; native Julia uses ordinary RNGs.
Pass the returned argument onward to continue its random stream. Device outputs
can be materialized with `Array` or a scalar conversion. The whole batch executes
in one synchronized call. Default compilation uses the measured 64 KiB CPU loop
policy. This remains an experimental interface with finite source support.
There is one runtime argument. Loop ranges are fixed at preparation; recursive
calls, helper keyword splats and nonterminal returns inside free helpers reject.

## Run the examples

```sh
julia benchmark/sampler_transpiler/setup.jl
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/prepared_hmc.jl native
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/prepared_hmc.jl reactant
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/prepared_interface_probe.jl
```

`prepared_interface_probe.jl` exercises input/output ownership, continuation,
argument type/shape rejection, zero work, derived outputs and real HMC RNGs.
The existing `loop_control_probe.jl`, `runtime_keyword_probe.jl`,
`partial_transfer_probe.jl` and `range_draw_probe.jl` cover compiler mechanisms.

## Throughput and provenance

The [public walkthrough](https://nsiccha.github.io/ReactiveKernels.jl/dev/hmc-transpiler)
shows the mathematical sources and source-linked numerical results. Compare
multinomial HMC with AdvancedHMC `MultinomialTS`, and endpoint HMC with
AdvancedHMC `EndPointTS` and Reactant ProbProg `:HMC`.

```sh
julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/position_multinomial_scaling.jl /absolute/multinomial.csv
timeout --signal=TERM --kill-after=15s 600s julia --project=benchmark/sampler_transpiler benchmark/sampler_transpiler/matched_endpoint_comparison.jl /absolute/endpoint scaling
```

Those historical producers measure the compiler directly with final-position/RNG
output; the prepared interface additionally returns reusable state and independent
output snapshots. Its focused same-program comparison is recorded in
[`prepared-interface-v2.toml`](prepared-interface-v2.toml), with every sample in
the adjacent CSV. Reproduce it with
`prepared_interface_benchmark.jl /absolute/interface.csv` in this environment.
Preparation and compilation stay separate from warm execution, all replicates
are retained, and these fixed-work measurements omit adaptation and history.
No matching numerical trajectories or RNG streams are required across backends.

The [experiment history](HISTORY.md) preserves earlier interfaces, probes,
receipts and superseded performance checkpoints. NUTS/WALNUTS and user-added
online statistics are parked follow-ups to this consolidation.
