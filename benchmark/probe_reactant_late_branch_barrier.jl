# Compiler-policy experiment, NOT an installed repair or a required strict mode.
# Add barriers only after ordinary AD and MLIR optimization, then invoke the
# same default XLA compiler. This uses private Reactant 0.2.290 compiler APIs.
include("repro_reactant_pure_lazy_guard.jl")
using ReactiveKernels
const LazyIR = Reactant.MLIR.IR

function protect_lazy_returns!(op)
    protected = 0
    for region in op, block in region
        for child in collect(block)
            protected += protect_lazy_returns!(child)
        end
        if LazyIR.name(op) == "stablehlo.if"
            ret = LazyIR.terminator(block)
            values = LazyIR.operands(ret)
            isempty(values) && continue
            barrier = LazyIR.create_operation_common("stablehlo.optimization_barrier",
                LazyIR.Location(); results=LazyIR.type.(values), operands=values,
                result_inference=false)
            LazyIR.insert_before!(block, ret, barrier)
            for i in eachindex(values)
                LazyIR.setoperand!(ret, i, LazyIR.result(barrier, i))
            end
            protected += 1
        end
    end
    return protected
end

function late_branch_prototype(thunk, output, label)
    LazyIR.@dispose ctx = Reactant.ReactantContext() begin
        LazyIR.activate(ctx)
        try
            mod = parse(LazyIR.Module, thunk.module_string)
            try
                @test protect_lazy_returns!(LazyIR.Operation(mod)) > 0
                @test LazyIR.verifyall(mod)
                protected_mlir = repr(mod)
                write(joinpath(output, "$label.protected.mlir"), protected_mlir)
                e = thunk.exec
                options = Reactant.XLA.make_compile_options(
                    device_id=Int64(Reactant.XLA.device_ordinal(thunk.device)))
                executable = Reactant.XLA.compile(thunk.client, mod;
                    compile_options=options, num_outputs=e.num_outputs,
                    num_parameters=e.num_parameters, is_sharded=e.is_sharded,
                    num_replicas=e.num_replicas, num_partitions=e.num_partitions)
                # Keep the serialized module consistent with the new executable.
                return typeof(thunk)(thunk.f, executable, thunk.device,
                    protected_mlir, thunk.client, thunk.global_device_ids,
                    thunk.donated_args_mask, thunk.compiled_with_sync)
            finally
                LazyIR.dispose(mod)
            end
        finally
            LazyIR.deactivate(ctx)
        end
    end
end

function nested_lazy_guard(x, scale)
    value = zero(x)
    Reactant.@trace if scale > 0
        if scale > 1
            value = log(scale) + x / scale
        else
            value = sqrt(scale) + x / scale
        end
    else
        value = -2 * one(x)
    end
    return value
end

@kernel prototype_shared_guard(x::Vector{Float64}, scale::Float64) = begin
    pointwise = plate(x, scale) do xi, si
        cell::Float64 = si > 0 ? log(si) + xi / si : -2.0
        cell
    end
    total::Float64 = sum(pointwise)
end

function check_late_branch_prototype(fn, initial, output, label; conditionals=1)
    gradient(args...) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(fn), args...)
    args = map(lazy_traced, initial)
    inventories = []
    for (direction, f) in (("primal", fn), ("reverse", gradient))
        tag = "$label-$direction"
        for (stage, optimize) in (("raw", false), ("default", true))
            write(joinpath(output, "$tag.$stage.mlir"),
                repr(Reactant.@code_hlo optimize=optimize f(args...)))
        end
        original = Reactant.@compile serializable=true f(args...)
        protected = late_branch_prototype(original, output, tag)
        pair = []
        for (stage, compiled) in (("default", original), ("protected", protected))
            hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            write(joinpath(output, "$tag.$stage.executable.hlo"), hlo)
            counts = lazy_executable_inventory(hlo)
            println(tag, " ", stage, " executable inventory: ", counts)
            push!(pair, counts)
        end
        @test get(pair[2], "conditional", 0) >= conditionals
        # The stock executable is a semantic control, not a failed strict-mode
        # assertion. Only the experiment promises the extra branch regions.
        @test !isempty(pair[1])
        for scale in (2.0, 0.7, -1.0, 0.0, NaN)
            host_args = (initial[1], typeof(initial[2])(scale))
            traced_args = map(lazy_traced, host_args)
            expected = f(host_args...)
            for compiled in (original, protected)
                actual = lazy_host(compiled(traced_args...))
                @test all(isapprox.(actual, expected))
                @test isequal(map(lazy_host, traced_args), host_args)
            end
        end
        push!(inventories, pair)
    end
    return inventories
end

function check_late_branch_prototype()
    output = get(ENV, "REACTANT_LAZY_GUARD_OUTPUT", mktempdir())
    mkpath(output)
    println("Julia=", VERSION, " Reactant=", pkgversion(Reactant),
        " Enzyme=", pkgversion(Enzyme), " RK=", pkgversion(ReactiveKernels))
    println("Complete modules: ", output)
    @testset "late branch protection experiment" begin
        for T in (Float32, Float64)
            check_late_branch_prototype(pure_lazy_guard, (T(0.5), T(2)),
                output, "backend-$T")
            check_late_branch_prototype(nested_lazy_guard, (T(0.5), T(2)),
                output, "nested-$T"; conditionals=2)
        end
        inventories = []
        for n in (3, 7, 11)
            x = collect(range(0.2, 1.0; length=n))
            fn = prepare(prototype_shared_guard; want=:total)
            push!(inventories, check_late_branch_prototype(
                fn, (x, 2.0), output, "shared-$n"))
        end
        @test allequal(inventories)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_late_branch_prototype()
end
