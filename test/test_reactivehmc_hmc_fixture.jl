using ReactiveKernels
using Test
using Random
import TOML

include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_hmc_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "receipts", "validate_reactivehmc_hmc.jl"))

const RHMC_HMC = ReactiveHMCHMCFixture

function _rhmc_hmc_ir_nodes(value)
    nodes = Any[value]
    children = if value isa Tuple
        value
    elseif value isa ReactiveKernels.MethodIR ||
           (parentmodule(typeof(value)) === ReactiveKernels &&
            startswith(String(nameof(typeof(value))), "_"))
        ntuple(index -> getfield(value, index), fieldcount(typeof(value)))
    else
        ()
    end
    for child in children
        append!(nodes, _rhmc_hmc_ir_nodes(child))
    end
    nodes
end

@testset "ReactiveHMC fixed-step HMC source and independent receipt" begin
    registration = ReactiveKernels.kernel_registration(RHMC_HMC.hmc_state)
    @test registration.kind == :object_kernel
    @test isempty(registration.write_roots)
    @test ReactiveKernels.kernel_port_names(RHMC_HMC.hmc_state) ==
          (:init, :n_steps, :min_dham, :step_f, :stats_f,
           :gofwd, :fwd, :dham, :diverged)
    @test map(ir -> ir.id.name, ReactiveKernels.method_irs(RHMC_HMC.hmc_state)) ==
          (:randbernoullilog, :step!)

    randbernoulli_ir, step_ir = ReactiveKernels.method_irs(RHMC_HMC.hmc_state)
    @test randbernoulli_ir.ok
    @test randbernoulli_ir.control == :branch
    @test step_ir.ok
    @test step_ir.control == :loop

    randbernoulli_nodes = _rhmc_hmc_ir_nodes(randbernoulli_ir.body)
    step_nodes = _rhmc_hmc_ir_nodes(step_ir.body)
    @test count(node -> nameof(typeof(node)) == :_OpCall,
                randbernoulli_nodes) == 0
    @test count(node -> nameof(typeof(node)) == :_OpCall, step_nodes) == 0
    @test count(node -> node isa ReactiveKernels._PlaceWrite, step_nodes) == 4
    @test count(node -> node isa ReactiveKernels._For, step_nodes) == 1
    @test count(node -> node isa ReactiveKernels._Guard, step_nodes) == 2
    @test count(node -> node isa ReactiveKernels._IfExpr, step_nodes) == 1
    @test count(node -> node isa ReactiveKernels._CallExpr, step_nodes) == 1
    @test count(node -> node isa ReactiveKernels._FieldCall, step_nodes) == 2

    randexp_calls = filter(randbernoulli_nodes) do node
        node isa ReactiveKernels._RegisteredCall &&
            getfield(getfield(node, :registration), :source) === Random.randexp
    end
    randn_calls = filter(step_nodes) do node
        node isa ReactiveKernels._RegisteredCall &&
            getfield(getfield(node, :registration), :source) === Random.randn!
    end
    @test length(randexp_calls) == 1
    @test length(randn_calls) == 1
    for call in (only(randexp_calls), only(randn_calls))
        effect = getfield(getfield(call, :registration), :primitive_effect)
        @test effect isa ReactiveKernels._PrimitiveEffect
        @test getfield(effect, :kind) == :rng
        @test getfield(effect, :order) == :ordered
    end

    loop = only(statement for statement in step_ir.body
                if statement isa ReactiveKernels._For)
    @test map(typeof, loop.body) == (
        ReactiveKernels._ExprStmt,
        ReactiveKernels._LocalAssign,
        ReactiveKernels._PlaceWrite,
        ReactiveKernels._ExprStmt,
        ReactiveKernels._Guard,
    )
    stats_call = loop.body[4].expr
    @test stats_call isa ReactiveKernels._FieldCall
    @test stats_call.path == (:stats_f,)
    @test loop.body[5].cond == ReactiveKernels._SelfField((:diverged,))

    accept_guard = last(step_ir.body)
    @test accept_guard isa ReactiveKernels._Guard
    @test accept_guard.cond isa ReactiveKernels._CallExpr
    @test accept_guard.cond.name == :randbernoullilog
    accepted_copy = only(accept_guard.body)
    @test accepted_copy isa ReactiveKernels._ExprStmt
    @test accepted_copy.expr isa ReactiveKernels._RegisteredCall
    @test getfield(accepted_copy.expr.registration, :kind) == :intrinsic
    @test getfield(accepted_copy.expr.registration, :source) ==
          ReactiveKernels.copy!!

    source_path = joinpath(@__DIR__, "..", "benchmark",
                           "reactivehmc_hmc_kernel_fixture.jl")
    source = read(source_path, String)
    @test occursin("@kernel hmc_state(", source)
    @test occursin("init.mom = sqrt(fwd.metric) * Random.randn!(rng, init.mom)", source)
    @test occursin("stats_f(__self__)", source)
    @test findfirst("stats_f(__self__)", source) < findfirst("diverged && return", source)
    @test occursin("randbernoullilog(__self__, rng, dham) && copy!!(init, fwd)", source)
    @test !occursin(r"^\s*rcopy!\("m, source)

    reference_path = joinpath(
        @__DIR__, "fixtures", "reactivehmc_ca9_hmc_reference.jl",
    )
    reference = read(reference_path, String)
    @test occursin("normal = Float32[0.3, -0.4]", reference)
    @test occursin("exponential = Float64[0.5]", reference)
    @test occursin("result === destination", reference)
    @test occursin("string(eltype(destination))", reference)
    @test occursin("string(typeof(result))", reference)
    @test occursin("float_bits(rng.exponential_returns)", reference)

    receipt_path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                            "reactivehmc-hmc-ca9-v1.toml")
    receipt = TOML.parsefile(receipt_path)
    @test isempty(validate_reactivehmc_hmc_receipt(receipt_path))
    @test receipt["pins"]["reactivehmc_revision"] ==
          ReactiveHMCAlgorithmCorpus.UPSTREAM.revision
    source_digests = Dict(ReactiveHMCAlgorithmCorpus.UPSTREAM.source_sha256)
    @test receipt["pins"]["hmc_sha256"] == source_digests["src/hmc.jl"]
end
