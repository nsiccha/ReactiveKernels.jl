# Standalone native-backend reproducer; requires only Enzyme and stdlib Test.
# Run with Julia 1.10.12 / Enzyme 0.13.209. Untyped collection passes a
# generator closure containing both active and constant Float64 arrays through
# Base.collect. Enzyme rejects the constant pointer store into that closure.
# Related upstream report: https://github.com/EnzymeAD/Enzyme.jl/issues/2386
module NativeGeneratorCaptureRepro
using Enzyme, Test

struct PublicPrepared{S}
    state::S
end

ordinary_reader(b, x, idx) =
    [b[idx[i], 1] + b[idx[i], 2] * x[i] for i in eachindex(idx)]

native_reader(t, b, f) =
    [b[f.idx[i], 1] + b[f.idx[i], 2] * t.state.x[i] for i in eachindex(f.idx)]

function prepared_reader(b, inputs)
    t = only(inputs)
    native_reader(t, b, only(t.state.fields))
end

# These are ordinary primal controls, not AD rules or activity overrides.
typed_reader(b, x, idx) =
    eltype(b)[b[idx[i], 1] + b[idx[i], 2] * x[i] for i in eachindex(idx)]

function inline_collect_reader(b, x, idx)
    values = (b[idx[i], 1] + b[idx[i], 2] * x[i] for i in eachindex(idx))
    @inline collect(values)
end

function loop_reader(b, x, idx)
    out = similar(b, length(idx))
    for i in eachindex(idx)
        out[i] = b[idx[i], 1] + b[idx[i], 2] * x[i]
    end
    out
end

objective(reader, b, x, idx) = sum(reader(b, x, idx))
prepared_objective(b, inputs) = sum(prepared_reader(b, inputs))

function reverse_outcome(reader, b, x, idx)
    db = zero(b)
    try
        Enzyme.autodiff(Enzyme.Reverse, objective, Enzyme.Active,
            Enzyme.Const(reader), Enzyme.Duplicated(b, db),
            Enzyme.Const(x), Enzyme.Const(idx))
        db
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        println("UNSUPPORTED ", reader, ": ", nameof(typeof(err)))
        nothing
    end
end

function prepared_reverse_outcome(b, inputs)
    db = zero(b)
    try
        Enzyme.autodiff(Enzyme.Reverse, prepared_objective, Enzyme.Active,
            Enzyme.Duplicated(b, db), Enzyme.Const(inputs))
        db
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        println("UNSUPPORTED prepared_reader: ", nameof(typeof(err)))
        nothing
    end
end

function run_reproducer()
    println("Julia=", VERSION, " Enzyme=", pkgversion(Enzyme))
    @testset "native generator constant-array capture" begin
        for T in (Float32, Float64), n in (0, 1, 4, 17)
            b = reshape(T[-0.2, 0.1, 0.3, -0.1], 2, 2)
            x = T[(i - 2) / 4 for i in 1:n]
            idx = Int[mod1(i, 2) for i in 1:n]
            inputs = [PublicPrepared((; x, fields = ((; idx),)))]
            snapshot = deepcopy((b, x, idx, only(inputs).state))
            @test isequal((b, x, idx, only(inputs).state), snapshot)
            expected = zero(b)
            for i in eachindex(idx)
                expected[idx[i], 1] += one(T)
                expected[idx[i], 2] += x[i]
            end
            reference = loop_reader(b, x, idx)
            @test ordinary_reader(b, x, idx) == reference
            @test prepared_reader(b, inputs) == reference
            for reader in (typed_reader, inline_collect_reader, loop_reader)
                @test reader(b, x, idx) == reference
                for _ in 1:2
                    @test reverse_outcome(reader, b, x, idx) ≈ expected
                    @test isequal((b, x, idx, only(inputs).state), snapshot)
                end
            end
            # Capability gaps, not refusals. An upstream lift is an unexpected
            # pass here, prompting replacement with ordinary passing checks.
            @test_broken reverse_outcome(ordinary_reader, b, x, idx) ≈ expected
            @test_broken prepared_reverse_outcome(b, inputs) ≈ expected
            @test isequal((b, x, idx, only(inputs).state), snapshot)
        end
    end
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    NativeGeneratorCaptureRepro.run_reproducer()
end
