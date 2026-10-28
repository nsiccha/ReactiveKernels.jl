using Reactant

# A numerical leaf with an invalid logarithm in the inactive arm. The
# authored lazy branch must survive both native AD and compiled execution.
function _sc_lazy_value(b)
    Reactant.@trace if b > 0
        value = log(b)
    else
        value = log(-b)
    end
    return value
end
@rkppl _sc_lazy() = begin
    b ~ Normal(0, 1)
    return _sc_lazy_value(b)
end
const _SC_LAZY = quote
    z ~ _sc_lazy()
    q ~ Normal(z, 1)
    y .~ Normal.(q, 1)
end
function _sc_lazy_oracle(nt, data)
    b = nt.z.b
    value = b > 0 ? log(b) : log(-b)
    return logpdf(Normal(), b) + logpdf(Normal(value, 1), nt.q) +
        _sc_ll(nt.q, data.y)
end

# These models read both local paths and ordinary return values. Numerical
# references are independent Distributions.jl expressions from the native
# acceptance file, rather than parity against a second compiler path alone.
function _sc_compiled_measure(kind, ast, data)
    bound, built, u = _sc_build(ast, data)
    reference(v) = kind === :lazy ?
        _sc_lazy_oracle(constrain(built.layout, v), data) :
        _sc_oracle(kind, constrain(built.layout, v), data)
    q = prepare_sampler(built, bound, u; backend = _SC_BACKEND)
    ru = Reactant.to_rarray(u)
    return Base.invokelatest(_sc_compile_raw, kind, q, ru, reference, u)
end

# Enter the latest world after building the model. The barrier belongs
# around compilation, outside the traced mathematical call.
function _sc_compile_raw(kind, q, ru, reference, u)
    post = q.kernel
    g = similar(u)
    native_value, _ = sampler_value_and_gradient!(q, g, u)
    @test native_value ≈ reference(u) rtol = 1e-12
    @test g ≈ _sc_fd(reference, u) rtol = 1e-5 atol = 1e-7
    hlo = repr(Reactant.@code_hlo optimize = false post(ru))
    operations = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith)\.\w+", hlo)
        operations[m.match] = get(operations, m.match, 0) + 1
    end
    @test !isempty(operations)
    compiled = Reactant.@compile post(ru)
    @test Float64(compiled(ru)) ≈ reference(u) rtol = 1e-9
    cad = compile_ad_value_and_gradient(q.ad, ru)
    value, gradient = cad(ru)
    @test Float64(value) ≈ reference(u) rtol = 1e-9
    @test Array(gradient) ≈ _sc_fd(reference, u) rtol = 1e-5 atol = 1e-7
    if kind === :lazy
        @test occursin("stablehlo.if", hlo)
        bi = findfirst(==(Symbol("z.b")), coordinate_names(q.layout))
        for sign in (-1, 1)
            point = copy(u); point[bi] = sign * 0.4
            rp = Reactant.to_rarray(point)
            @test Float64(compiled(rp)) ≈ reference(point) rtol = 1e-9
            pv, pg = cad(rp)
            @test Float64(pv) ≈ reference(point) rtol = 1e-9
            @test Array(pg) ≈ _sc_fd(reference, point) rtol = 1e-5 atol = 1e-7
            nv, _ = sampler_value_and_gradient!(q, g, point)
            @test nv ≈ reference(point) rtol = 1e-12
            @test g ≈ _sc_fd(reference, point) rtol = 1e-5 atol = 1e-7
        end
    end
    return operations
end

@testset "scoped submodels: compiled values, gradients and structural growth" begin
    for (kind, ast) in ((:collision, _SC_COLLISION), (:nested, _SC_NESTED),
            (:record, _SC_RECORD), (:cell, _SC_CELL), (:lazy, _SC_LAZY),
            (:array_record, _SC_ARRAY_RECORD))
        @testset "$kind" begin
            counts = Dict{String,Int}[]
            for n in (3, 8)
                x = collect(range(-0.7, 0.9; length = n))
                data = (; x, y = sin.(x), y2 = cos.(x), X = hcat(ones(n), x))
                push!(counts, Base.invokelatest(_sc_compiled_measure, kind, ast, data))
            end
            @test counts[1] == counts[2]
        end
    end
end
