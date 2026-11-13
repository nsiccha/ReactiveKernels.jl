# Comparative experiment, not a production lowering or a probability API.
# Run in a consumer environment with ReactiveKernels, SymbolicUtils 3,
# Enzyme 0.13, and BenchmarkTools. See the accompanying evidence report.
using ReactiveKernels, SymbolicUtils, BenchmarkTools, Random, Test, Printf
import Enzyme
const RK = ReactiveKernels
const SU = SymbolicUtils

SU.@syms a::Real b::Real y::Real

# Only structural scalar expressions enter this analysis. Data and its length
# enter the resulting RK graph through ordinary runtime ports.
reads(expr, variable) = occursin(variable, expr)
factors(expr) = SU.iscall(expr) && SU.operation(expr) === (*) ?
    collect(SU.arguments(expr)) : Any[expr]
terms(expr) = SU.iscall(expr) && SU.operation(expr) === (+) ?
    collect(SU.arguments(expr)) : Any[expr]
product(xs) = foldl(*, xs; init=1)

"""Discover sum_i coefficient(a,b) * basis(y_i), or decline a mixed factor.

SymbolicUtils performs the algebraic expansion. This routine only partitions
its factors by their input dependencies and groups equal data bases. It does
not implement an algebra rewrite engine, inspect a law name, or select an
alternative by cost. Expressions are straight-line scalar arithmetic; guards
and extra demanded outputs require a richer representation than this fixture.
"""
function separate(expr)
    grouped = Dict{Any,Any}()
    for term in terms(SU.expand(expr))
        data, live = Any[], Any[]
        for factor in factors(term)
            dy = reads(factor, y)
            dp = reads(factor, a) || reads(factor, b)
            dy && dp && return nothing
            push!(dy ? data : live, factor)
        end
        basis, coefficient = product(data), product(live)
        grouped[basis] = get(grouped, basis, 0) + coefficient
    end
    sort!(collect(grouped); by=p -> string(first(p)))
end

struct Literal{T}
    value::T
end
(x::Literal)() = x.value

# Compile the same symbolic expression to an ordinary RK graph. No eval, no
# custom AD rule, and no data value/extent is represented in generated source.
function emit!(graph, expr, memo)
    haskey(memo, expr) && return memo[expr]
    output_type = !SU.iscall(expr) && expr isa Number ? typeof(expr) : Float64
    output = value!(graph, Symbol(:v, length(graph.values) + 1), output_type)
    if SU.iscall(expr)
        inputs = Tuple(emit!(graph, arg, memo) for arg in SU.arguments(expr))
        add!(graph; inputs, outputs=(output,), op=SU.operation(expr))
    elseif expr isa Number
        add!(graph; inputs=(), outputs=(output,), op=Literal(expr))
    else
        error("unmapped symbolic leaf: $expr")
    end
    memo[expr] = output
end

function row_kernel(expr; data_only=false)
    graph = Graph()
    names = data_only ? [:y] : [:a, :b, :y]
    symbols = data_only ? [y] : [a, b, y]
    ports = Dict{Symbol,Value}(name => value!(graph, name, Float64) for name in names)
    memo = Dict{Any,Any}(symbol => ports[name] for (name, symbol) in zip(names, symbols))
    ports[:result] = emit!(graph, expr, memo)
    spec = KernelSpec(graph, ports, [names; :result], names, [:result])
    RK.plate(spec; have=Tuple(names), want=(:result,), batched=:y, reduce=:+)
end

struct CountRows end
(::CountRows)(data) = Float64(length(data))

function outer_graph(expr; reduced=false)
    graph = Graph()
    pa, pb = value!(graph, :a, Float64), value!(graph, :b, Float64)
    data = value!(graph, :data, Vector{Float64})
    basis = reduced ? separate(expr) : nothing
    if reduced && basis !== nothing
        memo = Dict{Any,Any}(a => pa, b => pb)
        summands = Value[]
        for (index, (term, coefficient)) in enumerate(basis)
            stat = value!(graph, Symbol(:stat, index), Float64)
            # Reuse Julia's reduction for the identity basis. A recipe-free
            # RK plate returning its HAVE port currently fails on nonempty
            # input at this pin; the evidence includes that separate boundary.
            op = isequal(term, 1) ? CountRows() :
                 isequal(term, y) ? sum : row_kernel(term; data_only=true)
            add!(graph; inputs=(data,), outputs=(stat,), op)
            c = emit!(graph, coefficient, memo)
            part = value!(graph, Symbol(:part, index), Float64)
            add!(graph; inputs=(c, stat), outputs=(part,), op=*)
            push!(summands, part)
        end
        result = value!(graph, :result, Float64)
        add!(graph; inputs=Tuple(summands), outputs=(result,), op=+)
    else
        result = value!(graph, :result, Float64)
        add!(graph; inputs=(pa, pb, data), outputs=(result,), op=row_kernel(expr))
    end
    (; graph, have=(pa, pb, data), want=(result,), data, basis)
end

bind(spec, data) = prepare(spec.graph; have=spec.have, want=spec.want,
    bound=(spec.data => data,))

# An empty reduction does not evaluate live coefficients. The runtime count
# stays outside the residual arithmetic, so even a=Inf on empty data is zero.
# This is the only extra guard this experiment supplies: it makes no claim to
# solve general floating-point admission or the preservation of authored guards.
struct EmptyGuard{K}
    kernel::K
    rows::Int
end
(k::EmptyGuard)(aa, bb) = k.rows == 0 ? 0.0 : k.kernel(aa, bb)
candidate(spec, data) = EmptyGuard(bind(spec, data), length(data))

function row_sum(row, aa, bb, data)
    result = zero(aa + bb)
    for datum in data
        result += row(aa, bb, datum)
    end
    result
end

struct Reference{F,D}
    row::F
    data::D
end
(r::Reference)(aa, bb) = row_sum(r.row, aa, bb, r.data)
gradient(k, aa, bb) = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k),
    Enzyme.Active, Enzyme.Active(aa), Enzyme.Active(bb))[1]

const CASES = (
    (name="poisson_parameter_terms", expr=y*a-exp(a),
     row=(a,b,y)->y*a-exp(a), reducible=true),
    (name="exponential_lograte", expr=a-exp(a)*y,
     row=(a,b,y)->a-exp(a)*y, reducible=true),
    (name="gamma_parameter_terms", expr=(exp(a)-1)*log(y)-exp(b)*y+exp(a)*b,
     row=(a,b,y)->(exp(a)-1)*log(y)-exp(b)*y+exp(a)*b, reducible=true),
    (name="generic_polynomial_log", expr=sin(a)*(y+1)^2+cos(b)*log(y)+a*b,
     row=(a,b,y)->sin(a)*(y+1)^2+cos(b)*log(y)+a*b, reducible=true),
    (name="mixed_exponential_control", expr=exp(a*y)+b*y,
     row=(a,b,y)->exp(a*y)+b*y, reducible=false),
)

function sample_cost(f; seconds=0.06)
    trial = @benchmark $f() seconds=seconds samples=300 evals=1
    estimate = minimum(trial)
    (; ns=estimate.time, bytes=estimate.memory, allocs=estimate.allocs)
end

function run_experiment()
    println("ENV Julia=", VERSION, " RK=", pkgversion(RK),
        " SymbolicUtils=", pkgversion(SU), " Enzyme=", pkgversion(Enzyme))
    rng = Xoshiro(20261113)
    @testset "Symbolic reduction comparative experiment" begin
        for case in CASES
            println("CASE ", case.name)
            construction = @timed (outer_graph(case.expr), outer_graph(case.expr; reduced=true))
            direct_spec, reduced_spec = construction.value
            println("BASIS ", reduced_spec.basis)
            println("CONSTRUCTION seconds=", construction.time, " bytes=", construction.bytes)
            @test (reduced_spec.basis !== nothing) == case.reducible
            ast = nothing
            fn_type = nothing
            for n in (0, 1, 32, 4096, 65536)
                data = 0.25 .+ 2 .* rand(rng, n)
                dp = @timed bind(direct_spec, data)
                cp = @timed candidate(reduced_spec, data)
                direct, reduced = dp.value, cp.value
                reference = Reference(case.row, data)
                first_value = @timed (direct(0.17, -0.21), reduced(0.17, -0.21))
                first_gradient = @timed (gradient(direct, 0.17, -0.21), gradient(reduced, 0.17, -0.21))
                for (aa, bb) in ((0.17, -0.21), (-0.6, 0.4), (0.0, 0.0))
                    truth = setprecision(BigFloat, 256) do
                        row_sum(case.row, BigFloat(aa), BigFloat(bb), BigFloat.(data))
                    end
                    @test isapprox(direct(aa, bb), truth; rtol=3e-11, atol=3e-10)
                    @test isapprox(reduced(aa, bb), truth; rtol=3e-11, atol=3e-10)
                    native_gradient = gradient(reference, aa, bb)
                    @test all(isapprox.(gradient(direct, aa, bb), native_gradient; rtol=3e-10, atol=3e-9))
                    @test all(isapprox.(gradient(reduced, aa, bb), native_gradient; rtol=3e-10, atol=3e-9))
                end
                if n == 0
                    @test reduced(Inf, Inf) === 0.0
                    @test gradient(reduced, Inf, Inf) === (0.0, 0.0)
                end
                # Same emitted residual and callable type across new values
                # and lengths. Graph construction above never sees a dataset.
                current_ast = sprint(show, RK.code_expr(reduced.kernel))
                if ast === nothing
                    ast = current_ast
                    fn_type = typeof(reduced.kernel.f)
                else
                    @test ast == current_ast
                    @test fn_type === typeof(reduced.kernel.f)
                end
                # Runtime parameter loads prevent benchmarking a residual
                # whose exp/sin/cos coefficients LLVM constant-folded from
                # scalar literals in a zero-argument benchmark closure.
                parameters = [0.17, -0.21]
                direct_cost = sample_cost(() -> direct(parameters[1], parameters[2]))
                reduced_cost = sample_cost(() -> reduced(parameters[1], parameters[2]))
                direct_grad = sample_cost(() -> gradient(direct, parameters[1], parameters[2]))
                reduced_grad = sample_cost(() -> gradient(reduced, parameters[1], parameters[2]))
                direct_bind = sample_cost(() -> bind(direct_spec, data))
                reduced_bind = sample_cost(() -> candidate(reduced_spec, data))
                saving = direct_cost.ns - reduced_cost.ns
                breakeven = saving > 0 ? max(0, (reduced_bind.ns-direct_bind.ns)/saving) : Inf
                println("RESULT case=", case.name, " n=", n,
                    " direct=", direct_cost, " reduced=", reduced_cost,
                    " direct_grad=", direct_grad, " reduced_grad=", reduced_grad,
                    " direct_bind=", direct_bind, " reduced_bind=", reduced_bind,
                    " breakeven_calls=", breakeven,
                    " direct_state_bytes=", Base.summarysize(direct.constants),
                    " reduced_state_bytes=", Base.summarysize(reduced.kernel.constants),
                    " direct_kernel_bytes=", Base.summarysize(direct),
                    " reduced_kernel_bytes=", Base.summarysize(reduced),
                    " ast_chars=", length(current_ast),
                    " first_bind_seconds=", (dp.time, cp.time),
                    " first_value_seconds=", first_value.time,
                    " first_gradient_seconds=", first_gradient.time)
            end
            # Rebinding after an in-place change must rerun the prefix, even
            # when the data object and its extent are unchanged.
            data = ones(32)
            before = candidate(reduced_spec, data)
            data .= 1.75
            after = candidate(reduced_spec, data)
            @test after(0.17, -0.21) ≈ row_sum(case.row, 0.17, -0.21, data)
            @test all(isapprox.(gradient(after, 0.17, -0.21),
                gradient(Reference(case.row, data), 0.17, -0.21); rtol=3e-10, atol=3e-9))
            @test typeof(before.kernel.f) === typeof(after.kernel.f)
        end
        # Algebraic validity does not imply safe floating-point admission.
        # These diagnostic counterexamples are evidence against enabling this
        # candidate indiscriminately; they are not weakened correctness tests.
        expr = (a*y-b)^2
        ds, rs = outer_graph(expr), outer_graph(expr; reduced=true)
        data = fill(1.0, 32)
        d, r = bind(ds, data), candidate(rs, data)
        for (aa, bb) in ((1e8, 1e8+1), (1e160, 1e160))
            truth = setprecision(BigFloat, 256) do
                sum((BigFloat(aa)*BigFloat(yy)-BigFloat(bb))^2 for yy in data)
            end
            println("ADVERSARY a=", aa, " b=", bb,
                " reference=", truth,
                " direct=", d(aa,bb), " candidate=", r(aa,bb),
                " direct_grad=", gradient(d,aa,bb), " candidate_grad=", gradient(r,aa,bb))
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_experiment()
