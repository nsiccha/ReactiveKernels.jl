# The partial-evaluation pre-pass: data-only subgraphs run once per
# preparation instead of on every call. One general Plan -> Plan transform
# (`partial_evaluation`), consumed by `prepare`/`prepare_ad`/compile backends
# through the opt-in `bound` kwarg; without the kwarg behavior is unchanged.
using ReactiveKernels
using Test
using DifferentiationInterface
import Enzyme

const PE_TEST_AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Positional ops have no global names in lowered code; assert structurally.
function pe_op_call_indices(ast)
    indices = Int[]
    function visit(node)
        node isa Expr || return
        if node.head === :call && node.args[1] isa Expr
            callee = node.args[1]
            if callee.head === :ref && length(callee.args) == 2 &&
               callee.args[1] === :__ops__ && callee.args[2] isa Int
                push!(indices, callee.args[2])
            end
        end
        foreach(visit, node.args)
    end
    visit(ast)
    indices
end

pe_allocated(k, x) = (k(x); @allocated k(x))

pe_contains_broadcast(ast) = begin
    found = Ref(false)
    function visit(node)
        node === GlobalRef(Base, :broadcast) && (found[] = true)
        node isa Expr && foreach(visit, node.args)
    end
    visit(ast)
    found[]
end

@testset "partial evaluation" begin
    @testset "data-only subgraph runs once per preparation" begin
        data_calls = Ref(0)
        data_op = d -> (data_calls[] += 1; 2 .* d)

        g = Graph()
        x = value!(g, :x, Float64)
        d = value!(g, :d, Vector{Float64})
        a = value!(g, :a, Vector{Float64})
        s = value!(g, :s, Float64)
        r = value!(g, :r, Float64)
        add!(g; inputs = (d,), outputs = (a,), op = data_op)
        add!(g; inputs = (a,), outputs = (s,), op = sum)
        add!(g; inputs = (x, s), outputs = (r,), op = *)

        dval = [1.0, 2.0, 3.0]
        plain = prepare(g; have = (x, d), want = (r,))
        expected2, expected3 = plain(2.0, dval), plain(3.0, dval)

        data_calls[] = 0
        bound_kernel = prepare(g; have = (x, d), want = (r,),
                               bound = (d => dval,))
        @test data_calls[] == 1                    # prefix ran at prepare time
        @test bound_kernel isa PreparedKernel
        @test bound_kernel(2.0) == expected2
        @test bound_kernel(3.0) == expected3
        @test data_calls[] == 1                    # and never per call
        @test Tuple(v.name for v in inputs(bound_kernel)) == (:x,)
        @test_throws MethodError bound_kernel(2.0, dval)

        # The residual body contains the constant and the residual multiply,
        # and none of the hoisted data ops.
        residual_ops = pe_op_call_indices(code_expr(bound_kernel))
        @test length(residual_ops) == 2

        # Warmed up above; the residual call allocates nothing. (Measure
        # through a function barrier so the non-const test binding does not
        # count against the kernel.)
        @test pe_allocated(bound_kernel, 2.0) == 0

        # The general pass is a plain Plan -> Plan transform.
        p = plan(g; have = (x, d), want = (r,))
        p2 = partial_evaluation(p, (d,), (dval,))
        @test p2 isa Plan
        @test Tuple(v.name for v in p2.have) == (:x,)
        @test prepare(p2)(2.0) == expected2

        # A single pair works unwrapped, and prepare(::Plan) accepts bound.
        @test prepare(p; bound = d => dval)(2.0) == expected2
    end

    @testset "bound ports consumed directly by residual recipes" begin
        g = Graph()
        x = value!(g, :x, Vector{Float64})
        d = value!(g, :d, Vector{Float64})
        r = value!(g, :r, Float64)
        add!(g; inputs = (x, d), outputs = (r,), op = (x, d) -> sum(x .* d))

        dval = [2.0, 3.0]
        bound_kernel = prepare(g; have = (x, d), want = (r,),
                               bound = (d => dval,))
        @test bound_kernel([1.0, 1.0]) == 5.0
        @test Tuple(v.name for v in inputs(bound_kernel)) == (:x,)

        # Array compiler backends can keep the q-only public kernel while
        # passing the large bound array as a hidden device operand. The
        # stripped call captures no array-valued bound constant and is exactly
        # equivalent when supplied the extracted value.
        external_call, external_values =
            ReactiveKernels._externalize_bound_arrays(bound_kernel)
        @test external_values == (dval,)
        @test external_call([1.0, 1.0], external_values...) == 5.0
        @test all(external_call.ops) do op
            !(op isa ReactiveKernels._BoundConstant &&
              op.value isa AbstractArray)
        end

        plain = prepare(g; have = (x, d), want = (r,))
        unchanged, no_values =
            ReactiveKernels._externalize_bound_arrays(plain)
        @test unchanged === plain
        @test isempty(no_values)
    end

    @testset "data-only WANTs become constants" begin
        g = Graph()
        x = value!(g, :x, Float64)
        d = value!(g, :d, Vector{Float64})
        s = value!(g, :s, Float64)
        r = value!(g, :r, Float64)
        add!(g; inputs = (d,), outputs = (s,), op = sum)
        add!(g; inputs = (x, s), outputs = (r,), op = +)

        dval = [1.0, 2.0]
        bound_kernel = prepare(g; have = (x, d), want = (r, s, d),
                               bound = (d => dval,))
        rv, sv, dv = bound_kernel(10.0)
        @test rv == 13.0
        @test sv == 3.0
        @test dv === dval          # the bound port itself passes through
    end

    @testset "collateral multi-output stays with its residual owner" begin
        g = Graph()
        x = value!(g, :x, Float64)
        d = value!(g, :d, Float64)
        u = value!(g, :u, Float64)
        v = value!(g, :v, Float64)
        w = value!(g, :w, Float64)
        rv = value!(g, :rv, Float64)
        rw = value!(g, :rw, Float64)
        # R1 (parameter-dependent, first in execution order) owns u and v.
        add!(g; inputs = (x,), outputs = (u, v), op = x -> (x + 1.0, x + 2.0))
        # R2 (data-only) collaterally emits v but owns only w.
        add!(g; inputs = (d,), outputs = (v, w), op = d -> (d + 10.0, d + 20.0))
        add!(g; inputs = (v,), outputs = (rv,), op = identity)
        add!(g; inputs = (w,), outputs = (rw,), op = identity)

        plain = prepare(g; have = (x, d), want = (u, rv, rw))
        bound_kernel = prepare(g; have = (x, d), want = (u, rv, rw),
                               bound = (d => 5.0,))
        @test bound_kernel(1.0) == plain(1.0, 5.0)
        # v's authoritative producer is residual R1: rv follows x, not d.
        @test bound_kernel(2.0)[2] == 4.0
    end

    @testset "structural CSE aliases resolve to bind-time constants" begin
        g = Graph()
        x = value!(g, :x, Float64)
        d = value!(g, :d, Vector{Float64})
        s1 = value!(g, :s1, Float64)
        s2 = value!(g, :s2, Float64)
        r = value!(g, :r, Float64)
        add!(g; inputs = (d,), outputs = (s1,), op = sum, cse_key = :sum_d)
        add!(g; inputs = (d,), outputs = (s2,), op = sum, cse_key = :sum_d)
        add!(g; inputs = (x, s2), outputs = (r,), op = *)

        dval = [1.0, 2.0, 3.0]
        bound_kernel = prepare(g; have = (x, d), want = (r,),
                               bound = (d => dval,))
        @test bound_kernel(2.0) == 12.0
    end

    @testset "zero-input recipes hoist under the bare pass" begin
        nullary_calls = Ref(0)
        g = Graph()
        x = value!(g, :x, Vector{Float64})
        c = value!(g, :c, Vector{Float64})
        r = value!(g, :r, Float64)
        add!(g; inputs = (), outputs = (c,),
             op = () -> (nullary_calls[] += 1; [1.0, 2.0]))
        add!(g; inputs = (x, c), outputs = (r,), op = (x, c) -> sum(x .* c))

        p = plan(g; have = (x,), want = (r,))
        p2 = partial_evaluation(p, (), ())
        k = prepare(p2)
        @test nullary_calls[] == 1
        @test k([3.0, 4.0]) == 11.0
        @test k([1.0, 1.0]) == 3.0
        @test nullary_calls[] == 1

        # The opt-in kwarg default runs no pass at all: the nullary recipe
        # stays per-call, byte-identically to today.
        k_plain = prepare(g; have = (x,), want = (r,))
        nullary_calls[] = 0
        k_plain([1.0, 1.0]); k_plain([1.0, 1.0])
        @test nullary_calls[] == 2
    end

    @testset "validation" begin
        g = Graph()
        x = value!(g, :x, Float64)
        d = value!(g, :d, Float64)
        e = value!(g, :e, Float64)
        r = value!(g, :r, Float64)
        add!(g; inputs = (x, d), outputs = (r,), op = +)
        p = plan(g; have = (x, d), want = (r,))

        @test_throws ArgumentError partial_evaluation(p, (e,), (1.0,))
        @test_throws ArgumentError partial_evaluation(p, (d, d), (1.0, 1.0))
        @test_throws ArgumentError partial_evaluation(p, (d,), (1.0, 2.0))
        @test_throws ArgumentError prepare(p; bound = (d, 1.0))
    end

    @testset "authored kernels: data-only plates hoist wholesale" begin
        spec = @kernel plate_model(x::Float64, d::Vector{Float64}) = begin
            doubled = plate(d) do element
                2.0 * element
            end
            total::Float64 = sum(doubled)
            result::Float64 = x + total
        end

        dval = [1.0, 2.0, 3.0]
        plain = prepare(spec; have = (:x, :d), want = :result)
        bound_kernel = prepare(spec; have = (:x, :d), want = :result,
                               bound = (; d = dval))
        @test bound_kernel(5.0) == plain(5.0, dval)
        @test Tuple(v.name for v in inputs(bound_kernel)) == (:x,)
        # The plate (and its fused sum) ran at prepare time: the residual body
        # has no broadcast left, only the constant fetch and the final +.
        residual = code_expr(bound_kernel)
        @test !pe_contains_broadcast(residual)
        @test length(pe_op_call_indices(residual)) == 2

        # Parameter-dependent plates (broadcast-style arguments) stay
        # per-call and stay correct.
        mixed = @kernel mixed_model(x::Float64, d::Vector{Float64}) = begin
            scaled = plate(d, x) do element, scale
                scale * element
            end
            total::Float64 = sum(scaled)
        end
        mixed_plain = prepare(mixed; have = (:x, :d), want = :total)
        mixed_bound = prepare(mixed; have = (:x, :d), want = :total,
                              bound = (; d = dval))
        @test mixed_bound(2.0) == mixed_plain(2.0, dval)

        @test_throws ArgumentError prepare(spec; have = (:x, :d),
                                           want = :result,
                                           bound = ((:d => dval),))
        # Planning over an insufficient HAVE boundary fails before the pass
        # even sees the bound values.
        @test_throws PlanningError prepare(spec; have = (:x,),
                                           want = :result, bound = (; d = dval))
        # A bound name that is not an authored HAVE port is rejected.
        @test_throws ArgumentError prepare(spec; have = (:x, :d),
                                           want = :result,
                                           bound = (; result = 1.0))
    end

    @testset "prepare_ad consumes the same pass" begin
        spec = @kernel ad_model(q::Vector{Float64}, data::Vector{Float64}) = begin
            shifted = data .- 1.0
            density::Float64 = sum(q .* shifted) - 0.5 * sum(abs2, q)
        end

        dval = [2.0, -1.0, 0.5]
        qv = [0.3, -0.4, 0.2]
        expected = (dval .- 1.0) .- qv

        plain_grad = ad_gradient(spec, PE_TEST_AD_BACKEND, qv, dval;
                                 active = :q, want = :density)
        @test plain_grad ≈ expected

        bound_grad = ad_gradient(spec, PE_TEST_AD_BACKEND, qv;
                                 active = :q, want = :density,
                                 bound = (; data = dval))
        @test bound_grad ≈ expected

        prepared = prepare_ad(spec, PE_TEST_AD_BACKEND, qv;
                              active = :q, want = :density,
                              bound = (; data = dval))
        @test prepared isa PreparedADKernel
        @test Tuple(v.name for v in inputs(prepared.kernel)) == (:q,)
        q2 = [1.0, 2.0, -3.0]
        @test ad_gradient(prepared, q2) ≈ (dval .- 1.0) .- q2

        value, gradient = ad_value_and_gradient!(
            prepared, similar(q2), q2)
        @test gradient ≈ (dval .- 1.0) .- q2
        @test value ≈ sum(q2 .* (dval .- 1.0)) - 0.5 * sum(abs2, q2)

        @test_throws ArgumentError prepare_ad(
            spec, PE_TEST_AD_BACKEND, qv;
            active = :q, want = :density, bound = (; data = dval),
            data = dval)

        # A bound preparation intentionally bypasses authored signature
        # conveniences. Defaulted ports that remain in HAVE are therefore
        # explicit positional exemplars and call arguments, as documented.
        defaults = @kernel bound_defaults(
                q::Vector{Float64}, scale::Float64 = 1.25;
                data::Vector{Float64}, offset::Float64 = 0.0) = begin
            density::Float64 =
                sum(q .* data) - scale * sum(abs2, q) + offset
        end
        default_bound = prepare_ad(
            defaults, PE_TEST_AD_BACKEND, qv, 1.25, 0.0;
            active = :q, want = :density, bound = (; data = dval))
        @test ad_gradient(default_bound, qv, 1.25, 0.0) ≈
              dval .- 2(1.25) .* qv
        @test_throws ArgumentError prepare_ad(
            defaults, PE_TEST_AD_BACKEND, qv;
            active = :q, want = :density, bound = (; data = dval))
    end
end
