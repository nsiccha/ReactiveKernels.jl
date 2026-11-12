using ReactiveKernels, MutatingFunctions, DifferentiationInterface, LinearAlgebra, Test
import Enzyme

module NestedProductFixture
using ReactiveKernels
@kernel inline_product(x, A, s, t) = begin
    z = (A * (x .* s) .+ A * x) .* t
    terms = plate(z) do v
        abs2(v)
    end
    total = sum(terms)
end
@kernel named_product(x, A, s, t) = begin
    scaled = A * (x .* s)
    direct = A * x
    z = (scaled .+ direct) .* t
    terms = plate(z) do v
        abs2(v)
    end
    total = sum(terms)
end
@kernel guarded_product(x, A, s, t, take) = begin
    z = t .* (take ? A * (x .* s) : A * x[1:1])
    terms = plate(z) do v
        abs2(v)
    end
    total = sum(terms)
end
end

# Count syntax, not the printed lengths of names or values in the tables.
# This measures RK emission; Julia may specialize its executable afterwards.
nested_product_nodes(ex) = ex isa Expr ? 1 + sum(nested_product_nodes, ex.args; init=0) : 0
function nested_product_bytes(k, x, A, s, t)
    k(x, A, s, t)
    @allocated k(x, A, s, t)
end
@testset "inline products in cell broadcasts reuse their destinations" begin
    RK = ReactiveKernels
    F = NestedProductFixture
    @test Base.get_extension(RK, :ReactiveKernelsMutatingFunctionsExt) !== nothing
    @test Base.get_extension(RK, :ReactiveKernelsEnzymeExt) !== nothing
    backend = AutoEnzyme(mode=Enzyme.Reverse)
    for matrix in (false, true)
        syntax_sizes = Int[]
        for n in (0, 1, 7, 41)
            A = reshape(sin.(1.0:(3n)), n, 3)
            x = matrix ? reshape([0.2, -0.3, 0.5, 0.4, 0.1, -0.2], 3, 2) : [0.2, -0.3, 0.5]
            s = [1.0, 2.0, 3.0]
            t = cos.(1.0:n)
            originals = deepcopy((x, A, s, t))
            ks = map((F.inline_product, F.named_product)) do spec
                prepare_nonallocating(spec, x, A, s, t;
                    have=(:x, :A, :s, :t), want=:total)
            end
            k, named = ks
            @test count(op -> op isa RK._MatMulStep, k.callables) == 2
            @test occursin("_cell_array_", string(code_expr(k)))
            push!(syntax_sizes, nested_product_nodes(code_expr(k)))
            for point in (x, x .+ 0.25, x)
                reference = sum(abs2, (A * (point .* s) .+ A * point) .* t)
                @test k(point, A, s, t) ≈ reference atol=1e-14
                @test k(point, A, s, t) == named(point, A, s, t)
            end
            @test (x, A, s, t) == originals
            @test nested_product_bytes(k, x, A, s, t) <= nested_product_bytes(named, x, A, s, t)

            # Binding data must retain the same steps. Differentiate the
            # authored primal with ordinary Enzyme Reverse, and compare to
            # Julia's native algebra for both vector and matrix parameters.
            bound = prepare(F.inline_product; have=(:x, :A, :s, :t),
                want=:total, bound=(; A, s, t))
            cached = prepare_nonallocating(bound, x)
            ad = prepare_ad(cached, backend, x; active=:x)
            ordinary = prepare_ad(bound, backend, x; active=:x)
            g, native = similar(x), similar(x)
            for point in (x, x .+ 0.1, x)
                z = (A * (point .* s) .+ A * point) .* t
                expected = 2 .* (transpose(A) * (t .* z)) .* (s .+ 1)
                v, _ = ad_value_and_gradient!(ad, g, point)
                vref, _ = ad_value_and_gradient!(ordinary, native, point)
                @test v ≈ vref atol=1e-14
                @test g ≈ native atol=1e-13
                @test g ≈ expected atol=1e-13
            end
            @test (x, A, s, t) == originals
        end
        @test allequal(syntax_sizes)
    end

    # The primal's destination guards still accommodate a different row
    # count. No shape is copied into generated source by this decomposition.
    x, s = [0.2, -0.3, 0.5], [1.0, 2.0, 3.0]
    A, t = ones(2, 3), ones(2)
    k = prepare_nonallocating(F.inline_product, x, A, s, t;
        have=(:x, :A, :s, :t), want=:total)
    for n in (2, 5, 0, 1, 2)
        B, u = fill(0.5, n, 3), ones(n)
        @test k(x, B, s, u) ≈ sum(abs2, (B * (x .* s) .+ B * x) .* u)
    end

    guarded = prepare_nonallocating(F.guarded_product, x, A, s, t, true;
        have=(:x, :A, :s, :t, :take), want=:total)
    @test guarded(x, A, s, t, true) ≈ sum(abs2, t .* (A * (x .* s)))
    @test_throws DimensionMismatch guarded(x, A, s, t, false)
    @test guarded(x, A, s, t, true) ≈ sum(abs2, t .* (A * (x .* s)))
    guarded_ad = prepare_ad(guarded, backend, x, A, s, t, true; active=:x)
    g = similar(x)
    v, _ = ad_value_and_gradient!(guarded_ad, g, x, A, s, t, true)
    @test v ≈ sum(abs2, t .* (A * (x .* s)))
    @test g ≈ 2 .* (transpose(A) * (t .* t .* (A * (x .* s)))) .* s
end
