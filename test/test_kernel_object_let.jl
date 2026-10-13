module KernelObjectLetTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A `let` in a pure endpoint only scopes its bindings, so the method is a
# transparent endpoint exactly like the parenthesized-block spelling. It used to
# make the method an opaque stateful method instead: the object had no
# `logkernel` endpoint, and keyword construction failed with a `MethodError` on
# the skeleton (todo `ReactiveKernels:kernel-let/2026-10-13T09-37-17-462-0euf55d`).
# The zero-count arm of a zero-inflated count density reads `nl1pe` twice.
@kernel inflated_let(n::Int, logit::Float64, zi::Float64) = begin
    logkernel(observed::Int)::Float64 =
        observed == 0 ?
            let nl1pe = n * log1p(exp(logit))
                nl1pe <= 700 ? log1p(zi * expm1(nl1pe)) - nl1pe : log(zi)
            end :
            log1p(-zi) - observed * log1p(exp(-logit)) -
                (n - observed) * log1p(exp(logit))
end

@kernel inflated_block(n::Int, logit::Float64, zi::Float64) = begin
    logkernel(observed::Int)::Float64 =
        observed == 0 ?
            (nl1pe = n * log1p(exp(logit));
             nl1pe <= 700 ? log1p(zi * expm1(nl1pe)) - nl1pe : log(zi)) :
            log1p(-zi) - observed * log1p(exp(-logit)) -
                (n - observed) * log1p(exp(logit))
end

@kernel let_total(y::Vector{Int}, logit::Vector{Float64}, n::Int, zi::Float64) = begin
    cells = plate(y, logit) do yi, li
        inflated_let(; n = n, logit = li, zi = zi).logkernel(yi)
    end
    total::Float64 = sum(cells)
end

@kernel block_total(y::Vector{Int}, logit::Vector{Float64}, n::Int, zi::Float64) = begin
    cells = plate(y, logit) do yi, li
        inflated_block(; n = n, logit = li, zi = zi).logkernel(yi)
    end
    total::Float64 = sum(cells)
end

@kernel let_scopes(scale::Float64) = begin
    sequential(x::Float64)::Float64 = let a = 2x, b = a + scale
        a * b
    end
    # The binding shadows the owner port inside the `let` only.
    shadowed(x::Float64)::Float64 = let scale = scale + 1
        scale * x
    end
    statement(x::Float64)::Float64 = begin
        y::Float64 = let t = x^2
            u = t + scale
            u * t
        end
        y + 1
    end
    # A sibling endpoint called with outer arguments inside a `let`.
    sibling(x::Float64)::Float64 = let s = 3.0
        s * sequential(x)
    end
    # A `let` binding that shadows an endpoint name is an ordinary local.
    shadowed_endpoint(x::Float64)::Float64 = let sequential = abs
        sequential(x)
    end
end

@kernel let_local_argument(scale::Float64) = begin
    base(x::Float64)::Float64 = scale * x
    outer(x::Float64)::Float64 = let z = 2x
        base(z)
    end
end

@testset "a let in a pure endpoint keeps the object transparent" begin
    @test ReactiveKernels.kernel_endpoint_names(inflated_let) == (:logkernel,)
    @test ReactiveKernels.kernel_endpoint_names(let_scopes) ==
          (:sequential, :shadowed, :statement, :sibling, :shadowed_endpoint)
    @test occursin("let nl1pe",
                   string(readable_code(prepare(inflated_let.logkernel))))

    endpoint_let = prepare(inflated_let.logkernel)
    endpoint_block = prepare(inflated_block.logkernel)
    for (n, logit, zi, observed) in ((8, -0.4, 0.3, 0), (8, 100.0, 0.3, 0),
                                     (8, 0.7, 0.2, 3), (8, -1.3, 0.6, 8))
        @test endpoint_let(n, logit, zi, observed) ===
              endpoint_block(n, logit, zi, observed)
    end

    y = [0, 0, 3, 5, 0, 8]
    logit = [-0.4, 100.0, 0.7, -1.3, 0.2, 1.1]
    n, zi = 8, 0.3
    total_let = prepare(let_total)
    total_block = prepare(block_total)
    @test total_let(y, logit, n, zi) === total_block(y, logit, n, zi)
    @test total_let(y, logit, n, zi) ≈
          sum(endpoint_block(n, l, zi, o) for (o, l) in zip(y, logit))

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    ad_let = prepare_ad(total_let, backend, y, logit, n, zi; active = :logit)
    ad_block = prepare_ad(total_block, backend, y, logit, n, zi; active = :logit)
    value_let, gradient_let = ad_value_and_gradient(ad_let, y, logit, n, zi)
    value_block, gradient_block = ad_value_and_gradient(ad_block, y, logit, n, zi)
    @test value_let === value_block
    @test gradient_let == gradient_block
    h = 1e-6
    central = map(eachindex(logit)) do i
        up = copy(logit); up[i] += h
        down = copy(logit); down[i] -= h
        (total_block(y, up, n, zi) - total_block(y, down, n, zi)) / 2h
    end
    @test gradient_let ≈ central rtol = 1e-6

    scopes = (
        sequential = x -> 2x * (2x + 1.5),
        shadowed = x -> 2.5x,
        statement = x -> (x^2 + 1.5) * x^2 + 1,
        sibling = x -> 3 * 2x * (2x + 1.5),
        shadowed_endpoint = abs,
    )
    for (name, expected) in pairs(scopes)
        endpoint = prepare(getproperty(let_scopes, name))
        for x in (-0.7, 2.0)
            @test endpoint(1.5, x) ≈ expected(x)
        end
    end

    # An endpoint call reading a `let`-bound name stays inside the `let`: the
    # object's builder prepares the callee once and the call applies it there.
    @test prepare(let_local_argument.outer)(1.5, 2.0) == 6.0

    # Capability gap: the in-place call passes the owner's ports by name, so
    # a `let` rebinding one of them would hand the callee the `let`'s value.
    # Until that is built, definition fails naming the `let`.
    defined = try
        Core.eval(@__MODULE__, :(@kernel let_shadowed_argument(scale::Float64) = begin
            base(x::Float64)::Float64 = scale * x
            outer(x::Float64)::Float64 = let scale = 2x
                base(scale)
            end
        end))
        :defined
    catch err
        err
    end
    @test_broken defined === :defined
    defined === :defined ||
        @test occursin("rename the `let` binding", sprint(showerror, defined))
end

end
