using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions

# A generated plate cell or scan step reads every value it shares across
# cells or steps by closing over it, never as a `Ref(...)` operand (user
# decision 0kvm2ip: RK deprecates `Ref` do-block operands). The operands of a
# generated `plate(...) do` / `scan(...) do` are the per-cell and per-step
# values only.

function _svc_do_operands(ex, out = Any[])
    ex isa Expr || return out
    if ex.head === :do && Meta.isexpr(ex.args[1], :call) &&
            ex.args[1].args[1] in (:plate, :scan)
        push!(out, ex.args[1].args[2:end])
    end
    foreach(a -> _svc_do_operands(a, out), ex.args)
    return out
end

_svc_is_ref(a) = Meta.isexpr(a, :call, 2) && a.args[1] in (:Ref, GlobalRef(Base, :Ref))

function _svc_check(program, data, observed)
    bound = bind_data(lower_rkppl(program, Tuple(keys(data)); mod = @__MODULE__,
        conditioned = observed), data)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    operands = _svc_do_operands(def)
    @test !isempty(operands)
    @test !any(ops -> any(_svc_is_ref, ops), operands)
    built = build_kernel(bound)
    u = [0.1 * sin(i) for i in 1:built.layout.total]
    @test isfinite(Base.invokelatest(prepare_query(built, bound, :sampler), u))
end

@testset "generated cells and steps close over shared values" begin
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.3, -0.2]
    # Ordered cutpoints: every cell gathers its own level from the whole vector.
    _svc_check(quote
        b ~ Normal(0, 2)
        c ~ Ordered(Normal(0.5, 2), 2)
        eta = b .* x
        y .~ OrderedLogistic.(eta, Ref(c))
    end, Dict{Symbol,AbstractVector}(:y => [1, 2, 3, 2, 1, 3, 3, 2], :x => x), (:y,))
    # Simplex mixture weights: one log-weight vector read by every cell.
    _svc_check(quote
        w ~ Dirichlet([1.0, 1.0])
        mu1 ~ Normal(-2.0, 0.1)
        mu2 ~ Normal(2.0, 0.1)
        sigma ~ Exponential(1.0)
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref(w))
    end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]), (:y,))
    # An `@plate` cell reading parameters and data columns whole.
    _svc_check(quote
        sigma ~ Exponential(1.0)
        b0 ~ Normal(0.0, 1.0)
        @plate for i in eachindex(obs)
            mu = (b0 * dose[i]) * t[i]
            obs[i] ~ Normal(mu, sigma)
        end
    end, Dict{Symbol,AbstractVector}(:t => [1.0, 2.0, 3.0], :dose => [0.5, 1.0, 1.5],
        :obs => [0.4, 1.1, 2.0]), (:obs,))
    # One array per index: the group plate zips the per-index values.
    xs = [[0.1, 0.2], Float64[], [0.3, 0.4, 0.5]]
    _svc_check(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(a .+ b .* x[i], sigma)
        end
    end, Dict{Symbol,Any}(:x => xs, :y => [0.3 .+ v for v in xs]), (:y,))
    # Scan recurrences read their parameters by closing over them.
    y = [0.3 * sin(t) for t in 1:6]
    _svc_check(quote
        phi ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], sigma)
            end
        end
        y .~ Normal.(h, 1.0)
    end, Dict{Symbol,Any}(:y => y), (:y,))
    _svc_check(quote
        phi ~ Normal(0, 1)
        @scan begin
            h[1] = phi
            for t in 2:T
                h[t] = phi * h[t - 1]
            end
        end
        y .~ Normal.(h, 1.0)
    end, Dict{Symbol,Any}(:y => y), (:y,))
end
