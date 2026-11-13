using ReactiveKernels, MutatingFunctions, Test

@kernel reusable_sum_init(x::Vector{Float64}, z::Vector{Float64}, initial::Float64) = begin
    total::Float64 = sum(x .* z; init = initial)
    return total
end
@kernel reusable_mapped_sum_init(x::Vector{Float64}, initial::Float64) = begin
    total::Float64 = sum(abs2, x .+ 1; init = initial)
    return total
end
@noinline init_positional(x, fail) = fail ? throw(ArgumentError("positional")) : x
@noinline init_keyword(x, fail) = fail ? throw(ArgumentError("keyword")) : x
@kernel reusable_sum_init_order(x::Vector{Float64}, fail::Bool) = begin
    total::Float64 = sum(init_positional(x, fail);
        init = sum(init_keyword(x, fail) .* 0))
    return total
end
_init_exception(f, args...) = try
    f(args...)
    nothing
catch e
    (typeof(e), sprint(showerror, e))
end
function _init_allocations(f::F, args::T) where {F,T}
    f(args...)
    @allocated f(args...)
end
_init_ast_size(x) = x isa Expr ? 1 + sum(_init_ast_size, x.args; init=0) : 1

@testset "runtime init in reusable scalar reductions" begin
    p = prepare(reusable_sum_init; want=:total)
    pm = prepare(reusable_mapped_sum_init; want=:total)
    k = prepare_nonallocating(p, Float64[], Float64[], 0.0)
    km = prepare_nonallocating(pm, Float64[], 0.0)
    sizes = Int[]
    for n in (0, 7, 128, 2048, 7)
        x = sin.(collect(1.0:n)); z = cos.(collect(1.0:n))
        oldx, oldz = copy(x), copy(z)
        for initial in (-3.0, 0.0, 2.5)
            @test k(x,z,initial) == p(x,z,initial) == sum(x .* z; init=initial)
            @test km(x,initial) == pm(x,initial) == sum(abs2,x .+ 1;init=initial)
            @test _init_allocations(k,(x,z,initial)) == 0
            @test _init_allocations(km,(x,initial)) == 0
        end
        @test x == oldx && z == oldz
        sized = prepare_nonallocating(p,x,z,0.0)
        push!(sizes,_init_ast_size(code_expr(sized)))
        # The emitted call still uses the reusable materialization operation;
        # its runtime array loop is not replicated in RK source by length.
        @test any(op -> op isa ReactiveKernels._MaterializeStep,sized.callables)
    end
    @test all(==(first(sizes)),sizes)
    order = prepare(reusable_sum_init_order;want=:total)
    reused = prepare_nonallocating(order,[1.,2.,3.],false)
    native(x,fail) = sum(init_positional(x,fail);init=sum(init_keyword(x,fail).*0))
    @test reused([1.,2.,3.],false) == native([1.,2.,3.],false)
    @test _init_exception(reused,[1.,2.,3.],true) == _init_exception(native,[1.,2.,3.],true)
    @test _init_exception(order,[1.,2.,3.],true) == _init_exception(native,[1.,2.,3.],true)
end
