using ReactiveKernels, Reactant, Enzyme, Test

# Prototype, not a public error-policy API. Run this file in a scratch
# environment with ReactiveKernels, Reactant 0.2.290, Enzyme and Test.
#
# A deliberately small source-transform proof: a transparent scalar helper
# with leading `condition || throw(ArgumentError(...))` guards and a final
# value. Keep its ordinary throwing definition; synthesize a continuation
# variant without constructing an exception on the handled path.
original = :(function checked_log(x)
    x > 0 || throw(ArgumentError("positive x required"))
    log(x)
end)
Core.eval(Main, original)

function recovered_definition(def)
    signature, body = def.args
    name = Symbol(signature.args[1], :_recover)
    statements = [s for s in body.args if !(s isa LineNumberNode)]
    function lower(statements)
        first, rest = statements[1], statements[2:end]
        if first isa Expr && first.head === :||
            thrown = first.args[2]
            thrown isa Expr && thrown.head === :call && thrown.args[1] === :throw ||
                error("probe accepts only an explicit guard throw")
            exception = thrown.args[2]
            exception isa Expr && exception.head === :call &&
                exception.args[1] === :ArgumentError ||
                error("probe accepts only ArgumentError")
            Expr(:if, first.args[1], lower(rest), Expr(:call, :reject, :context))
        elseif isempty(rest)
            Expr(:call, :accept, first, :context)
        else
            Expr(:block, first, lower(rest))
        end
    end
    Expr(:function, Expr(:call, name, :accept, :reject, signature.args[2:end]..., :context),
        lower(statements))
end
recovered = recovered_definition(original)
println("GENERATED_SOURCE ", recovered)
Core.eval(Main, Expr(:macrocall, Symbol("@traceable"), LineNumberNode(1), recovered))

@traceable function checked_sum_recover(X, fallback)
    ok, acc = true, 0.0
    for i in 1:length(X)
        # Copy an unchanged carry when returning it from a branch: traced
        # loop slots update their tracer objects, so these results need their
        # own wrappers. Native scalar copies preserve the authored value.
        ok, acc = if ok
            checked_log_recover((v, seed) -> (true, seed + v), seed -> (false, copy(seed)), X[i], acc)
        else
            (copy(ok), copy(acc))
        end
    end
    if ok
        acc
    else
        fallback
    end
end

@kernel recovered_sum(X, fallback) = begin
    result = checked_sum_recover(X, fallback)
    return result
end

function native_reference(X, fallback)
    try
        sum(checked_log, X; init=0.0)
    catch e
        e isa ArgumentError || rethrow()
        fallback
    end
end

@test_throws ArgumentError checked_log(-1.0)
k = prepare(recovered_sum; bound=(;fallback=-Inf))
gradient(X) = Enzyme.gradient(Enzyme.Reverse, k, X)
maps, reverse_maps = [], []
for N in (3, 8)
    X = collect(2.0:(N + 1.0))
    rx = Reactant.to_rarray(X)
    hlo = repr(Reactant.@code_hlo optimize=false k(rx))
    counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme)\.\w+", hlo)
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    push!(maps, counts)
    @test count("stablehlo.while", hlo) == 1
    @test occursin("stablehlo.if", hlo)
    compiled = Reactant.@compile k(rx)
    reverse = repr(Reactant.@code_hlo optimize=:only_enzyme gradient(rx))
    reverse_counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme)\.\w+", reverse)
        reverse_counts[m.match] = get(reverse_counts, m.match, 0) + 1
    end
    push!(reverse_maps, reverse_counts)
    @test occursin("stablehlo.while", reverse)
    @test occursin("stablehlo.if", reverse)
    compiled_gradient = Reactant.@compile gradient(rx)
    @test Float64(compiled(rx)) ≈ native_reference(X, -Inf)
    @test Array(only(compiled_gradient(rx))) ≈ 1 ./ X
    for bad in ([0.0; X[2:end]], [X[1]; -1.0; X[3:end]], fill(-1.0, N))
        rbad = Reactant.to_rarray(bad)
        old = copy(bad)
        @test k(bad) == native_reference(bad, -Inf) == -Inf
        @test Float64(compiled(rbad)) == -Inf
        @test only(gradient(bad)) == zeros(N)
        @test Array(only(compiled_gradient(rbad))) == zeros(N)
        @test bad == old && Array(rbad) == old
    end
    println("RECOVERY_OK N=$N while=1 ops=", counts)
end
@test maps[1] == maps[2]
@test reverse_maps[1] == reverse_maps[2]
custom = prepare(recovered_sum; bound=(;fallback=-123.0))
x = [-1.0, 2.0, 3.0]
cx = Reactant.to_rarray(x)
custom_compiled = Reactant.@compile custom(cx)
@test Float64(custom_compiled(cx)) == -123.0
println("CUSTOM_FALLBACK_OK")
