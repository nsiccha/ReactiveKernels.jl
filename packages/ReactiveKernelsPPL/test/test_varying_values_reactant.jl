using Reactant

@testset "Reactant: membership and stratified values retain iteration" begin
    for kind in (:membership, :distinct, :stratified, :single, :both)
        traces, recipes = Dict{String,Int}[], Int[]
        for (n, S) in ((7, 2), (19, 4))
            bound, built = _cv_build(kind, n, S)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            push!(recipes, length(built.spec.graph.recipes))
            push!(traces, Base.invokelatest(_pcr_measure, built, bound, u; structure_ad = true))
        end
        @test recipes[1] == recipes[2]
        @test traces[1] == traces[2]
    end
end

function _cv_hlo_ops(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", repr(hlo))
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    return ops
end

function _cv_hlo_retained_work(ops)
    # Constant sharing, scalar identities and singleton tape reshapes may
    # specialize a shape without replicating its authored computation. An
    # integer index broadcast to a 1x1 tape also folds to a reshape at K=2.
    simplified = ("stablehlo.constant", "stablehlo.reshape", "stablehlo.add",
        "stablehlo.multiply", "stablehlo.subtract", "stablehlo.negate",
        "stablehlo.broadcast_in_dim")
    return Dict(name => count for (name, count) in ops if name ∉ simplified)
end

function _cv_executable_ops(compiled)
    thunk = hasproperty(compiled, :exec) ? compiled : compiled.compiled
    hlo = repr(only(Reactant.XLA.get_hlo_modules(thunk.exec)))
    ops = Dict{String,Int}()
    # Tuple result types contain index comments with '=' in them.
    for m in eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", hlo)
        name = m.captures[1]
        ops[name] = get(ops, name, 0) + 1
    end
    return ops
end

function _cv_executable_retained_work(ops)
    # Inspect actual executable regions and nonlinear work separately from
    # XLA's shape-specific fusion, layout and derivative-tape machinery.
    names = ("while", "conditional", "log", "log-plus-one", "exponential",
        "sqrt", "tanh", "sine", "cosine", "floor", "dot", "gather")
    return Dict(name => get(ops, name, 0) for name in names)
end

@testset "Reactant: data-sized shared LKJ factor reverse" begin
    # Constructing one shared diagonal value avoids the raw matrix/guard
    # dominance failure isolated by the backend-only reproducer. Every prior
    # logarithm and its derivative still belongs to the live shape guard.
    traced, optimized, executable, recipes = [], [], [], Int[]
    for (K, n, S) in ((2, 7, 2), (2, 19, 4), (4, 7, 2),
            (8, 7, 2), (16, 19, 4))
        bound, built = _cv_data_width_build(n, S; K)
        original = deepcopy(bound.columns)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        q = prepare_sampler(built, bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        ru = Reactant.to_rarray(u)
        kernel = prepare_query(built, bound, :sampler)
        ad = q.ad
        both(w) = ad_value_and_gradient(ad, w)
        push!(recipes, length(built.spec.graph.recipes))
        push!(traced, (
            _cv_hlo_ops(Reactant.@code_hlo optimize = false kernel(ru)),
            _cv_hlo_ops(Reactant.@code_hlo optimize = false both(ru))))
        push!(optimized, (
            _cv_hlo_ops(Reactant.@code_hlo optimize = true kernel(ru)),
            _cv_hlo_ops(Reactant.@code_hlo optimize = true both(ru))))
        println("shared LKJ optimized inventory, (K, n, S)=", (K, n, S),
            ": ", optimized[end])
        @test get(traced[end][1], "stablehlo.while", 0) > 0
        primal = Reactant.@compile kernel(ru)
        compiled = compile_ad_value_and_gradient(q.ad, ru)
        push!(executable, map(_cv_executable_ops, (primal, compiled)))
        println("shared LKJ executable inventory, (K, n, S)=", (K, n, S),
            ": ", executable[end])
        # Reuse both executables at new coordinates; eta, scales and every
        # partial correlation change without rebuilding the graph.
        for w in (u, u .+ 0.03)
            rw = Reactant.to_rarray(w)
            value, gradient = sampler_value_and_gradient!(q, similar(w), w)
            reference = _cv_data_width_reference(built, bound, w)
            cvalue, cgradient = compiled(rw)
            @test value ≈ reference rtol = 1e-11
            @test Float64(primal(rw)) ≈ reference rtol = 1e-9
            @test Float64(cvalue) ≈ reference rtol = 1e-9
            @test Array(cgradient) ≈ gradient rtol = 1e-8 atol = 1e-9
            K <= 4 && @test Array(cgradient) ≈ _findiff_grad(
                v -> _cv_data_width_reference(built, bound, v), w) rtol = 1e-5 atol = 1e-7
            @test Array(rw) == w
        end
        if K == 2 && n == 7
            # Underflow the sampled positive shape to zero. The same compiled
            # model must take its invalid-shape guards on a later call.
            invalid = copy(u)
            eta = only(e for e in built.layout.entries if e.name === :eta)
            invalid[eta.offset] = -1000.0
            ri = Reactant.to_rarray(invalid)
            value, gradient = sampler_value_and_gradient!(q, similar(invalid), invalid)
            cvalue, cgradient = compiled(ri)
            @test value == -Inf
            @test all(isfinite, gradient)
            @test Float64(cvalue) == value
            @test Array(cgradient) ≈ gradient rtol = 1e-8 atol = 1e-9
            @test Array(ri) == invalid
        end
        @test bound.columns == original
        @test Array(ru) == u
    end
    @test allequal(recipes)
    @test allequal(traced)
    # Default optimized AD has expanded its derivative graph. Beyond the
    # small-loop optimizer boundary both primal and reverse retain structure.
    @test optimized[1] == optimized[2]
    @test optimized[4] == optimized[5]
    @test all(ops -> get(ops, "stablehlo.while", 0) > 0, optimized[4])
    # Complete inventories above must stop growing. Across every dimension,
    # control flow, nonlinear work and indexing must keep one authored body;
    # only the measured scalar/tape simplifications may alter raw counts.
    retained = [map(_cv_hlo_retained_work, pair) for pair in optimized]
    # Stock Reactant still expands small data-derived loops. Promote this
    # named marker when the generic compiler retention fix is delivered.
    @test_broken allequal(retained)
    @test executable[1] == executable[2]
    # MLIR retention alone does not protect a singleton loop or lazy branch
    # from later XLA simplification. Keep this stock limitation named too.
    @test_broken allequal([map(_cv_executable_retained_work, pair)
        for pair in executable])
end

@testset "Reactant: retained LKJ prior guard is lazy" begin
    kernel = _cv_lkj_diagonal_guard(4)
    valid = [1.5, 1.0, 0.8, 0.7, 0.6]
    ad = prepare_ad(kernel, _GEN_BACKEND, valid; active = :unconstrained)
    rv = Reactant.to_rarray(valid)
    compiled = compile_ad_value_and_gradient(ad, rv)
    value, gradient = compiled(rv)
    reference(u) = sum((4 - i + 2u[1] - 2) * log(u[i + 1]) for i in 2:4)
    @test Float64(value) ≈ reference(valid) rtol = 1e-12
    @test Array(gradient) ≈ _findiff_grad(reference, valid) rtol = 1e-7 atol = 1e-8
    for eta in (-1.0, 0.0, Inf, NaN)
        u = [eta, 1.0, -0.2, -0.3, -0.4]
        ru = Reactant.to_rarray(u)
        value, gradient = compiled(ru)
        @test Float64(value) == 0.0
        @test Array(gradient) == zeros(length(u))
        @test isequal(Array(ru), u)
    end
end

@testset "Reactant: ordered grouping axes retain iteration" begin
    for kind in (:unique, :alias, :computed, :provided, :range, :stepped,
            :descending, :cell, :crossed, :labels, :factor)
        traces, recipes = Dict{String,Int}[], Int[]
        for (n, G) in ((7, 3), (19, 5))
            bound, built = kind === :cell ? _gv_plate_build(n, G) :
                kind === :factor ? _gv_factor_build(n, G) :
                kind === :crossed ? _gv_crossed_build(n, G) :
                kind === :labels ? _gv_build(:alias, n, G; labels = true) :
                _gv_build(kind, n, G)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            push!(recipes, length(built.spec.graph.recipes))
            push!(traces, Base.invokelatest(_pcr_measure, built, bound, u; structure_ad = true))
        end
        @test recipes[1] == recipes[2]
        @test traces[1] == traces[2]
    end
end
