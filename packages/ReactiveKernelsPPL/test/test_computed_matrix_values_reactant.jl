using Reactant

function _cm_inventory(text, executable = false)
    pattern = executable ?
        r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(" :
        r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+"
    counts = Dict{String,Int}()
    for m in eachmatch(pattern, text)
        name = executable ? m.captures[1] : m.match
        counts[name] = get(counts, name, 0) + 1
    end
    return counts
end

function _cm_save_backend(label, text)
    haskey(ENV, "RK_PPL_ACCEPTANCE_HLO_DIR") || return nothing
    dir = ENV["RK_PPL_ACCEPTANCE_HLO_DIR"]
    mkpath(dir)
    write(joinpath(dir, label), text)
    return nothing
end

@testset "computed matrix whole reads: default compiled parity and structure" begin
    previous = Dict{Symbol,Any}()
    for mode in (:product, :matvec), n in (3, 7, 11)
        fx = _cm_fixture(mode, true, n, 2n + 1)
        original = deepcopy(fx.data)
        f = _cm_build(fx)
        u = [0.13, -0.2]
        sampler = prepare_sampler(f.built, f.bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        kernel = sampler.kernel
        ru = Reactant.to_rarray(u)
        primal = Reactant.@compile kernel(ru)
        reverse = compile_ad_value_and_gradient(sampler.ad, ru)
        for v in (u, u .+ 0.17)
            ref = _cm_oracle(fx, f.built.layout, v)
            native, grad = sampler_value_and_gradient!(sampler, similar(v), v)
            rv = Reactant.to_rarray(v)
            value, gradient = reverse(rv)
            @test Float64(primal(rv)) ≈ ref.value rtol = 1e-12 atol = 1e-12
            @test Float64(value) ≈ native rtol = 1e-12 atol = 1e-12
            @test Float64(value) ≈ ref.value rtol = 1e-12 atol = 1e-12
            @test Array(gradient) ≈ grad rtol = 1e-12 atol = 1e-12
            @test Array(gradient) ≈ ref.gradient rtol = 1e-12 atol = 1e-12
        end
        @test isequal(fx.data, original)
        both = reverse.f
        modules = (repr(Reactant.@code_hlo optimize=false kernel(ru)),
            repr(Reactant.@code_hlo kernel(ru)),
            repr(Reactant.@code_hlo optimize=false both(ru)),
            repr(Reactant.@code_hlo both(ru)))
        executables = (repr(only(Reactant.XLA.get_hlo_modules(primal.exec))),
            repr(only(Reactant.XLA.get_hlo_modules(reverse.exec))))
        labels = ("primal.raw.mlir", "primal.default.mlir", "reverse.raw.mlir",
            "reverse.default.mlir", "primal.hlo", "reverse.hlo")
        for (label, text) in zip(labels, (modules..., executables...))
            _cm_save_backend("matrix-$mode-$n-$label", text)
        end
        inventories = (map(_cm_inventory, modules)...,
            map(t -> _cm_inventory(t, true), executables)...)
        @test all(!isempty, inventories)
        println("MATRIX_INVENTORIES ", mode, " ", n, " ", inventories)
        if haskey(previous, mode)
            # At three rows the raw product trace folds one constant and
            # two broadcasts. Compare complete raw inventories after that
            # bounded simplification; default MLIR and executable HLO must
            # stay fixed across every size, including three rows.
            @test map(i -> inventories[i], (2, 4, 5, 6)) ==
                map(i -> previous[mode][i], (2, 4, 5, 6))
            if n > 7 || mode === :matvec
                @test inventories == previous[mode]
            end
        end
        previous[mode] = inventories
    end
end
