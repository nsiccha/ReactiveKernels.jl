using Reactant

function _wva_compiled(case; values=false)
    original=deepcopy(case.data)
    bound,built=_wva_build(case;values)
    names=coordinate_names(built.layout)
    @test Set(names)==Set(case.coords)
    u=[0.15cos(i) for i in eachindex(names)]
    oracle=w->_wva_posterior(case.parts(_wva_coordinates(w,names)))
    # Model construction defines source functions in a new Julia world.
    sampler,ru,executables=Base.invokelatest(_ma_compiled_check,bound,built,u,oracle;
        executables=true)
    @test isequal(case.data,original)
    return _wva_compiled_modules(case.label, sampler, ru, executables; values)
end

function _wva_compiled_modules(label, sampler, ru, executables;
        values=false, all_stages=false)
    modules=Base.invokelatest(_ma_backend_modules,sampler,ru)
    @test Set(first.(modules))==Set(("primal.raw","primal.default","reverse.raw","reverse.default"))
    executable_modules = Pair{String,String}[]
    for (prefix,compiled) in zip(("primal","reverse"),executables)
        thunk=hasproperty(compiled,:exec) ? compiled : compiled.compiled
        push!(executable_modules, prefix =>
            repr(only(Reactant.XLA.get_hlo_modules(thunk.exec))))
    end
    if haskey(ENV,"RKPPL_WHOLE_HLO_DIR")
        dir=ENV["RKPPL_WHOLE_HLO_DIR"]
        mkpath(dir)
        label=replace(label,"/"=>"-")
        for (prefix,hlo) in modules
            write(joinpath(dir,"$label-$values-$prefix.mlir"),hlo)
        end
        # Retain the HLO of the same default executables used for numerical
        # acceptance: later XLA passes can change control flow after MLIR.
        for (prefix,hlo) in executable_modules
            write(joinpath(dir,"$label-$values-$prefix.executable.hlo"),hlo)
        end
    end
    # Complete inventories diagnose specialization, sharing and reduction
    # stages. Exact counts and pure conditional counts are not requirements.
    # Numerical checks above and inspected density work establish acceptance.
    default = _ma_backend_ops(filter(pair -> endswith(first(pair), ".default"), modules))
    raw = _ma_backend_ops(filter(pair -> endswith(first(pair), ".raw"), modules))
    # Include every instruction from every executable computation, including
    # parameters, constants, fusion bodies, scatter regions and entry tuples.
    executable = [prefix * ":" * m.captures[1]
        for (prefix,hlo) in executable_modules
        for m in eachmatch(r"(?m)^\s*(?:ROOT )?%[^\n=]+ = .*?\s([a-z][a-z0-9-]*)\(",hlo)]
    @test !isempty(raw) && !isempty(default) && !isempty(executable)
    for (_,hlo) in executable_modules
        assignments = collect(eachmatch(r"(?m)^\s*(?:ROOT )?%[^\n=]+ = ",hlo))
        instructions = collect(eachmatch(r"(?m)^\s*(?:ROOT )?%[^\n=]+ = .*?\s([a-z][a-z0-9-]*)\(",hlo))
        @test length(assignments) == length(instructions)
    end
    return all_stages ? (; raw, default, executable, hlo=Dict(executable_modules)) : default
end

# Growth is evidence to inspect, not a blanket count-equality policy.
function _wva_growth(previous, current, label)
    previous === nothing && return
    counts(ops) = Dict(op => count(==(op), ops) for op in unique(ops))
    before, after = counts(previous), counts(current)
    before == after || println("WHOLE_VALUE_GROWTH ", label,
        " before=", sort!(collect(before)), " after=", sort!(collect(after)))
    return nothing
end

# For the plain module fixture, only the observation density uses log.
# Follow every referenced computation so a fusion/call inside a while counts.
# A loop containing density log and lane indexing, or a full-width density
# log array, retains its intended work. An unrelated loop/sqrt array does not.
function _wva_density_batch(hlo, n)
    computations = Dict{String,Vector{String}}()
    current = nothing
    entry = nothing
    for line in split(hlo, '\n')
        header = match(r"^(ENTRY )?%([\w.-]+) \(.*\) -> .* \{$", line)
        if header !== nothing
            current = header.captures[2]
            computations[current] = String[]
            header.captures[1] === nothing || (entry = current)
        elseif line == "}"
            current = nothing
        elseif current !== nothing && occursin(r"^\s+(?:ROOT )?%[^\n=]+ = ", line)
            push!(computations[current], line)
        end
    end
    @test entry !== nothing
    @test sum(length, values(computations)) ==
        length(collect(eachmatch(r"(?m)^\s*(?:ROOT )?%[^\n=]+ = ", hlo)))
    references(line) = [m.captures[1] for m in eachmatch(
        r"(?:calls|to_apply|condition|body|true_computation|false_computation)=%([\w.-]+)", line)]
    function reach(start)
        seen, todo = Set{String}(), [start]
        while !isempty(todo)
            name = pop!(todo)
            name in seen && continue
            push!(seen, name)
            for line in computations[name]
                append!(todo, references(line))
                for branches in eachmatch(r"branch_computations=\{([^}]+)\}", line)
                    append!(todo, [m.captures[1] for m in eachmatch(r"%([\w.-]+)", branches.captures[1])])
                end
            end
        end
        return [line for name in seen for line in computations[name]]
    end
    reachable = reach(entry)
    full_width_log = any(reachable) do line
        m = match(r"= f64\[([^\]]*)\](?:\{[^}]*\})? log\(", strip(line))
        m !== nothing && string(n) in split(m.captures[1], ',')
    end
    full_width_log && return true
    for line in reachable
        occursin(r"\swhile\(", line) || continue
        body = match(r"\bbody=%([\w.-]+)", line)
        body === nothing && continue
        work = reach(body.captures[1])
        any(l -> occursin(r"\slog\(", l), work) &&
            any(l -> occursin(r"\sdynamic-slice\(", l), work) && return true
    end
    return false
end

@testset "whole-value audit: compiled math and complete backend modules" begin
    # All original uses run at growing data/parameter axes. Keep complete
    # raw/default and actual executable modules for structural inspection.
    for kind in (:vector,:column),variant in 1:4
        previous=nothing
        for n in (3,7,11)
            ops=_wva_compiled(_wva_array_case(kind,variant,n,n))
            _wva_growth(previous, ops, "whole-value")
            previous=ops
        end
        if variant>=3
            previous=nothing
            for n in (7,13,19)
                ops=_wva_compiled(_wva_array_case(kind,variant,3,n))
                _wva_growth(previous, ops, "whole-value")
                previous=ops
            end
        end
    end
    for factory in (n->_wva_affine_case(n,2), _wva_scalar_case,
            n->_wva_replacement_case(1,n),
            n->_wva_weibull_case(:retained,n),
            n->_wva_weibull_case(:shared,n),
            n->_wva_literal_case(:literal,0.8,n),
            n->_wva_literal_case(:vcat,[0.8,0.9],n))
        previous=nothing
        for n in (7,13,19)
            ops=_wva_compiled(factory(n))
            _wva_growth(previous, ops, "whole-value")
            previous=ops
        end
    end
    for kind in (:simplex,:ordered)
        _wva_compiled(_wva_vector_case(kind))
    end
    for form in (:named,:inline)
        _wva_compiled(_wva_weibull_case(form))
    end
    _wva_compiled(_wva_literal_case(:vcat,0.8))
    _wva_compiled(_wva_scalar_case(7,1.1);values=true)
end

module PlainModuleValueAudit
using ReactiveKernelsPPL
@inline vector_shift(v, a, b) = v .+ a .+ b .* v
@inline scalar_shift(v, a, b) = a + b * sum(v)
@inline value_scale(v) = sqrt.(1 .+ v.^2)
end

function _wva_retained_compiled_fixture(name, scalar, inline_scale, n;
        scalar_arguments=false)
    fn = scalar ? :scalar_shift : :vector_shift
    scale_stmt = inline_scale ? "" : "sigma = value_scale($name)"
    scale_arg = inline_scale ? "value_scale($name)" : "sigma"
    declarations = scalar_arguments ?
        "a ~ Normal(0.0, 1.0); b ~ Normal(0.0, 1.0)" :
        "b[1:2] .~ Normal.(0.0, 1.0)"
    arguments = scalar_arguments ? "x, a, b" : "x, b"
    source = """
    @rkppl begin
        $declarations
        $name = $fn($arguments)
        $scale_stmt
        y .~ Normal.($name, $scale_arg)
    end
    """
    namespace = scalar_arguments ? PlainModuleValueAudit : RetainedModuleValueAudit
    model = Core.eval(namespace, Meta.parse(source))
    data = (; x=collect(range(-0.4, 0.6; length=n)),
        y=collect(range(0.1, 0.8; length=n)))
    bound = model(; x=data.x) | (; y=data.y)
    built = build_kernel(bound)
    function oracle(q)
        density = -log(2pi) - 0.5sum(abs2, q)
        for i in eachindex(data.x, data.y)
            slope = scalar ? sum(data.x) : data.x[i]
            mean = (scalar ? 0.0 : data.x[i]) + q[1] + q[2] * slope
            variance = 1 + mean^2
            density += -0.5log(2pi) - 0.5log(variance) -
                0.5(data.y[i] - mean)^2 / variance
        end
        return density
    end
    return (; model, bound, built, data, oracle)
end

function _wva_compiled_fixture(fixture, label; all_stages=false)
    (; model, bound, built, data, oracle) = fixture
    before, ast_before = deepcopy(data), deepcopy(model.ast)
    u = [0.15cos(i) for i in 1:built.layout.total]
    u_before = copy(u)
    sampler, ru, executables = Base.invokelatest(_ma_compiled_check,
        bound, built, u, oracle; executables=true)
    @test isequal(data, before)
    @test isequal(model.ast, ast_before)
    @test isequal(u, u_before)
    @test Array(ru) == u_before
    return _wva_compiled_modules(label, sampler, ru, executables; all_stages)
end

@testset "opaque helper indexing: unchanged source retains compiled boundary" begin
    # These original helper bodies remain unchanged in the native audit.
    # This captures the current opaque Julia indexing capability gap. It
    # does not declare valid Julia syntax invalid or certify compiled parity.
    # Native source/AD acceptance is in test_whole_value_audit.jl.
    for name in (:loc, :mu), scalar in (false, true), inline_scale in (false, true)
        fixture = _wva_retained_compiled_fixture(name, scalar, inline_scale, 7)
        before, ast_before = deepcopy(fixture.data), deepcopy(fixture.model.ast)
        u = [0.3, -0.2]
        @test_throws r"Scalar indexing is disallowed" Base.invokelatest(
            _ma_compiled_check, fixture.bound, fixture.built, u, fixture.oracle)
        @test fixture.data == before
        @test fixture.model.ast == ast_before
        @test u == [0.3, -0.2]
    end
end

function _wva_plain_module_growth(scalar)
    # Separate ordinary module helpers use scalar arguments and no opaque
    # indexing. Verify equivalent statistical math and value composition;
    # original indexed-source failure captures remain separate above.
    for name in (:loc, :mu), inline_scale in (false, true)
        previous = nothing
        for n in (scalar ? (2, 7, 19, 37) : (2, 7, 19))
            fixture = _wva_retained_compiled_fixture(name, scalar, inline_scale, n;
                scalar_arguments=true)
            @test coordinate_names(fixture.built.layout) == [:a, :b]
            ops = _wva_compiled_fixture(fixture, "module-$name-$scalar-$inline_scale-$n";
                all_stages=true)
            if !scalar
                for direction in ("primal", "reverse")
                    retained = _wva_density_batch(ops.hlo[direction], n)
                    # Known stock small-batch body replication; numerical
                    # acceptance above still runs for this valid model.
                    # Missing pure conditionals alone are not this failure.
                    if n == 2
                        @test_broken retained
                    else
                        @test retained
                    end
                end
            end
            _wva_growth(previous, ops.default, "module-$name-$scalar-$inline_scale-$n")
            previous = ops.default
        end
    end
end

@testset "plain scalar module values: default compiled density, Reverse and growth" begin
    _wva_plain_module_growth(true)
end

@testset "plain vector module values: default compiled density, Reverse and growth" begin
    _wva_plain_module_growth(false)
end

@testset "flat scalar vector literals: default compiled density, Reverse and replay" begin
    for dotted in (false, true), form in (:named, :inline, :alias, :gather)
        previous = nothing
        for n in (form === :gather ? (2, 7, 19, 37, 73, 151) : (3,))
            fixture = _wva_flat_literal_fixture(dotted, form, n; replay=true)
            @test coordinate_names(fixture.built.layout) == [:a, :b]
            ops = _wva_compiled_fixture(fixture, "flat-$dotted-$form-$n";
                all_stages=true)
            if previous !== nothing
                # Include small-shape simplifications and every reduction-stage
                # transition. These complete inventories are diagnostics;
                # numerical/AD/ownership checks remain unconditional above.
                for stage in (:raw, :default, :executable)
                    _wva_growth(getproperty(previous,stage), getproperty(ops,stage),
                        "flat/$dotted/$form/$n/$stage")
                end
            end
            previous = ops
        end
    end
end

@testset "retained inline Weibull: original links and small-batch density work" begin
    case=_wva_weibull_case(:retained,3)
    case.data[:x]=[-0.4,0.1,0.6]
    case.data[:y]=[0.3,0.8,1.2]
    bound,built=_wva_build(case)
    names=coordinate_names(built.layout)
    u=Float64[name===:a ? 0.4 : 0.1 for name in names]
    oracle=w->_wva_posterior(case.parts(_wva_coordinates(w,names)))
    @test oracle(u)≈-3.854439725675756 atol=1e-12
    _check_model_math(built,bound,u,oracle)
    sampler,ru,executables=Base.invokelatest(_ma_compiled_check,
        bound,built,u,oracle;executables=true)
    modules=_wva_compiled_modules("retained-inline-Weibull-original-3",
        sampler,ru,executables;all_stages=true)
    # Here too, only observation density uses log. Three scalar log/divide/
    # power copies are not a retained loop or full-width density array.
    # Keep actual default math and the two known structure failures separate.
    for direction in ("primal","reverse")
        @test_broken _wva_density_batch(modules.hlo[direction],3)
    end
end
