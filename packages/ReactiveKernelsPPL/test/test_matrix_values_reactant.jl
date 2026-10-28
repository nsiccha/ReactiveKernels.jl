using Reactant

function _ma_backend_modules(sampler,ru; reverse=true)
    post=sampler.kernel
    modules=[("primal.raw",repr(Reactant.@code_hlo optimize=false post(ru))),
        ("primal.default",repr(Reactant.@code_hlo post(ru)))]
    if reverse
        ad=sampler.ad
        both(v)=ad_value_and_gradient(ad,v)
        append!(modules,[("reverse.raw",repr(Reactant.@code_hlo optimize=false both(ru))),
            ("reverse.default",repr(Reactant.@code_hlo both(ru)))])
    end
    return modules
end

_ma_backend_ops(modules)=[prefix*":"*m.match for (prefix,hlo) in modules
    for m in eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith|scf|cf|tensor|math|linalg|memref)\.\w+",hlo)]

function _ma_save_modules(modules,kind,n)
    # Optional full IR receipts for a focused verification run. Tests compare
    # every operation without filtering packing or control-flow operations.
    haskey(ENV,"RKPPL_MATRIX_HLO_DIR") || return nothing
    dir=ENV["RKPPL_MATRIX_HLO_DIR"]
    mkpath(dir)
    for (prefix,hlo) in modules
        write(joinpath(dir,"$(kind)-$(n)-$(prefix).mlir"),hlo)
    end
    return nothing
end

function _ma_compiled_check(bound,built,u,oracle; executables=false)
    sampler=prepare_sampler(built,bound,u;backend=_GEN_BACKEND)
    grad=similar(u)
    native,_=sampler_value_and_gradient!(sampler,grad,u)
    ru=Reactant.to_rarray(u)
    primal=Reactant.@compile sampler.kernel(ru)
    reverse=nothing
    @test Float64(primal(ru))≈oracle(u) rtol=1e-12
    if isempty(u)
        # Standard zero-coordinate native reverse works; released Reactant
        # cannot export an empty gradient (the existing standalone boundary).
        @test isempty(grad)
        @test_throws r"'tensor.empty' op unsupported op for export to XLA" compile_ad_value_and_gradient(sampler.ad,ru)
    else
        reverse=compile_ad_value_and_gradient(sampler.ad,ru)
        for shift in (0.0,0.17)
            v=u.+shift
            value,gradient=reverse(Reactant.to_rarray(v))
            @test Float64(value)≈oracle(v) rtol=1e-12
            @test Array(gradient)≈_findiff_grad(oracle,v) rtol=1e-5 atol=1e-7
            if shift==0
                @test Float64(value)≈native
                @test Array(gradient)≈grad
            end
        end
    end
    return executables ? (sampler,ru,(primal,reverse)) : (sampler,ru)
end

@testset "ordinary matrix values: default compiled density and reverse" begin
    for (kind,n) in _MA_SURFACE_FIXTURES
        @testset "$kind / $n" begin
            case=_ma_surface_case(kind,n)
            original=deepcopy(case.data)
            bound,built=_ma_surface_model(case)
            u=[0.2sin(i) for i in 1:built.layout.total]
            _ma_compiled_check(bound,built,u,w->_ma_surface_parts(case,built.layout,w).posterior)
            @test isequal(case.data,original)
        end
    end
end

@testset "matrix roles and derived R2D2: default compiled reverse" begin
    for kind in (:response,:term,:weights,:scale,:group)
        @testset "$kind" begin
            plan,data=_ma_role_plan(kind,9)
            original=deepcopy(data)
            bound=bind_data(plan,data)
            built=build_kernel(bound)
            _ma_compiled_check(bound,built,[0.3,-0.2,0.1],w->_ma_role_parts(kind,data,w).posterior)
            @test isequal(data,original)
        end
    end
    case=_ma_r2_case(7)
    bound=bind_data(lower_rkppl(case.ast,Set(keys(case.data));conditioned=(:y,)),case.data)
    built=build_kernel(bound)
    u=[0.2sin(i) for i in 1:built.layout.total]
    _ma_compiled_check(bound,built,u,w->_ma_r2_parts(case,built.layout,w).posterior)
end

@testset "unused data and matrix submodel: default compiled reverse" begin
    for (name,data) in _ma_unused_columns(9)
        original=deepcopy(data)
        bound=bind_data(_unbind(_gaussian_plan(9)),data)
        built=build_kernel(bound)
        _ma_compiled_check(bound,built,[0.3,-0.2,0.1],
            w->_ma_role_parts(:unused,data,w).posterior)
        @test isequal(data,original)
    end
    for unused in (ones(10,2),fill("a",9,2),ones(9,0),
            reshape(Union{Missing,Float64}[missing;ones(17)],9,2),
            ones(9,2,2),1.5)
        case=_ma_surface_case(:unused,9)
        case.data[:extra]=unused
        original=deepcopy(case.data)
        bound,built=_ma_surface_model(case)
        u=[0.2sin(i) for i in 1:built.layout.total]
        _ma_compiled_check(bound,built,u,w->_ma_surface_parts(case,built.layout,w).posterior)
        @test isequal(case.data,original)
    end
    case=_ma_surface_case(:raw_location,5)
    ast=quote
        b[axes(X,2)] .~Normal.(0,1)
        X=hcat(ones(length(x1)),x1)
        mu=X*b
        y~_ma_stream(X,1.0)
    end
    case=merge(case,(;ast,coords=[Symbol("b.1"),Symbol("b.2")]))
    bound,built=_ma_surface_model(case)
    u=[0.2sin(i) for i in 1:built.layout.total]
    _ma_compiled_check(bound,built,u,w->_ma_surface_parts(case,built.layout,w).posterior)
end

@testset "matrix values retain backend bodies as observations grow" begin
    for kind in (:nested,:inline,:unused_b,:scalar_s,:plus_scalar,:matrix_scale)
        previous=nothing
        for n in (5,9,13)
            @testset "$kind / $n" begin
                case=_ma_surface_case(kind,n)
                bound,built=_ma_surface_model(case)
                u=[0.2sin(i) for i in 1:built.layout.total]
                sampler,ru=_ma_compiled_check(bound,built,u,w->_ma_surface_parts(case,built.layout,w).posterior)
                modules=Base.invokelatest(_ma_backend_modules,sampler,ru)
                _ma_save_modules(modules,kind,n)
                ops=_ma_backend_ops(modules)
                @test !isempty(ops)
                if previous===nothing
                    previous=ops
                else
                    @test ops==previous
                end
            end
        end
    end
end

@testset "matrix roles and derived R2D2 retain backend bodies" begin
    for kind in (:response,:term,:weights,:scale,:group,:derived_r2)
        previous=nothing
        for n in (7,13,19)
            @testset "$kind / $n" begin
                if kind==:derived_r2
                    case=_ma_r2_case(n)
                    bound=bind_data(lower_rkppl(case.ast,Set(keys(case.data));conditioned=(:y,)),case.data)
                    built=build_kernel(bound)
                    u=[0.2sin(i) for i in 1:built.layout.total]
                    oracle=w->_ma_r2_parts(case,built.layout,w).posterior
                else
                    plan,data=_ma_role_plan(kind,n)
                    bound=bind_data(plan,data)
                    built=build_kernel(bound)
                    u=[0.3,-0.2,0.1]
                    oracle=w->_ma_role_parts(kind,data,w).posterior
                end
                sampler,ru=_ma_compiled_check(bound,built,u,oracle)
                modules=Base.invokelatest(_ma_backend_modules,sampler,ru)
                _ma_save_modules(modules,kind,n)
                ops=_ma_backend_ops(modules)
                @test !isempty(ops)
                if previous===nothing
                    previous=ops
                else
                    @test ops==previous
                end
            end
        end
    end
end

@testset "matrix values: column reads of a per-observation product compile" begin
    # Native column reads are views of the product; compiled tracing reads the
    # same slices (`ReactiveKernels._tensorized_view`).
    for n in (1,6), kind in (:named,:alias,:inline,:submodel,:data_matrix,:data_column,:scale)
        @testset "$kind / $n" begin
            case=_ma_column_case(kind,n)
            original=deepcopy(case.data)
            plan=lower_rkppl(case.ast,Set(keys(case.data));
                conditioned=(:y1,:y2),mod=@__MODULE__)
            bound=bind_data(plan,case.data)
            built=build_kernel(bound)
            u=[0.2sin(i) for i in 1:built.layout.total]
            _ma_compiled_check(bound,built,u,w->case.oracle(built.layout,w))
            @test isequal(case.data,original)
        end
    end
end
