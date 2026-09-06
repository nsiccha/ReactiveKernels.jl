# Repeated preparation reuses code while retaining independent state/metadata.
# Run "checks" for bound-value/identity checks or "timings" in a fresh process.
include(joinpath(@__DIR__,"hmc_eight_schools.jl"))

@kernel scaled_accumulating_value(value; gain=2.0,rounds=4) = begin
    advance!(increment)=begin
        for _ in 1:rounds
            value=value+increment*gain
        end
    end
end

function prepared_accumulation(initial,gain;rounds=4)
    parent=fast_native_factory(scaled_accumulating_value,initial;gain,rounds)
    native=compile_native_slots(parent.kernel,parent.state,:advance!;
        endpoints=parent.endpoints,effects=parent.effects,peel_loops=true)
    compile_traced_slots(native)
end

function check_bound_accumulation(program,initial_value,expected)
    f,metadata=program.f,program.metadata
    driver=(state,increment,counts)->f(deepcopy(state),increment,counts,metadata)
    inputs=(Reactant.to_rarray(deepcopy(program.state);track_numbers=true),
        Reactant.to_rarray(1.0;track_numbers=true),
        Reactant.to_rarray((0,0);track_numbers=true))
    compiled=Reactant.compile(driver,inputs;sync=true,donated_args=:none)
    result=compiled(inputs...)
    Float64(only(result.state.values))==expected || error("incorrect bound accumulation")
    Float64(only(inputs[1].values))==initial_value || error("caller state changed")
    println("bound_accumulation=",Float64(only(result.state.values)))
end

function check_code_reuse()
    a=prepared_accumulation(1.0,2.0)
    b=prepared_accumulation(10.0,5.0)
    a.f===b.f || error("equivalent emitted code was not reused")
    Base.invokelatest(check_bound_accumulation,a,1.0,9.0)
    Base.invokelatest(check_bound_accumulation,b,10.0,30.0)
    c=prepared_accumulation(10.0,5.0;rounds=5)
    Base.invokelatest(check_bound_accumulation,c,10.0,35.0)

    # Literal mutable objects are identities, even when their contents match.
    first_authority=[1.0];second_authority=[1.0]
    first_code=TracedSlotCompiler.compile_slot_code(
        Expr(:->,Expr(:tuple),QuoteNode(first_authority)))
    second_code=TracedSlotCompiler.compile_slot_code(
        Expr(:->,Expr(:tuple),QuoteNode(second_authority)))
    Base.invokelatest(first_code)===first_authority || error("first authority replaced")
    Base.invokelatest(second_code)===second_authority || error("distinct authorities merged")

    # A caller's inspection-tree edit cannot corrupt the cached structural key.
    inspection=Expr(:->,:x,Expr(:call,:+,:x,1))
    original=TracedSlotCompiler.slot_code_copy(inspection)
    f=TracedSlotCompiler.compile_slot_code(inspection)
    inspection.args[2].args[end]=2
    same=TracedSlotCompiler.compile_slot_code(original)
    same===f || error("inspection edit corrupted the code cache")
    Base.invokelatest(same,3)==4 || error("inspection edit changed cached code")
    println("independent_state_metadata_and_literal_identity=true")
end

function compile_reuse_case(program,projection,label)
    call=TracedSlotCall(program.f,program.metadata)
    driver=(state,seed,counts)->traced_owned_batch(call,projection,state,seed,counts,1000)
    inputs=traced_slot_inputs(program.state)
    compiled=measured(label) do
        Reactant.compile(driver,inputs;sync=true,donated_args=:none,serializable=true)
    end
    result=compiled(inputs...)
    Int(result.counts[1])==4000 || error("shortened workload")
    check_traced_inputs_unchanged(inputs,program.state)
    println(label," integration_steps=",Int(result.counts[1]),
        " mlir_bytes=",sizeof(compiled.module_string))
end

function measure_code_reuse()
    prototype=build_fast_prototype(compiler_options=(peel_loops=true,))
    reset_native_slots!(prototype.prepared)
    native=prototype.prepared.program
    programs=(compile_traced_slots(native),compile_traced_slots(native),
        compile_traced_slots(native;reuse_code=false),compile_traced_slots(native),
        compile_traced_slots(native;reuse_code=false))
    programs[1].f===programs[2].f===programs[4].f || error("code cache missed")
    all(p->TracedSlotCompiler.slot_code_equal(p.expression,programs[1].expression),programs) ||
        error("emissions differ")
    println("equal_emissions=true distinct_function_types=",
        length(unique(typeof(p.f) for p in programs)))
    prefix=Symbol(native.contexts[2].prefix,:_owned)
    projection=SlotStatic(Tuple(i for (i,pair) in enumerate(programs[1].ordered)
        if first(first(pair))===prefix))
    labels=("first_compile","reuse_existing_code_1","new_equal_code_1",
        "reuse_existing_code_2","new_equal_code_2")
    for (program,label) in zip(programs,labels)
        Base.invokelatest(compile_reuse_case,program,projection,label)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    mode=isempty(ARGS) ? "checks" : only(ARGS)
    if mode=="checks"
        check_code_reuse()
    elseif mode=="timings"
        measure_code_reuse()
    else
        error("expected checks or timings")
    end
end
