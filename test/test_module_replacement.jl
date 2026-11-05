using ReactiveKernels, Test

# Re-evaluating a module definition (a REPL or test file `include`d again,
# `include_string`) makes a new module object at the old one's path. A kernel
# prepared from the new module's `@kernel` called the OLD module's functions:
# both modules' compiled programs named their module by its path, so they had
# one content id and the first one's compiled code ran for both (snag
# `prepared-kernel-b879ce51`). Each module's kernels must run that module's
# code, as they would in a fresh process.
module ModuleReplacementHost end

replaced_source(factor; threshold = 0.0) = """
module Replaced
using ReactiveKernels
const C = Ref(0)
counted(x) = (C[] += 1; $(factor) * x)
positive(x) = (C[] += 1; x > $(threshold))
@kernel scalar(x::Float64) = begin
    y = counted(x)
    return y
end
@kernel branched(x::Vector{Float64}) = begin
    cells = plate(x) do xi
        positive(xi) ? $(factor) * xi : -xi
    end
    return sum(cells)
end
end
"""

function define_replaced(source)
    include_string(ModuleReplacementHost, source)
    Base.invokelatest(getglobal, ModuleReplacementHost, :Replaced)
end

# A compiled program's body as RuntimeGeneratedFunctions holds it.
generated_body(f) = ReactiveKernels.RuntimeGeneratedFunctions.get_expression(f).args[2]

@testset "a replaced module's kernels run their own module's code" begin
    RK = ReactiveKernels
    old = define_replaced(replaced_source(2))
    old_scalar = prepare(old.scalar)
    @test old_scalar(1.0) == 2.0
    @test old.C[] == 1

    # The identical definition, evaluated again.
    same = define_replaced(replaced_source(2))
    @test same !== old
    @test fullname(same) == fullname(old)
    @test prepare(same.scalar)(1.0) == 2.0
    @test (old.C[], same.C[]) == (1, 1)

    # An edited definition runs its own code.
    edited = define_replaced(replaced_source(3))
    @test prepare(edited.scalar)(1.0) == 3.0
    @test (old.C[], same.C[], edited.C[]) == (1, 1, 1)

    # Kernels of the older modules keep running theirs, also when they are
    # prepared after the replacement.
    @test old_scalar(1.0) == 2.0
    @test prepare(same.scalar)(2.0) == 4.0
    @test (old.C[], same.C[], edited.C[]) == (2, 2, 1)

    # A plate cell whose branch condition calls a module function.
    x = [-1.0, 2.0, 3.0]
    @test prepare(old.branched)(x) == 1.0 + 4.0 + 6.0
    @test old.C[] == 5
    moved = define_replaced(replaced_source(3; threshold = 2.5))
    @test prepare(moved.branched)(x) == 1.0 - 2.0 + 9.0
    @test (old.C[], moved.C[]) == (5, 3)
    @test prepare(old.branched)(x) == 11.0

    # Only programs naming a replaced module carry its generation, and their
    # expressions read as lowered.
    old_f = RK._native_generated_function(old_scalar.f)
    edited_f = RK._native_generated_function(prepare(edited.scalar).f)
    @test typeof(old_f) !== typeof(edited_f)
    for f in (old_f, edited_f)
        body = RK._program_expression(f).args[2]
        @test !occursin("ModuleGenerations", sprint(show, body))
    end
    @test RK._program_expression(edited_f).args[1] == RK._program_expression(old_f).args[1]
    @test generated_body(edited_f).args[1] isa RK._ModuleGenerations
    @test !(generated_body(old_f).args[1] isa RK._ModuleGenerations)
end
