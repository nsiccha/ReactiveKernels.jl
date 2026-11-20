# Enzyme-only reproducer for reverse differentiation of a checked writer.
# On Julia 1.10.12 with registered Enzyme 0.13.221 the alias test raises
# EnzymeNoDerivativeError: cannot reverse gc preserve region with several ends.
using Enzyme, Test

@inline function overlaps(a, b)
    isempty(a) && return false
    isempty(b) && return false
    GC.@preserve a b begin
        pa, pb = UInt(pointer(a)), UInt(pointer(b))
        pa <= pb ? pb - pa < sizeof(a) : pa - pb < sizeof(b)
    end
end

function checked!(dst, x, h, g)
    length(dst) == length(x) == length(g) || error("length mismatch")
    if overlaps(dst, x) || overlaps(dst, g)
        copyto!(dst, x .+ (h/2) .* g)
    else
        dst .= x .+ (h/2) .* g
    end
    nothing
end

function test_checked!(f)
    x, g, dst = [2.0, 3.0], [5.0, 7.0], zeros(2)
    dx, dg = zeros(2), zeros(2)
    partials = only(Enzyme.autodiff(Reverse, f, Const, Duplicated(dst, ones(2)),
        Duplicated(x, dx), Active(0.5), Duplicated(g, dg)))
    @test dst == x .+ 0.25 .* g
    @test dx == ones(2)
    @test dg == fill(0.25, 2)
    @test partials[3] == sum(g) / 2
end
test_checked!(checked!)
