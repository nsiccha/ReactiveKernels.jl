using ReactiveKernels
using Test

# GAP-1a — reactive dual-averaging authored through @reactive specialize=true.
# DualAveragingState stays a public NOMINAL wrapper type over the reactive object.

# Local ordinary Nesterov recurrence — parity ORACLE for this test only.
mutable struct _OldDA{T}
    target::T; reg::T; rel::T; off::T; it::T; er::T; cn::T; lc::T; lf::T; cur::T; fin::T
end
function _old_da(x::T; target = 0.8, reg = 0.05, rel = 0.75, off = 10) where {T}
    c = log(T(10)) + log(x)
    _OldDA{T}(T(target), T(reg), T(rel), T(off), one(T), zero(T), c, c, zero(T),
              exp(c), exp(zero(T)))
end
function _old_fit!(s::_OldDA, a)
    s.it += 1
    s.er += (s.target - a - s.er) / (s.it + s.off)
    s.lc = s.cn - sqrt(s.it) / s.reg * s.er
    w = s.it^(-s.rel)
    s.lf += w * (s.lc - s.lf)
    s.cur = exp(s.lc); s.fin = exp(s.lf); s
end

@testset "reactive dual-averaging — nominal type, parity, F32/F64, inference" begin
    for initial in (0.1, 1.0, 2.5)
        oracle = _old_da(initial)
        state = dual_averaging_state(initial)
        @test state isa DualAveragingState                 # public nominal type
        @test state.current ≈ oracle.cur
        for a in (0.9, 0.6, 0.85, 0.3, 0.95, 0.7, 0.8, 0.5, 0.99, 0.75, 0.1)
            _old_fit!(oracle, a)
            r = fit!(state, a)
            @test r === state                              # fit! returns the wrapper
            @test state.current == oracle.cur              # exact parity
            @test state.final == oracle.fin
            @test state.iteration == oracle.it
        end
    end

    # Generic over precision: Float32 stays Float32 (concrete).
    o32 = _old_da(0.1f0); s32 = dual_averaging_state(0.1f0)
    for a in (0.9f0, 0.4f0, 0.85f0)
        _old_fit!(o32, a); fit!(s32, a)
        @test s32.current === o32.cur && s32.final === o32.fin   # === (same Float32 bits)
    end
    @test s32.current isa Float32 && s32.final isa Float32

    # Concrete inference behind a function barrier.
    s = dual_averaging_state(0.5)
    @test (@inferred((x -> x.current)(s)))::Float64 == s.current
    @test (@inferred((x -> x.final)(s)))::Float64 == s.final
    @test :current in propertynames(s) && :iteration in propertynames(s)

    # Mutable-compat: manual public field assignment forwards to the reactive
    # source, invalidates the derived current/final, and returns the assigned value.
    m = dual_averaging_state(0.5)
    fit!(m, 0.9)                                    # make error != 0 so iteration matters
    before = m.current
    m.iteration = m.iteration + 5.0                 # mutate an accumulator source
    @test m.current != before                       # derived recomputed reactively
    @test (m.target = 0.7) == 0.7                    # Julia assignment return semantics
    @test :target in propertynames(m, true)
end
