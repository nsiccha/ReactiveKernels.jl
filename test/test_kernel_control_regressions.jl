# Regressions for the control-compiler fixes surfaced while lowering the REAL nuts_state (which the synthetic
# adversaries never exercised): _subst completeness + source-order _argmap (commit 6215c6d) and evaluation of
# a discarded non-call return expression (commit e0393fc). Each asserts behavior that FAILS on the old code.
using Test
using ReactiveKernels
const RKC = ReactiveKernels

# minimal IR node builders
_fr(name, pos=1) = RKC._FormalRef(name, pos, :pos)
_lit(v) = RKC._Lit(v)
_sf(path...) = RKC._SelfField(path)
_idx(base, idxs...) = RKC._Index(base, idxs)
_pw(target, rhs) = RKC._PlaceWrite(target, :self, (:proposals,), nothing, rhs, false)
# collect every _FormalRef name remaining below a node (should be empty after a complete substitution)
function _residual_formals(x, acc=Symbol[])
    if x isa RKC._FormalRef; push!(acc, x.arg)
    elseif x isa Tuple || x isa AbstractVector; for e in x; _residual_formals(e, acc); end
    elseif x isa Pair; _residual_formals(x.second, acc)
    elseif x isa RKC._MExpr || x isa RKC._MStmt; for f in fieldnames(typeof(x)); _residual_formals(getfield(x, f), acc); end
    end
    acc
end

@testset "control regression — _subst reaches FormalRefs inside PlaceSwap / Index / kw Pair (6215c6d)" begin
    # swapproposal!-shaped: (proposals[i], proposals[j]) = (proposals[j], proposals[i]) as a _PlaceSwap
    swap = RKC._PlaceSwap((
        _pw(_idx(_sf(:proposals), _fr(:i)), _idx(_sf(:proposals), _fr(:j))),
        _pw(_idx(_sf(:proposals), _fr(:j)), _idx(_sf(:proposals), _fr(:i))),
    ))
    out = RKC._subst(swap, Dict{Symbol,Any}(:i => _lit(3), :j => _lit(7)))
    @test isempty(_residual_formals(out))                 # OLD _subst left i/j unsubstituted inside the swap
    @test out isa RKC._PlaceSwap
    # a formal nested in an _Index inside a _PlaceWrite target
    @test isempty(_residual_formals(RKC._subst(_pw(_idx(_sf(:proposals), _fr(:i)), _lit(0)),
                                               Dict{Symbol,Any}(:i => _lit(9)))))
    # a formal inside a keyword Pair value
    p = (:stepsize => _fr(:h))
    @test isempty(_residual_formals(RKC._subst(p, Dict{Symbol,Any}(:h => _lit(0.1)))))
end

# mock callee/call for _argmap (duck-typed: _argmap reads callee.formals, callee.id.name, call.pos/.kw/.name)
_formal(name, kind, required, default=nothing) = RKC._Formal(name, kind, required, nothing, default)
_mid(name) = RKC.MethodId(name, 1, 0, 0, (), (), (), ())
_callee(name, fs) = (id = _mid(name), formals = fs)
_call(name, pos, kw=()) = RKC._Call(name, (), RKC._SelfRef(), Tuple(pos), Tuple(kw))  # target::_MExpr

@testset "control regression — source-order _argmap: left-to-right defaults + reject missing/extra/dup (6215c6d)" begin
    # b's default references the earlier formal a — must resolve left-to-right through the bound actual
    callee = _callee(:f, [_formal(:a, :pos, true), _formal(:b, :pos, false, RKC._OpCall(GlobalRef(Base, :+), (_fr(:a), _lit(1)), (), false))])
    fmap = RKC._argmap(callee, _call(:f, [_lit(10)]))
    @test isempty(_residual_formals(fmap[:b]))             # a-in-default replaced by the actual (10)
    @test haskey(fmap, :a) && haskey(fmap, :b)
    # supplied positional wins over the default
    fmap2 = RKC._argmap(callee, _call(:f, [_lit(10), _lit(20)]))
    @test fmap2[:b] == _lit(20)
    # missing required positional rejects
    @test_throws ErrorException RKC._argmap(callee, _call(:f, RKC._Lit[]))
    # extra positional rejects
    @test_throws ErrorException RKC._argmap(callee, _call(:f, [_lit(1), _lit(2), _lit(3)]))
    # duplicate keyword rejects
    kwc = _callee(:g, [_formal(:k, :kw, false, _lit(0))])
    @test_throws ErrorException RKC._argmap(kwc, _call(:g, RKC._Lit[], [:k => _lit(1), :k => _lit(2)]))
    # unknown keyword rejects
    @test_throws ErrorException RKC._argmap(kwc, _call(:g, RKC._Lit[], [:zzz => _lit(1)]))
end

@testset "control regression — build_region! evaluates a DISCARDED non-call return expression (e0393fc)" begin
    # a method whose tail is `return <effect-expr>` with the value discarded (ret_val = nothing): the
    # expression must still be emitted as an effect, not dropped (this is how step!'s terminal
    # `return copy!!(init, proposals[end])` was silently lost).
    bb = RKC.BB(RKC.Blk[], 1)
    effect = RKC._OpCall(GlobalRef(Base, :identity), (_lit(42),), (), false)  # stand-in nontrivial expr
    RKC.build_region!(bb, Any[RKC._Return(effect)], 0, 0, nothing, Dict{Int,Any}(), Set{Int}())
    emitted = reduce(vcat, [collect(b.effects) for b in bb.blks]; init = Any[])
    @test any(e -> e isa RKC._ExprStmt && e.expr === effect, emitted)   # OLD code dropped it entirely
    # value-bound case ASSIGNS (a well-formed path-tuple lhs) and does not double-emit as a bare effect
    bb2 = RKC.BB(RKC.Blk[], 1)
    RKC.build_region!(bb2, Any[RKC._Return(effect)], 0, 0, :ret, Dict{Int,Any}(), Set{Int}())
    emitted2 = reduce(vcat, [collect(b.effects) for b in bb2.blks]; init = Any[])
    la = findfirst(e -> e isa RKC._LocalAssign, emitted2)
    @test la !== nothing && emitted2[la].lhs === (:ret,)   # lhs is a path-tuple, not a bare Symbol
    @test !any(e -> e isa RKC._ExprStmt && e.expr === effect, emitted2)
end
