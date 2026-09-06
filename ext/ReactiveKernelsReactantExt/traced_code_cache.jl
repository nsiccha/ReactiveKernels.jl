# Cache emitted functions, never prepared state, metadata, or XLA executables.
# Syntax is compared structurally; literal leaves retain exact Julia identity.
# In particular, equal contents do not merge distinct mutable authorities.
struct SlotCodeKey
    expression::Expr
    fingerprint::UInt
end
slot_code_hash(value,h::UInt)=hash(objectid(value),h)
function slot_code_hash(value::Expr,h::UInt)
    h=hash(length(value.args),hash(value.head,h))
    for arg in value.args
        h=slot_code_hash(arg,h)
    end
    h
end
slot_code_equal(a,b)=a===b
function slot_code_equal(a::Expr,b::Expr)
    a.head===b.head && length(a.args)==length(b.args) &&
        all(slot_code_equal(x,y) for (x,y) in zip(a.args,b.args))
end
Base.hash(key::SlotCodeKey,h::UInt)=hash(key.fingerprint,h)
Base.isequal(a::SlotCodeKey,b::SlotCodeKey)=slot_code_equal(a.expression,b.expression)

slot_code_copy(value)=value
slot_code_copy(value::Expr)=Expr(value.head,(slot_code_copy(a) for a in value.args)...)
const SLOT_CODE_CACHE=Dict{SlotCodeKey,Function}()
const SLOT_CODE_CACHE_LOCK=ReentrantLock()

function compile_slot_code(expression;reuse_code=true)
    reuse_code || return Core.eval(@__MODULE__,slot_code_copy(expression))
    query=SlotCodeKey(expression,slot_code_hash(expression,zero(UInt)))
    lock(SLOT_CODE_CACHE_LOCK) do
        existing=get(SLOT_CODE_CACHE,query,nothing)
        existing===nothing || return existing
        # The returned inspection AST and Julia's evaluator each get their own
        # syntax tree, so neither can mutate the dictionary's structural key.
        key=SlotCodeKey(slot_code_copy(expression),query.fingerprint)
        f=Core.eval(@__MODULE__,slot_code_copy(expression))
        SLOT_CODE_CACHE[key]=f
        f
    end
end
