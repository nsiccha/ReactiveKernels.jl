# Per-compilation CPU policy for the small numerical loops emitted here.
# XLA retains its own eligibility checks and emits unsupported/larger loops
# through its ordinary runtime. This does not unroll the mathematical loop.
function cpu_compile_options(; small_loop_bytes::Union{Nothing,Integer}=65536,
        xla_debug_options::NamedTuple=NamedTuple())
    small_loop_bytes === nothing && return (;xla_debug_options)
    small_loop_bytes >= 0 || throw(ArgumentError("small_loop_bytes must be nonnegative"))
    defaults=Reactant.XLA.get_default_debug_options().xla_backend_extra_options
    explicit=get(xla_debug_options,:xla_backend_extra_options,Dict{String,String}())
    extras=merge(defaults,explicit)
    # Existing user settings take precedence over this compiler policy.
    get!(extras,"xla_cpu_small_while_loop_byte_threshold",string(small_loop_bytes))
    (;xla_debug_options=merge(xla_debug_options,(xla_backend_extra_options=extras,)))
end
