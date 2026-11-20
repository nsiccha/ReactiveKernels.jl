# Count plate log-scale assignments outside and inside row loops. A plate
# emitted as a separate function runs at the depth of its call site.
function scale_log_bindings(ex, in_loop=false)
    ex isa ReactiveKernels._PlateProgram && return scale_log_bindings(ex.ast, in_loop)
    ex isa Expr || return (0, 0)
    ex.head in (:quote, :inert) && return (0, 0)
    in_loop |= ex.head in (:for, :while)
    own = if ex.head === :(=) && ex.args[1] isa Symbol &&
            occursin("plate_", String(ex.args[1])) && occursin("log_scale", String(ex.args[1]))
        in_loop ? (0, 1) : (1, 0)
    else
        (0, 0)
    end
    foldl((a, b) -> a .+ b, (scale_log_bindings(x, in_loop) for x in ex.args); init=own)
end
