# A derivative rule's execution tables: what a native call of the rule reads.
# A cut that runs a plate, a scan or a branch keeps its source plan beside
# them for source backends (`_RuleRegion`, RK 0.1.293), so such a rule is not
# an isbits value as a whole. Source plans are build-time metadata; the tables
# passed to compiled native regions must still be plain values.
_rule_execution_isbits(x) = isbits(x)
_rule_execution_isbits(x::Tuple) = all(_rule_execution_isbits, x)
_rule_execution_isbits(r::ReactiveKernels._RuleRegion) =
    isbits((r.f, r.constants, r.callables))
_rule_execution_isbits(rule::ReactiveKernels.DerivativeRule) =
    _rule_execution_isbits((rule.primal_ops, rule.forward_ops, rule.reverse_ops,
                            rule.staged_ops, rule.lane))
