import importlib.util, pathlib, sys, numpy as np, jax
jax.config.update('jax_enable_x64',True)
root=pathlib.Path(sys.argv[1])

def regions(jaxpr):
    jaxpr=getattr(jaxpr,'jaxpr',jaxpr)
    result=[]
    for eq in jaxpr.eqns:
        if eq.primitive.name in ('scan','while','cond'):
            result.append((eq.primitive.name, eq.params.get('length'), eq.params.get('reverse'), [str(v.aval) for v in eq.outvars]))
        for value in eq.params.values():
            for v in value if isinstance(value,(list,tuple)) else (value,):
                if hasattr(v,'jaxpr') or hasattr(v,'eqns'): result.extend(regions(v))
    return result
for n in (100,1000):
    env={'np':np};exec((root/f'data_{n}.py').read_text(),env)
    for tag in ('baseline','candidate'):
        spec=importlib.util.spec_from_file_location('model',root/f'{tag}_{n}.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        p=m.prepare_data(*env['RAW'])
        for batch in (False,True):
            fun=jax.grad(lambda u:m.log_density(u,**p))
            x=env['POINT']
            if batch: fun=jax.vmap(fun); x=np.tile(x,(16,1))
            trace=jax.make_jaxpr(fun)(x)
            rs=regions(trace)
            print(n,tag,'batch',batch,'regions',len(rs),'reverse-loops',sum(r[2] is True for r in rs),flush=True)
            (root/f'trace_{tag}_{n}_{batch}.txt').write_text('\n'.join(map(str,rs)))
            if tag=='candidate': assert not any(r[2] is True for r in rs)
