import{_ as s,o as a,c as n,a6 as i}from"./chunks/framework.BTpfZnOf.js";const k=JSON.parse('{"title":"JAX translations","description":"","frontmatter":{},"headers":[],"relativePath":"jax-translations.md","filePath":"jax-translations.md","lastUpdated":null}'),t={name:"jax-translations.md"};function l(p,e,r,o,h,d){return a(),n("div",null,[...e[0]||(e[0]=[i(`<h1 id="JAX-translations" tabindex="-1">JAX translations <a class="header-anchor" href="#JAX-translations" aria-label="Permalink to &quot;JAX translations {#JAX-translations}&quot;">​</a></h1><p>Companion to <a href="./jax">JAX source</a>: statistics reductions and package mappings. The Julia that <code>prepare_data</code> runs is in <a href="./jax-host">JAX host preparation</a>.</p><h2 id="Factorizations-and-triangular-solves" tabindex="-1">Factorizations and triangular solves <a class="header-anchor" href="#Factorizations-and-triangular-solves" aria-label="Permalink to &quot;Factorizations and triangular solves {#Factorizations-and-triangular-solves}&quot;">​</a></h2><p>For a dense <code>Float32</code> or <code>Float64</code> matrix <code>A</code>, use <code>F = qr(A)</code>, <code>R = F.R</code> and <code>Q = Matrix(F.Q)</code>. Machine integers promote to floating point as in Julia. The two projections share one unpivoted QR factorization. For an <code>m × n</code> matrix, <code>Q</code> has <code>min(m, n)</code> orthonormal columns and <code>Q * R</code> reconstructs <code>A</code>. On a tall matrix the explicit <code>Matrix</code> conversion selects thin Q from Julia&#39;s implicit square operator. QR signs follow the numerical library. For a full-column-rank matrix, make the diagonal of R positive by multiplying each column of Q and the corresponding row of R by the sign of that diagonal entry. JAX can differentiate both factors on tall, full-column-rank inputs. On Julia 1.10.12, Enzyme 0.13.221 fails to differentiate the native signed factor sum in <code>LinearAlgebra.getproperty</code>. A device cell with a padded matrix of variable size cannot yet export its QR.</p><p><code>rk_cholesky_lower(A)</code> reads only the lower triangle, including when unused upper entries are nonfinite. It checks for square input and positive pivots, with runtime checks masked by the enclosing branch or plate. Empty factors and padded leading square blocks are supported. <code>diag(L)</code> reads the diagonal. Data-only factorizations and solves run in <code>prepare_data</code>. Arithmetic follows the module&#39;s <code>float_type</code>.</p><p>Use <code>LowerTriangular(A) \\ b</code> or <code>UpperTriangular(A) \\ b</code> with a square matrix and a vector or matrix right-hand side. Only the selected triangle contributes. <code>norm(v)</code> computes a real vector&#39;s Euclidean norm with scaling for large and small magnitudes.</p><h2 id="Symmetric-eigendecomposition" tabindex="-1">Symmetric eigendecomposition <a class="header-anchor" href="#Symmetric-eigendecomposition" aria-label="Permalink to &quot;Symmetric eigendecomposition {#Symmetric-eigendecomposition}&quot;">​</a></h2><p><code>F = eigen(Symmetric(A))</code> exports one symmetric eigendecomposition of a dense real matrix. <code>F.values</code> contains ascending eigenvalues and the columns of <code>F.vectors</code> are orthonormal. Use <code>eigvals(Symmetric(A))</code> for eigenvalues alone. The default and <code>:U</code> use the upper triangle, and <code>:L</code> uses the lower triangle. When comparing Julia and JAX, compare reconstructed matrices or eigenspace projectors because signs and bases within repeated eigenspaces can differ. The RK-owned <code>rk_symmetric_eigvals</code> and <code>rk_symmetric_eigvecs</code> use the upper triangle and retain their authored derivative rules. A dynamically padded matrix cannot be passed to the eigensolver because the padding would contribute additional eigenvalues.</p><h2 id="Index-of-a-vector-maximum" tabindex="-1">Index of a vector maximum <a class="header-anchor" href="#Index-of-a-vector-maximum" aria-label="Permalink to &quot;Index of a vector maximum {#Index-of-a-vector-maximum}&quot;">​</a></h2><p>Use <code>grid[argmax(signal)]</code> to select the grid entry at the first maximum of a nonempty real vector. <code>argmax</code> returns a one-based integer index in both device code and host preparation. The grid can be in any order. For example, <code>signal = [1.0, 4.0, 4.0]</code> selects the second grid entry. As in Julia, the first NaN wins and positive zero sorts above negative zero. An evaluated <code>argmax</code> on an empty vector raises an error.</p><h2 id="Statistics-reductions" tabindex="-1">Statistics reductions <a class="header-anchor" href="#Statistics-reductions" aria-label="Permalink to &quot;Statistics reductions {#Statistics-reductions}&quot;">​</a></h2><p>With Statistics loaded (<code>ReactiveKernelsStatisticsExt</code>, RK 0.1.262), <code>mean(x)</code>, <code>mean(f, x)</code>, <code>var(x)</code> and <code>std(x)</code> translate, with the keywords <code>dims</code> (one integer known while JAX traces, or <code>:</code>), and for <code>var</code> and <code>std</code> also <code>corrected</code> (a literal <code>true</code> or <code>false</code>) and <code>mean</code>. Each is the core <code>sum</code>, <code>length</code> and <code>size</code> translation it composes, <code>sum(x) / length(x)</code> and <code>sum(abs2.(x .- m)) / (n - corrected)</code>, so a padded row or a <code>dims</code> reduction reads only its valid entries and <code>prepare_data</code> takes NumPy&#39;s spelling. Over <code>dims</code> the reduced axis keeps length one, as in Julia. An empty argument, or one entry with <code>corrected</code>, gives Julia&#39;s NaN. Before, Statistics&#39; own methods, which branch, were refused as &quot;the function <code>_mean</code> has loops or branches&quot;.</p><h2 id="Package-translations" tabindex="-1">Package translations <a class="header-anchor" href="#Package-translations" aria-label="Permalink to &quot;Package translations {#Package-translations}&quot;">​</a></h2><p>A package extension can map its own callable during source emission. Julia execution is unchanged; ordinary <code>jax.grad</code> differentiates the emitted primal without a registered derivative rule.</p><p>Use <code>ReactiveKernels._jax_elementwise(::typeof(f))</code> for scalar calls and <code>f.(xs)</code>. Return a Python callable name, <code>name =&gt; source</code> defining a module-level helper, or <code>nothing</code> for fallback translation. This does <strong>not</strong> register ordinary array calls: Julia&#39;s <code>exp(A)</code> is a matrix exponential; <code>exp.(A)</code> applies scalar exponentials.</p><p>For a positional whole-function call, including an array-valued helper, define <code>_jax_function_call(ctx::_JaxContext, ::typeof(my_function), args::Vector{_PyVal})</code>. Qualify RK names with <code>ReactiveKernels</code>, as in the executable example below.</p><p>These underscored names form the package extension interface. The context and vector annotations make the method more specific than RK&#39;s generic method. Selection precedes recorded/provider source and lowered-code translation, but follows applicable scalar/elementwise mappings. This hook registers positional calls, not keyword handling.</p><p>Each <code>args[i]</code> describes an argument: <code>code</code> is its Python expression, <code>T</code> its Julia type. Parenthesize expressions when composing operators. Return a <code>_PyVal</code>, not a string or <code>nothing</code>; call <code>_jax_traceable_or_lowered(ctx, f, args)</code> for the generic hook&#39;s source fallback.</p><p><code>_jax_result(f, args, expression)</code> infers the native result type from <code>f</code> and the argument types, preserving matrix rank for subsequent multiplication. The expression must implement that native result and shape. Shapes and numerical inputs remain runtime expressions, not sampled data. For elementwise whole-array calls preserving the arrays&#39; axes, <code>_jax_elementwise_result(f, args, expression)</code> additionally carries padded-axis metadata.</p><p>For changed shape or ragged structure, the full constructor is <code>_PyVal(code, T, static, extent, limit, padded)</code>. Short constructors default <code>static=false</code>, <code>extent=&quot;&quot;</code>, <code>limit=&quot;&quot;</code>, and <code>padded=0</code>.</p><table tabindex="0"><thead><tr><th style="text-align:right;">Field</th><th style="text-align:right;">Meaning</th></tr></thead><tbody><tr><td style="text-align:right;"><code>static::Bool</code></td><td style="text-align:right;">Known as a Python value while JAX traces; it can decide Python control flow. A JAX array is not static merely because its inputs are constants.</td></tr><tr><td style="text-align:right;"><code>extent::String</code></td><td style="text-align:right;">Valid length of a padded row, or lengths of a ragged collection; empty for an unpadded value.</td></tr><tr><td style="text-align:right;"><code>limit::String</code></td><td style="text-align:right;">Static upper-bound expression of a data-dependent integer, for loops and allocation; empty when absent.</td></tr><tr><td style="text-align:right;"><code>padded::Int</code></td><td style="text-align:right;">Bitmask of padded axes (bit <code>k-1</code> for axis <code>k</code>), implicit for a vector; for a tuple, of its fields with a length.</td></tr></tbody></table><p><code>_jax_result</code> carries an input extent to an array result without deriving new lengths or padding axes. Operations must account for padding and return their actual result metadata: copying input metadata cannot describe arbitrary reductions, reshapes or matrix functions. The matrix-exponential example takes an ordinary dense matrix, not a padded ragged row.</p><h3 id="Host-preparation-and-helpers" tabindex="-1">Host preparation and helpers <a class="header-anchor" href="#Host-preparation-and-helpers" aria-label="Permalink to &quot;Host preparation and helpers {#Host-preparation-and-helpers}&quot;">​</a></h3><p>The hook also serves <code>prepare_data</code>, where <code>ctx.host !== nothing</code>. RK rewrites <code>jnp.*</code> to <code>np.*</code>, <code>jsp.*</code> to <code>ssp.*</code> (SciPy special functions), and <code>jax.scipy.linalg.*</code> to <code>scipy.linalg.*</code>. <code>_jax_result</code> uses these host spellings for all-static operands too, keeping shape/literal arithmetic usable in Python guards. Branch on <code>ctx.host</code> if different host mathematics is needed. Leftover device-only constructs (<code>jax.lax</code>, <code>.at[...]</code>) cannot form host preparation: the recipe stays on device or is reported untranslated, according to the preparation boundary.</p><p>Define Python helpers with <code>_jax_helper!(ctx, name, source)</code> and use its <strong>returned name</strong>. It emits each name once; host mode emits a NumPy twin with an <code>_np</code> suffix. Names must be unique across the module. For all-static operands in device mode, use <code>_host_helper!(ctx, name, source)</code> when a host spelling exists: <code>_jax_result</code> cannot discover JAX calls inside separate helper bodies. <code>_jax_elementwise</code> pairs select automatically. Custom helpers must not mutate inputs; RK cannot enforce ownership on arbitrary Python text or translate independently written derivatives.</p><h3 id="Executable-scalar-and-matrix-example" tabindex="-1">Executable scalar and matrix example <a class="header-anchor" href="#Executable-scalar-and-matrix-example" aria-label="Permalink to &quot;Executable scalar and matrix example {#Executable-scalar-and-matrix-example}&quot;">​</a></h3><p>This file defines owned helpers, mappings and kernels. The loop in <code>triple</code> requires its mapping for translation; matrix exponential demonstrates whole-array mathematics distinct from elementwise mapping.</p><div class="vp-raw-html"><div class="markdown"><pre><code class="language-julia">module JAXTranslationExample

using ReactiveKernels
using LinearAlgebra
const RK &#61; ReactiveKernels

# A package owns these Julia functions; its RK extension owns the mappings.
function triple&#40;x&#41;
    result &#61; zero&#40;x&#41;
    for _ in 1:3
        result &#43;&#61; x
    end
    result
end

RK._jax_elementwise&#40;::typeof&#40;triple&#41;&#41; &#61; &quot;_example_triple&quot; &#61;&gt; &quot;&quot;&quot;
def _example_triple&#40;x&#41;:
    return 3 * x
&quot;&quot;&quot;

# An ordinary call on an array has its own translation. Here its mathematics
# happens to be elementwise too, so preserve the input&#39;s padded axes as well.
function RK._jax_function_call&#40;ctx::RK._JaxContext, ::typeof&#40;triple&#41;,
                               args::Vector&#123;RK._PyVal&#125;&#41;
    RK._jax_elementwise_result&#40;triple, args, &quot;3 * &#40;&quot; * only&#40;args&#41;.code * &quot;&#41;&quot;&#41;
end

matrix_exponential&#40;A::AbstractMatrix&#41; &#61; exp&#40;A&#41;
function RK._jax_function_call&#40;ctx::RK._JaxContext, ::typeof&#40;matrix_exponential&#41;,
                               args::Vector&#123;RK._PyVal&#125;&#41;
    RK._jax_result&#40;matrix_exponential, args,
                   &quot;jax.scipy.linalg.expm&#40;&quot; * only&#40;args&#41;.code * &quot;&#41;&quot;&#41;
end

# Types supplied to jax_source select scalar, vector, or matrix inputs.
@kernel mapped&#40;x&#41; &#61; begin
    y &#61; triple&#40;x&#41;
    total &#61; sum&#40;y&#41;
    return y, total
end

# The matrix result must keep its rank for the following matrix-vector product.
@kernel mapped_product&#40;A::Matrix&#123;Float64&#125;, v::Vector&#123;Float64&#125;&#41; &#61; begin
    B &#61; triple&#40;A&#41;
    total &#61; sum&#40;B * v&#41;
    return total
end

@kernel prepared&#40;scale::Float64, x&#41; &#61; begin
    y &#61; triple&#40;x&#41;
    total &#61; scale * sum&#40;y&#41;
    return total
end

@kernel exponential&#40;scale::Float64, A::Matrix&#123;Float64&#125;&#41; &#61; begin
    E &#61; matrix_exponential&#40;A&#41;
    total &#61; scale * sum&#40;E&#41;
    return total
end

end # module
</code></pre>
</div></div><p>Emit scalar and matrix device calls, and a data-only matrix exponential in host preparation:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ex </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> JAXTranslationExample</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">scalar_source </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> jax_source</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ex</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mapped; have</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,), want</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:total</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, types</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Float64,))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">matrix_source </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> jax_source</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ex</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mapped_product; have</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:A</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:v</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), want</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:total</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">host_source </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> jax_source</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ex</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">exponential; have</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:scale</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:A</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), want</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:total</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:A</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,))</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(host_source)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">&quot;&quot;&quot;Generated by ReactiveKernels.jax_source; do not edit.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">import scipy.linalg</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">import numpy as np</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">import scipy.special as ssp</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">from typing import Any</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">import jax</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">import jax.numpy as jnp</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">import jax.scipy.linalg</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">import jax.scipy.special as jsp</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">from jax.typing import ArrayLike</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">jax.config.update(&quot;jax_enable_x64&quot;, True)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">INPUTS = (&quot;scale&quot;, &quot;E&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">PREPARED = {&quot;E&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 2, &quot;ragged&quot;: False}}</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def prepare_data(A: ArrayLike) -&gt; dict:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The functions&#39; data inputs, computed from the raw data once per dataset.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    Pass the data by the names the model reads them; a list of arrays is a ragged</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    column. Call the other functions with the result: \`f(u, **prepared)\`.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    A = np.asarray(A, dtype=np.float64)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    E = scipy.linalg.expm(A)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return {&quot;E&quot;: np.asarray(E, dtype=np.float64)}</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def kernel(scale: ArrayLike, **prepared: ArrayLike) -&gt; jax.Array:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;Compute \`total\` from the inputs.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    E = prepared[&quot;E&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    E = jnp.asarray(E)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    total = scale * jnp.sum(E)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return jnp.asarray(total, dtype=jnp.float64)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def _rk_batched_gradient(density, positions, data):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return jax.vmap(jax.value_and_grad(lambda p: density(p, **data)))(positions)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def bind_gradient(density, prepared, *, device=None, memory_budget_bytes=None):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;Bind a batched value-and-gradient to prepared data on one device.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    Call the result with a floating-point array whose leading axis is chains.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    The first call at a shape and dtype runs one feasible layout. An explicit</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    gradient.calibrate(positions, repeats=3) measures eligible layouts and</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    installs its winner for subsequent calls on this binding. It returns the</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    timing rows, winner first (empty without a layout selector). Overlapping</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    pilot timings trigger 12 interleaved rounds among the close contenders.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    memory_budget_bytes limits the executable&#39;s planned device buffer bytes.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    The device allocator, other live arrays and backend overhead are separate.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    from copy import deepcopy</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    from functools import partial</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    from operator import index</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    from threading import RLock</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if memory_budget_bytes is not None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        memory_budget_bytes = index(memory_budget_bytes)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if memory_budget_bytes &lt; 0:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            raise ValueError(&quot;memory_budget_bytes must be nonnegative&quot;)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    # Calibration and execution must see the same bound values, even if the</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    # caller later edits a host array. JAX arrays already have value semantics.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    immutable = {id(x): x for x in jax.tree_util.tree_leaves(prepared) if isinstance(x, jax.Array)}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    data = deepcopy(prepared, immutable)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    measure, layout_variants = None, None</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    confirm_layouts = None</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    programs = {}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    lock = RLock()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def signature(positions):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if not hasattr(positions, &quot;dtype&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            positions = np.asarray(positions)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if not positions.shape or positions.shape[0] == 0:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            raise ValueError(&quot;positions must have a nonempty leading chain axis&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        dtype = jax.dtypes.canonicalize_dtype(positions.dtype)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if not jnp.issubdtype(dtype, jnp.floating):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            raise TypeError(&quot;positions must have a real floating-point dtype&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        target = device</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if target is None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if isinstance(positions, jax.Array):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                devices = positions.devices()</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                if len(devices) != 1:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    raise ValueError(&quot;specify a device for sharded positions&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                target = next(iter(devices))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            else:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                # Respect jax.default_device for host positions.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                target = jax.config.jax_default_device</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                if target is None or isinstance(target, str):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    target = jax.local_devices(backend=target)[0]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return (target, tuple(positions.shape), dtype, getattr(positions, &quot;weak_type&quot;, False)), positions</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def compile_program(chosen, key):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        target, shape, dtype, weak = key</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        sharding = jax.sharding.SingleDeviceSharding(target)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        function = jax.tree_util.Partial(density)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        batch = jax.jit(_rk_batched_gradient, in_shardings=sharding)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        abstract_positions = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding, weak_type=weak)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        program = batch.lower(function, abstract_positions, chosen).compile()</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if memory_budget_bytes is not None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            try:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                stats = program.memory_analysis()</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            except NotImplementedError as error:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                raise MemoryError(&quot;backend cannot estimate gradient buffer bytes&quot;) from error</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            size = getattr(stats, &quot;total_allocation_bytes&quot;, None)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if size is None or size &lt; 0:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                raise MemoryError(&quot;backend did not report planned device buffer bytes&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if size &gt; memory_budget_bytes:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                raise MemoryError(f&quot;gradient requires {size} planned bytes, budget is {memory_budget_bytes}&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        device_data = jax.device_put(chosen, target)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return lambda positions: program(function, positions, device_data)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    supplied = (&quot;supplied&quot;, None, None, None)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def alternatives():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if layout_variants is not None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            for name, groups, sequential, mode, chosen, _ in layout_variants(data, None):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                yield (name, groups, sequential, mode), chosen</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def candidates():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        yield supplied, data</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        for layout, chosen in alternatives():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if chosen is not data:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                yield layout, chosen</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def gradient(positions):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        key, positions = signature(positions)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        with lock:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            _, program = programs.get(key, (supplied, None))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if program is None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                failure = None</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                for layout, chosen in candidates():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    try:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        program = compile_program(chosen, key)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        result = jax.block_until_ready(program(jax.device_put(positions, key[0])))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    except (MemoryError, jax.errors.JaxRuntimeError) as error:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        if not isinstance(error, MemoryError) and &quot;RESOURCE_EXHAUSTED&quot; not in str(error):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                            raise</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        failure = str(error)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        program = None</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        continue</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    programs[key] = layout, program</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    return result</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                raise MemoryError(f&quot;no gradient layout fit on {key[0]}: {failure}&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return program(jax.device_put(positions, key[0]))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def same_as_supplied(chosen):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        # Selectors share the bound model data. Compare only changed layout</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        # metadata, including the members and widths of a caller&#39;s own groups.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        for name, value in chosen.items():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            original = data.get(name)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if value is original:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                continue</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if name == &quot;_rk_plate_layout&quot;:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                value = {k: v for k, v in value.items() if v is not None}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                original = {k: v for k, v in (original or {}).items() if v is not None}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            leaves, tree = jax.tree_util.tree_flatten(value)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            others, other_tree = jax.tree_util.tree_flatten(original)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if tree != other_tree or any(a is not b and not np.array_equal(a, b)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                         for a, b in zip(leaves, others)):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                return False</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return True</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def calibration_candidates(active):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        found = False</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        for layout, chosen in alternatives():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            current = same_as_supplied(chosen) if active == supplied else layout == active</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            found = found or current</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            yield layout, chosen, current</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if not found:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            # A caller can supply a partition outside the default search.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            yield supplied, data, True</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def confirm(finalists, rows, steps):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        confirmed = confirm_layouts(finalists, rows, steps, 12)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if not confirmed:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return None</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        confirmed.sort(key=lambda k: rows[k][&quot;confirmation_median_seconds&quot;])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        current = next((k for k in confirmed if rows[k][&quot;current&quot;]), None)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if current is None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return confirmed[0], &quot;confirmation&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        baseline = rows[current][&quot;confirmation_samples_seconds&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        median = rows[current][&quot;confirmation_median_seconds&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        for k in confirmed:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if rows[k][&quot;confirmation_median_seconds&quot;] &gt;= median:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                break</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            samples = rows[k][&quot;confirmation_samples_seconds&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if sum(a &lt; b for a, b in zip(samples, baseline)) &gt;= 10:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                return k, &quot;paired_win&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return current, &quot;retain_current&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def select(rows, steps):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        order = sorted(steps, key=lambda k: rows[k][&quot;median_seconds&quot;])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        quartiles = {k: np.percentile(rows[k][&quot;samples_seconds&quot;], (25, 75)) for k in order}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        while order:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            low, high = quartiles[order[0]]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            finalists = [k for k in order if quartiles[k][0] &lt;= high and low &lt;= quartiles[k][1]]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if len(finalists) == 1:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                return finalists[0], &quot;pilot&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            winner = confirm(finalists, rows, steps)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            if winner is not None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                return winner</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            # Every finalist ran out of memory. Compare the remaining pilots.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            order = [k for k in order if k in steps]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        errors = &quot;, &quot;.join(dict.fromkeys(row[&quot;error&quot;] for row in rows.values()))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        raise RuntimeError(f&quot;bind_gradient: no candidate ran on this device: {errors}&quot;)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def calibrate(positions, repeats=3):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        repeats = index(repeats)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if repeats &lt; 1:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            raise ValueError(&quot;repeats must be positive&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if measure is None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            gradient(positions)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return []</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        key, positions = signature(positions)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        with lock:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            active, program = programs.get(key, (supplied, None))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            batch = jax.device_put(positions, key[0])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            rows, steps = {}, {}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            for layout, chosen, current in calibration_candidates(active):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                def build():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    if current and program is not None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        return partial(program, batch)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    bound_data = data if current and active == supplied else chosen</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    executable = compile_program(bound_data, key)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    return partial(executable, batch)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                step, row = measure(layout, len(positions), build, repeats)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                row.update(current=current, confirmation_samples_seconds=[], confirmation_median_seconds=None)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                rows[layout] = row</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                if step is not None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    steps[layout] = step</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    row[&quot;seconds&quot;] = row[&quot;median_seconds&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                row[&quot;fit_seconds&quot;] = row[&quot;seconds&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            winner, reason = select(rows, steps)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            programs[key] = winner, steps[winner].func</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            first = rows.pop(winner)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            first[&quot;selection_reason&quot;] = reason</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return [first, *sorted(rows.values(), key=lambda r: (r[&quot;error&quot;] is not None, r[&quot;seconds&quot;]))]</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    gradient.calibrate = calibrate</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return gradient</span></span></code></pre></div><p>After <code>write(&quot;matrix_model.py&quot;, host_source)</code>, run <code>python examples/jax_translations.py matrix_model.py</code> with a Python that has JAX and SciPy. This consumer checks the result and gradient against SciPy and that preparation returns NumPy arrays:</p><div class="vp-raw-html"><div class="markdown"><pre><code class="language-python">&quot;&quot;&quot;Run with: python examples/jax_translations.py matrix_model.py&quot;&quot;&quot;
import importlib.util
import sys

import jax
import numpy as np
from scipy.linalg import expm

spec &#61; importlib.util.spec_from_file_location&#40;&quot;matrix_model&quot;, sys.argv&#91;1&#93;&#41;
model &#61; importlib.util.module_from_spec&#40;spec&#41;
spec.loader.exec_module&#40;model&#41;

A &#61; np.array&#40;&#91;&#91;0.0, 0.2&#93;, &#91;-0.3, 0.1&#93;&#93;&#41;
before &#61; A.copy&#40;&#41;
prepared &#61; model.prepare_data&#40;A&#61;A&#41;
assert prepared and all&#40;isinstance&#40;v, np.ndarray&#41; for v in prepared.values&#40;&#41;&#41;
np.testing.assert_array_equal&#40;A, before&#41;
value &#61; jax.jit&#40;model.kernel&#41;&#40;0.7, **prepared&#41;
gradient &#61; jax.grad&#40;lambda scale: model.kernel&#40;scale, **prepared&#41;&#41;&#40;0.7&#41;
np.testing.assert_allclose&#40;value, 0.7 * expm&#40;A&#41;.sum&#40;&#41;, rtol&#61;1e-12&#41;
np.testing.assert_allclose&#40;gradient, expm&#40;A&#41;.sum&#40;&#41;, rtol&#61;1e-12&#41;
print&#40;&quot;Host preparation and device value/gradient agree with SciPy.&quot;&#41;
</code></pre>
</div></div><p>The repository&#39;s JAX tests execute these examples with <code>RK_JAX_PYTHON</code> enabled: direct and wrapped calls, matrix rank, changed sizes, host preparation, and ordinary gradients against native Julia. The documentation build executes the Julia examples; it displays the Python consumer&#39;s source.</p>`,34)])])}const u=s(t,[["render",l]]);export{k as __pageData,u as default};
