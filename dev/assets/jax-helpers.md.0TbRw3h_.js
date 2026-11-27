import{_ as e,o as a,c as i,a6 as n}from"./chunks/framework.BTpfZnOf.js";const u=JSON.parse('{"title":"JAX helpers and submodels","description":"","frontmatter":{},"headers":[],"relativePath":"jax-helpers.md","filePath":"jax-helpers.md","lastUpdated":null}'),t={name:"jax-helpers.md"};function p(l,s,r,o,h,k){return a(),i("div",null,[...s[0]||(s[0]=[n(`<h1 id="JAX-helpers-and-submodels" tabindex="-1">JAX helpers and submodels <a class="header-anchor" href="#JAX-helpers-and-submodels" aria-label="Permalink to &quot;JAX helpers and submodels {#JAX-helpers-and-submodels}&quot;">​</a></h1><p>This guide continues <a href="./jax">JAX source</a>: how emitted modules share submodel functions, and which helpers with loops and branches translate.</p><h2 id="Named-submodels-and-shared-entries" tabindex="-1">Named submodels and shared entries <a class="header-anchor" href="#Named-submodels-and-shared-entries" aria-label="Permalink to &quot;Named submodels and shared entries {#Named-submodels-and-shared-entries}&quot;">​</a></h2><p>Repeated calls to an authored submodel share a Python function when their selected computations agree. A scoped rewrite that changes the computation gets a separate function. Each function carries the submodel&#39;s documentation. The native model retains its expanded graph, so individual locals remain queryable.</p><p>The public density functions and <code>constrain</code> call a shared model function. Their output selectors are fixed by the wrappers, allowing JAX to trace only the requested computation. Call them as before: <code>log_density(u, **prepared)</code>. The submodel functions likewise take a static <code>outputs</code> selection; their arguments and array lengths remain runtime inputs. Consecutive statements that the same outputs need share one selection block (<code>if _entry in (...):</code>, or <code>if any(name in outputs for name in (...)):</code> in a submodel function), and the checks every public function runs, such as the packed-length check, run once before the per-function returns.</p><p>A plate may use separate functions for a scalar return value and a batched density. Combined computations, such as grouped scalar priors, retain their shared graph boundary. Interleaved dependencies split a submodel into named portions that execute in the original dependency order.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactiveKernels, ReactiveKernelsPPL</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@rkppl</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> effect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(loc, scale, spread_rate, n) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;Mean of the effect.&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mean </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(loc, scale)</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;Spread of the members around the mean.&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    spread </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Exponential</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(spread_rate)</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;Standardised member deviations.&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    z[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mean </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> spread </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> z</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @rkppl</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    a </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> effect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(a_loc, a_scale, a_rate, n)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    b </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> effect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(b_loc, b_scale, b_rate, n)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(a </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b, noise)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">bound </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; n </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, a_loc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, a_scale </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, a_rate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">               b_loc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, b_scale </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, b_rate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, noise </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (; y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">built </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> build_kernel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(bound)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">jax_source</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(built, bound))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">&quot;&quot;&quot;Generated by ReactiveKernelsPPL.jax_source from a built @rkppl program; do not edit.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">Data: a_loc, a_rate, a_scale, b_loc, b_rate, b_scale, n, noise, y.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">Model documentation:</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">- \`a.mean\`: Mean of the effect.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">- \`a.spread\`: Spread of the members around the mean.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">- \`a.z\`: Standardised member deviations.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">- \`b.mean\`: Mean of the effect.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">- \`b.spread\`: Spread of the members around the mean.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">- \`b.z\`: Standardised member deviations.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">&quot;&quot;&quot;</span></span>
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
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">DATA = {&quot;a_loc&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;a_rate&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;a_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;b_loc&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;b_rate&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;b_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;n&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;noise&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;y&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 1, &quot;ragged&quot;: False}}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">INPUTS = (&quot;unconstrained&quot;, &quot;ppl_last_a_z&quot;, &quot;ppl_first_b_z&quot;, &quot;ppl_last_b_z&quot;, &quot;y&quot;, &quot;noise&quot;, &quot;_ppl_pw_y_resp_lanes_1&quot;, &quot;_ppl_pw_y_resp_lanes_2&quot;, &quot;_ppl_pw_y_resp_lane_order&quot;, &quot;a_loc&quot;, &quot;a_scale&quot;, &quot;ppl_prior_a_mean_log_scale&quot;, &quot;ppl_prior_a_spread_log_scale&quot;, &quot;a_rate&quot;, &quot;b_loc&quot;, &quot;b_scale&quot;, &quot;ppl_prior_b_mean_log_scale&quot;, &quot;ppl_prior_b_spread_log_scale&quot;, &quot;b_rate&quot;, &quot;n&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">PREPARED = {&quot;ppl_last_a_z&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;ppl_first_b_z&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;ppl_last_b_z&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;y&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 1, &quot;ragged&quot;: False}, &quot;noise&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;_ppl_pw_y_resp_lanes_1&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 1, &quot;ragged&quot;: False}, &quot;_ppl_pw_y_resp_lanes_2&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 1, &quot;ragged&quot;: False}, &quot;_ppl_pw_y_resp_lane_order&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 1, &quot;ragged&quot;: False}, &quot;a_loc&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;a_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;ppl_prior_a_mean_log_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;ppl_prior_a_spread_log_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;a_rate&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;b_loc&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;b_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;ppl_prior_b_mean_log_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;ppl_prior_b_spread_log_scale&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;b_rate&quot;: {&quot;dtype&quot;: &quot;float64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}, &quot;n&quot;: {&quot;dtype&quot;: &quot;int64&quot;, &quot;ndim&quot;: 0, &quot;ragged&quot;: False}}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">UNTRANSLATED = {}</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">class StaticInt(int):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;An integer of the prepared data that sets a shape.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    It is an \`int\` everywhere, and \`jax.jit\` treats it as static, so the</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    functions jit with the prepared data as arguments; a new value traces</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    them again.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def __repr__(self):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return &quot;StaticInt(%d)&quot; % int(self)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">jax.tree_util.register_pytree_node(StaticInt, lambda s: ((), int(s)), lambda v, _: StaticInt(v))</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def effect(</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    loc: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    mean: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_mean_log_scale: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_spread_log_scale: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    scale: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    spread: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    spread_rate: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    z: ArrayLike,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    *,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    outputs: tuple[str, ...],</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">) -&gt; dict:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;\`mean\`: Mean of the effect.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    \`spread\`: Spread of the members around the mean.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    \`z\`: Standardised member deviations.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    results = {}</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if any(name in outputs for name in (&quot;value&quot;,)):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        def value_cell(mean_2, spread_2, z_2):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            plate_value = mean_2 + spread_2 * z_2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return plate_value</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        value = (jnp.zeros(z.shape[:1], dtype=jnp.float64) if 0 in z.shape[:1] else jax.vmap(value_cell, in_axes=(None, None, 0))(mean, spread, z))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if any(name in outputs for name in (&quot;ppl_prior_mean&quot;, &quot;ppl_prior_mean_standard_logpdf&quot;)):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_prior_mean_logpdf_z = (mean - loc) / jnp.where(jnp.logical_and(scale == 0, mean == loc), 1.0, scale)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_prior_mean_standard_logpdf = -0.5 * np.log(2 * np.pi) - 0.5 * ppl_prior_mean_logpdf_z ** 2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if any(name in outputs for name in (&quot;ppl_prior_mean&quot;,)):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_prior_mean = ppl_prior_mean_standard_logpdf - jnp.where(jnp.logical_and(scale == 0, mean != loc), 0.0, ppl_prior_mean_log_scale)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if any(name in outputs for name in (&quot;ppl_prior_spread&quot;,)):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_prior_spread = jax.lax.cond(spread &lt; 0, lambda: -jnp.inf, lambda: -ppl_prior_spread_log_scale - spread / spread_rate)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if any(name in outputs for name in (&quot;ppl_prior_z&quot;, &quot;ppl_pw_prior_z&quot;)):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        def ppl_pw_prior_z_cell(z_2):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            log_scale = np.log(1.0)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            standardized = (z_2 - 0.0) / jnp.where((z_2 == 0.0 if 1.0 == 0 else False), 1.0, 1.0)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            standard_logpdf = -0.5 * np.log(2 * np.pi) - 0.5 * standardized ** 2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            logpdf = standard_logpdf - jnp.where((z_2 != 0.0 if 1.0 == 0 else False), 0.0, log_scale)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return logpdf</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_pw_prior_z = (jnp.zeros(z.shape[:1], dtype=jnp.float64) if 0 in z.shape[:1] else jax.vmap(ppl_pw_prior_z_cell, in_axes=(0,))(z))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if any(name in outputs for name in (&quot;ppl_prior_z&quot;,)):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_prior_z = jnp.sum(ppl_pw_prior_z)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if &quot;ppl_prior_mean&quot; in outputs:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        results[&quot;ppl_prior_mean&quot;] = ppl_prior_mean</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if &quot;ppl_prior_spread&quot; in outputs:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        results[&quot;ppl_prior_spread&quot;] = ppl_prior_spread</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if &quot;ppl_prior_z&quot; in outputs:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        results[&quot;ppl_prior_z&quot;] = ppl_prior_z</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if &quot;ppl_pw_prior_z&quot; in outputs:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        results[&quot;ppl_pw_prior_z&quot;] = ppl_pw_prior_z</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if &quot;ppl_prior_mean_standard_logpdf&quot; in outputs:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        results[&quot;ppl_prior_mean_standard_logpdf&quot;] = ppl_prior_mean_standard_logpdf</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if &quot;value&quot; in outputs:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        results[&quot;value&quot;] = value</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return results</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def prepare_data(a_loc: ArrayLike, a_rate: ArrayLike, a_scale: ArrayLike, b_loc: ArrayLike, b_rate: ArrayLike, b_scale: ArrayLike, n: ArrayLike, noise: ArrayLike, y: ArrayLike) -&gt; dict:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The functions&#39; data inputs, computed from the raw data once per dataset.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    Pass the data by the names the model reads them; a list of arrays is a ragged</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    column. Call the other functions with the result: \`f(u, **prepared)\`.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    y = np.asarray(y, dtype=np.float64)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_a_loc = a_loc</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_a_rate = a_rate</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_a_scale = a_scale</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_b_loc = b_loc</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_b_rate = b_rate</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_b_scale = b_scale</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_n = n</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    rkppl_value_noise = noise</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_last_a_z = 4 + n</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_first_b_z = ppl_last_a_z + 1</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_last_b_z = ppl_first_b_z + (n - 1)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_a_mean_log_scale = np.log(a_scale)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_a_spread_log_scale = np.log(a_rate)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_b_mean_log_scale = np.log(b_scale)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_b_spread_log_scale = np.log(b_rate)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def lane_conditions(lane_input_1, lane_input_2):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return (bool(lane_input_2 &gt; 0),)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    lane_conditions_2 = np.asarray([lane_conditions(entry, noise) for entry in y], dtype=bool).reshape((-1, 1))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    lanes = np.flatnonzero(lane_conditions_2[:, 0])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    lanes_2 = np.flatnonzero(~lane_conditions_2[:, 0])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_pw_y_resp_lanes_1 = lanes</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_pw_y_resp_lanes_2 = lanes_2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_pw_y_resp_lane_order = np.argsort(np.concatenate((lanes, lanes_2,)), kind=&quot;stable&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return {&quot;ppl_last_a_z&quot;: StaticInt(ppl_last_a_z), &quot;ppl_first_b_z&quot;: StaticInt(ppl_first_b_z), &quot;ppl_last_b_z&quot;: StaticInt(ppl_last_b_z), &quot;y&quot;: np.asarray(y, dtype=np.float64), &quot;noise&quot;: noise, &quot;_ppl_pw_y_resp_lanes_1&quot;: np.asarray(ppl_pw_y_resp_lanes_1, dtype=np.int64), &quot;_ppl_pw_y_resp_lanes_2&quot;: np.asarray(ppl_pw_y_resp_lanes_2, dtype=np.int64), &quot;_ppl_pw_y_resp_lane_order&quot;: np.asarray(ppl_pw_y_resp_lane_order, dtype=np.int64), &quot;a_loc&quot;: a_loc, &quot;a_scale&quot;: a_scale, &quot;ppl_prior_a_mean_log_scale&quot;: ppl_prior_a_mean_log_scale, &quot;ppl_prior_a_spread_log_scale&quot;: ppl_prior_a_spread_log_scale, &quot;a_rate&quot;: a_rate, &quot;b_loc&quot;: b_loc, &quot;b_scale&quot;: b_scale, &quot;ppl_prior_b_mean_log_scale&quot;: ppl_prior_b_mean_log_scale, &quot;ppl_prior_b_spread_log_scale&quot;: ppl_prior_b_spread_log_scale, &quot;b_rate&quot;: b_rate, &quot;n&quot;: StaticInt(n)}</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def model(_entry: str, unconstrained: ArrayLike, **prepared: ArrayLike) -&gt; Any:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;Shared model computation. The generated entry wrappers select a static output.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_last_a_z = prepared[&quot;ppl_last_a_z&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_first_b_z = prepared[&quot;ppl_first_b_z&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_last_b_z = prepared[&quot;ppl_last_b_z&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    y = prepared[&quot;y&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    noise = prepared[&quot;noise&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    _ppl_pw_y_resp_lanes_1 = prepared[&quot;_ppl_pw_y_resp_lanes_1&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    _ppl_pw_y_resp_lanes_2 = prepared[&quot;_ppl_pw_y_resp_lanes_2&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    _ppl_pw_y_resp_lane_order = prepared[&quot;_ppl_pw_y_resp_lane_order&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    a_loc = prepared[&quot;a_loc&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    a_scale = prepared[&quot;a_scale&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_a_mean_log_scale = prepared[&quot;ppl_prior_a_mean_log_scale&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_a_spread_log_scale = prepared[&quot;ppl_prior_a_spread_log_scale&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    a_rate = prepared[&quot;a_rate&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    b_loc = prepared[&quot;b_loc&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    b_scale = prepared[&quot;b_scale&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_b_mean_log_scale = prepared[&quot;ppl_prior_b_mean_log_scale&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    ppl_prior_b_spread_log_scale = prepared[&quot;ppl_prior_b_spread_log_scale&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    b_rate = prepared[&quot;b_rate&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    n = prepared[&quot;n&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    unconstrained = jnp.asarray(unconstrained)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    y = jnp.asarray(y)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    _ppl_pw_y_resp_lanes_1 = jnp.asarray(_ppl_pw_y_resp_lanes_1)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    _ppl_pw_y_resp_lanes_2 = jnp.asarray(_ppl_pw_y_resp_lanes_2)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    _ppl_pw_y_resp_lane_order = jnp.asarray(_ppl_pw_y_resp_lane_order)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;, &quot;constrain&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        a_mean = unconstrained[0]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        u_argument = unconstrained[1]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        constrain_2 = jnp.exp(u_argument)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        b_mean = unconstrained[2]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        u_argument_2 = unconstrained[3]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        constrain_3 = jnp.exp(u_argument_2)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        a_z = unconstrained[4:ppl_last_a_z]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        a = effect(</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            loc=(a_loc if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            mean=(a_mean if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ppl_prior_mean_log_scale=(ppl_prior_a_mean_log_scale if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ppl_prior_spread_log_scale=(ppl_prior_a_spread_log_scale if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            scale=(a_scale if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            spread=(constrain_2 if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            spread_rate=(a_rate if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            z=(a_z if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            outputs=tuple(</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                name for name, needed in (</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_mean&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_spread&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_z&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_pw_prior_z&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_mean_standard_logpdf&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;value&quot;, (&quot;log_density&quot;, &quot;log_likelihood&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                )</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                if _entry in needed</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;, &quot;constrain&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        b_z = unconstrained[ppl_first_b_z - 1:ppl_last_b_z]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        b = effect(</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            loc=(b_loc if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            mean=(b_mean if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ppl_prior_mean_log_scale=(ppl_prior_b_mean_log_scale if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ppl_prior_spread_log_scale=(ppl_prior_b_spread_log_scale if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            scale=(b_scale if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            spread=(constrain_3 if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            spread_rate=(b_rate if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            z=(b_z if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;, &quot;log_likelihood&quot;) else None),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            outputs=tuple(</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                name for name, needed in (</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_mean&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_spread&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_z&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_pw_prior_z&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;ppl_prior_mean_standard_logpdf&quot;, (&quot;log_density&quot;, &quot;log_prior&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    (&quot;value&quot;, (&quot;log_density&quot;, &quot;log_likelihood&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                )</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                if _entry in needed</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;, &quot;log_likelihood&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        def y_eta_cell(rkppl_leaf_1, rkppl_leaf_2):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            plate_value = rkppl_leaf_1 + rkppl_leaf_2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return plate_value</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        y_eta = (jnp.zeros(a[&quot;value&quot;].shape[:1], dtype=jnp.float64) if 0 in a[&quot;value&quot;].shape[:1] and b[&quot;value&quot;].shape[:1] == a[&quot;value&quot;].shape[:1] else jax.vmap(y_eta_cell, in_axes=(0, 0))(a[&quot;value&quot;], b[&quot;value&quot;]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        lane_entries = jnp.asarray(y)[_ppl_pw_y_resp_lanes_1]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        lane_entries_2 = jnp.asarray(y_eta)[_ppl_pw_y_resp_lanes_1]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        def ppl_pw_y_resp_cell(y_2, y_eta_2, noise_2):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            log_scale = jnp.log(noise_2)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            standardized = (y_2 - y_eta_2) / jnp.where(jnp.logical_and(noise_2 == 0, y_2 == y_eta_2), 1.0, noise_2)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            standard_logpdf = -0.5 * np.log(2 * np.pi) - 0.5 * standardized ** 2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ppl_density = standard_logpdf - jnp.where(jnp.logical_and(noise_2 == 0, y_2 != y_eta_2), 0.0, log_scale)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return ppl_density</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        plate_part = (jnp.zeros(lane_entries.shape[:1], dtype=jnp.float64) if 0 in lane_entries.shape[:1] and lane_entries_2.shape[:1] == lane_entries.shape[:1] else jax.vmap(ppl_pw_y_resp_cell, in_axes=(0, 0, None))(lane_entries, lane_entries_2, noise))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        lane_entries_3 = jnp.asarray(y)[_ppl_pw_y_resp_lanes_2]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        lane_entries_4 = jnp.asarray(y_eta)[_ppl_pw_y_resp_lanes_2]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        def ppl_pw_y_resp_cell_2(y_2, y_eta_2, noise_2):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            ppl_density = -jnp.inf</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return ppl_density</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        plate_part_2 = (jnp.zeros(lane_entries_3.shape[:1], dtype=jnp.float64) if 0 in lane_entries_3.shape[:1] and lane_entries_4.shape[:1] == lane_entries_3.shape[:1] else jax.vmap(ppl_pw_y_resp_cell_2, in_axes=(0, 0, None))(lane_entries_3, lane_entries_4, noise))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_pw_y_resp = jnp.concatenate((plate_part, plate_part_2,))[_ppl_pw_y_resp_lane_order]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        ppl_lik_y_resp = jnp.sum(ppl_pw_y_resp)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;, &quot;log_prior&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        prior = a[&quot;ppl_prior_mean&quot;] + a[&quot;ppl_prior_spread&quot;] + b[&quot;ppl_prior_mean&quot;] + b[&quot;ppl_prior_spread&quot;] + a[&quot;ppl_prior_z&quot;] + b[&quot;ppl_prior_z&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;, &quot;log_jacobian&quot;):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        u_argument_3 = unconstrained[1]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        u_argument_4 = unconstrained[3]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        log_jacobian_2 = u_argument_3 + u_argument_4</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry in (&quot;log_density&quot;,):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        posterior = prior + ppl_lik_y_resp + log_jacobian_2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if not (unconstrained.size == ppl_last_b_z):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        raise ValueError(&quot;[layout] packed length ≠ layout total: a packed point holds the coordinates of the binding whose layout packed it&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry == &quot;log_density&quot;:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return jnp.asarray(posterior, dtype=jnp.float64)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry == &quot;log_prior&quot;:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return jnp.asarray(prior, dtype=jnp.float64)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry == &quot;log_likelihood&quot;:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return jnp.asarray(ppl_lik_y_resp, dtype=jnp.float64)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry == &quot;log_jacobian&quot;:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return jnp.asarray(log_jacobian_2, dtype=jnp.float64)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    if _entry == &quot;constrain&quot;:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return {&quot;a&quot;: {&quot;mean&quot;: jnp.asarray(a_mean, dtype=jnp.float64), &quot;spread&quot;: jnp.asarray(constrain_2, dtype=jnp.float64), &quot;z&quot;: jnp.asarray(a_z)}, &quot;b&quot;: {&quot;mean&quot;: jnp.asarray(b_mean, dtype=jnp.float64), &quot;spread&quot;: jnp.asarray(constrain_3, dtype=jnp.float64), &quot;z&quot;: jnp.asarray(b_z)}}</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def log_density(unconstrained: ArrayLike, **prepared: ArrayLike) -&gt; jax.Array:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The log density a sampler targets at \`unconstrained\`, the packed unconstrained parameters (length \`dimension(prepared)\`), for the data \`prepare_data\` returned: \`f(u, **prepared)\`.</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    The sum of \`log_prior\`, \`log_likelihood\` and \`log_jacobian\`.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return model(&quot;log_density&quot;, unconstrained, **prepared)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def log_prior(unconstrained: ArrayLike, **prepared: ArrayLike) -&gt; jax.Array:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The log prior density of the constrained parameters at \`unconstrained\`, the packed unconstrained parameters (length \`dimension(prepared)\`), for the data \`prepare_data\` returned: \`f(u, **prepared)\`.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return model(&quot;log_prior&quot;, unconstrained, **prepared)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def log_likelihood(unconstrained: ArrayLike, **prepared: ArrayLike) -&gt; jax.Array:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The log likelihood of the observations at \`unconstrained\`, the packed unconstrained parameters (length \`dimension(prepared)\`), for the data \`prepare_data\` returned: \`f(u, **prepared)\`.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return model(&quot;log_likelihood&quot;, unconstrained, **prepared)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def log_jacobian(unconstrained: ArrayLike, **prepared: ArrayLike) -&gt; jax.Array:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The log Jacobian of the constraining transforms at \`unconstrained\`, the packed unconstrained parameters (length \`dimension(prepared)\`), for the data \`prepare_data\` returned: \`f(u, **prepared)\`.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return model(&quot;log_jacobian&quot;, unconstrained, **prepared)</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def constrain(unconstrained: ArrayLike, **prepared: ArrayLike) -&gt; dict:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The constrained value of each parameter, and of each requested value, at \`unconstrained\`, the packed unconstrained parameters (length \`dimension(prepared)\`), for the data \`prepare_data\` returned: \`f(u, **prepared)\`. Returns a dict, a submodel&#39;s values nested under its name (a submodel call&#39;s own value under \`return\` there):</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    - \`a.mean\`: Mean of the effect.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    - \`a.spread\`: Spread of the members around the mean.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    - \`b.mean\`: Mean of the effect.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    - \`b.spread\`: Spread of the members around the mean.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    - \`a.z\`: Standardised member deviations.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    - \`b.z\`: Standardised member deviations.</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return model(&quot;constrain&quot;, unconstrained, **prepared)</span></span>
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
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    import time</span></span>
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
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        for i in range(12):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            start = i % len(finalists)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            for k in finalists[start:] + finalists[:start]:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                row = rows[k]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                if row[&quot;error&quot;] is not None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    continue</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                started = time.perf_counter()</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                try:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    jax.block_until_ready(steps[k]())</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                except Exception as error:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    if not isinstance(error, MemoryError) and &quot;RESOURCE_EXHAUSTED&quot; not in str(error):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                        raise</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    row[&quot;error&quot;] = str(error)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    row[&quot;seconds&quot;] = row[&quot;fit_seconds&quot;] = float(&quot;inf&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    del steps[k]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                elapsed = time.perf_counter() - started</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                row[&quot;calibration_seconds&quot;] += elapsed</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                if row[&quot;error&quot;] is None:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                    row[&quot;confirmation_samples_seconds&quot;].append(elapsed)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        confirmed = [k for k in finalists if k in steps]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        if not confirmed:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            return None</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        for k in confirmed:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            samples = rows[k][&quot;confirmation_samples_seconds&quot;]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">            rows[k][&quot;confirmation_median_seconds&quot;] = float(np.median(samples))</span></span>
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
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return gradient</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def dimension(prepared: dict) -&gt; int:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;The length of \`unconstrained\` for the data \`prepared\` (what \`prepare_data\` returned).&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return int(prepared[&quot;ppl_last_b_z&quot;])</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def parameters(prepared: dict) -&gt; dict:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;Each parameter&#39;s 0-based \`start\`/\`stop\` in \`unconstrained\`, its transform and dimensions,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    for the data \`prepared\` (what \`prepare_data\` returned).&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return {&quot;a.mean&quot;: {&quot;start&quot;: 0, &quot;stop&quot;: 1, &quot;transform&quot;: &quot;identity&quot;, &quot;dims&quot;: ()}, &quot;a.spread&quot;: {&quot;start&quot;: 1, &quot;stop&quot;: 2, &quot;transform&quot;: &quot;exp&quot;, &quot;dims&quot;: ()}, &quot;b.mean&quot;: {&quot;start&quot;: 2, &quot;stop&quot;: 3, &quot;transform&quot;: &quot;identity&quot;, &quot;dims&quot;: ()}, &quot;b.spread&quot;: {&quot;start&quot;: 3, &quot;stop&quot;: 4, &quot;transform&quot;: &quot;exp&quot;, &quot;dims&quot;: ()}, &quot;a.z&quot;: {&quot;start&quot;: 4, &quot;stop&quot;: int(prepared[&quot;ppl_last_a_z&quot;]), &quot;transform&quot;: &quot;identity&quot;, &quot;dims&quot;: (int(prepared[&quot;n&quot;]),)}, &quot;b.z&quot;: {&quot;start&quot;: int(prepared[&quot;ppl_first_b_z&quot;]) - 1, &quot;stop&quot;: int(prepared[&quot;ppl_last_b_z&quot;]), &quot;transform&quot;: &quot;identity&quot;, &quot;dims&quot;: (int(prepared[&quot;n&quot;]),)}}</span></span></code></pre></div><h2 id="Helpers-with-loops-and-branches" tabindex="-1">Helpers with loops and branches <a class="header-anchor" href="#Helpers-with-loops-and-branches" aria-label="Permalink to &quot;Helpers with loops and branches {#Helpers-with-loops-and-branches}&quot;">​</a></h2><p>A helper becomes a named Python function with its method&#39;s argument names, reused by calls with the same argument representation: from recorded source, or, for a plain helper outside Julia and ReactiveKernels, from its lowered code. Its Julia docstring accompanies the body; a broadcast vectorizes the function itself. A leading underscore and ReactiveKernels&#39; own helpers stay private (<code>_name</code>). Arrays, captured numerical values and padded lengths are normally explicit arguments. A helper that carries loop bounds or reads scan history stays nested in that scope, preserving the bounds and history access analysis. Nested plate cells and scan steps are named after the authored values they compute.</p><p>A helper translates from recorded source when available: <code>@traceable</code> records it, and a trailing Vararg argument (<code>args...</code>, <code>args::Vararg{T,N}</code>) reads as the tuple of the trailing arguments, as the method receives them, and <code>sum(f, t; init)</code> folds such a tuple over its fixed length; <code>for</code> becomes <code>jax.lax.fori_loop</code>, <code>if</code> becomes <code>jax.lax.cond</code>, and <code>A[i] = v</code> becomes <code>A.at[i].set(v)</code>, converting <code>v</code> to <code>A</code>&#39;s element type as <code>setindex!</code> does. Loops and generators iterate elements, <code>enumerate</code> and <code>zip</code>. A <code>jax.lax.cond</code> arm that is one expression is a <code>lambda</code>, and a branch between two values already bound or literal (<code>s == 0 &amp;&amp; x == m ? 1.0 : s</code>) is <code>jnp.where</code>, whose value and derivative are the conditional&#39;s; an arm that computes anything stays lazy. A condition decided while tracing (a shape or literal test) is Python&#39;s own <code>if</code>, so a guard <code>n == k || throw(...)</code> reads <code>if not (n == k): raise ValueError(...)</code>. Plain lowered helpers support straight-line code and terminal forward branches, including size guards. They can create immutable closures and fold fixed argument lists with <code>Base.afoldl</code>. Broadcasting <code>Ref(x)</code> passes <code>x</code> whole to each call. Captures remain runtime inputs; host <code>prepare_data</code> uses the same translation.</p><p><code>@traceable</code> methods also accept named keywords, including required and typed keywords. Calls such as <code>weighted(xs; scale)</code> bind by name. Omitted defaults are evaluated in the helper&#39;s module, in declaration order, and may read positional arguments and earlier keywords. Supplied keywords skip their defaults. Keyword values and data-derived sizes remain runtime arguments in both host preparation and JAX functions.</p><p>A captured <code>NamedTuple</code> of data can be read whole, for example passed to a helper (<code>metadata = (; weights = w, indices = idx)</code> captured by a kernel that calls <code>weighted(x, metadata)</code>). It is the Python tuple of its fields, as a named-tuple HAVE crosses, and the helper&#39;s field reads (<code>metadata.weights[1]</code>) index it by position. Each array field is a typed module constant named after the record and the field (<code>METADATA_WEIGHTS = jnp.asarray(..., dtype=jnp.float64)</code>), with Julia&#39;s shape, element type and one-based reads. A record holding an array is itself a module constant (<code>METADATA = (METADATA_WEIGHTS, METADATA_INDICES)</code>), so a call passes that one name; a record of numbers stays a static literal and can size an allocation. A field that is a function or a type raises <code>JAXEmissionError</code>.</p><p>Field names obtained from <code>keys(record)</code> or the <code>names</code> parameter of <code>record::NamedTuple{names}</code> remain structural through locals and helper calls. For example, <code>getproperty(record, first(keys(record)))</code> selects the first field in the record&#39;s declared order, in host preparation and JAX functions. The record&#39;s numerical fields and array lengths remain runtime values; this does not specialize a module to the data used when emitting it.</p><p>A helper writing its argument in place (<code>rv[k] = x</code>, or via a helper it calls; host also <code>push!</code>, <code>sort!</code>) writes the caller&#39;s array, as in Julia, read or not. The host (RK 0.1.278) passes the caller&#39;s own local, copying an input or module constant, a list when grown; a slice or other expression raises <code>JAXEmissionError</code> (<code>x[r]</code> copies, <code>view(x, r)</code> does not; NumPy spells both alike). A device function (RK 0.1.303) also returns written arrays for the caller to rebind, through loops and branches too (0.1.304); a write it cannot rebind (a helper named only inside a loop, another statement&#39;s value) raises <code>JAXEmissionError</code>.</p><p>A method that a package generates, such as one its macro writes into the caller&#39;s module, cannot carry <code>@traceable</code>. The generating package keeps the method&#39;s source as a <code>TraceableSource</code> (argument names, body, and the module whose globals the body reads) and hands it over through a provider. It registers the provider with <code>register_source_provider!</code>, once, from the <code>__init__</code> of the package extension that loads with ReactiveKernels. The emitter asks the provider for a call that has no other translation and no <code>@traceable</code> source, with the function and the <code>Tuple</code> type of the argument types. The provider answers <code>nothing</code> or the source of the method those arguments call natively.</p><p>The example below stands in for such a package. Its macro defines the method and keeps its source.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactiveKernels</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> GENERATED </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dict{Any,TraceableSource}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">macro</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> generate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(def)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    call, body </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> def</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">args</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    names </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Tuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(arg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">args[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> arg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> call</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">args[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    source </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> TraceableSource</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(names, body, __module__)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    :(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">$</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">esc</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(def)); GENERATED[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">$</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">esc</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(call</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">args[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]))] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> $</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">source)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@generate</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> decay_sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{Float64}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, rate</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rate)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x[i] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> rate)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> rate </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> rate</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">provide_source</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, argument_types) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(GENERATED, f, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">register_source_provider!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(provide_source)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@kernel</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> decayed</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{Float64}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, rate</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    value </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> decay_sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, rate)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> value</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">jax_source</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(decayed; have </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:rate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), want </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> :value</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">&quot;&quot;&quot;Generated by ReactiveKernels.jax_source; do not edit.&quot;&quot;&quot;</span></span>
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
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def decay_sum(x: ArrayLike, rate: ArrayLike) -&gt; jax.Array:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;Compute \`decay_sum\` from its authored source.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    total = jnp.zeros_like(rate)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def loop_body(i, carry):</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        (total_2,) = carry</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        total_3 = total_2 + x[i - 1] * jnp.exp(-i * rate)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return (total_3,)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    loop_carry = jax.lax.fori_loop(1, 4, loop_body, (jnp.asarray(total, dtype=jnp.float64),))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    def then_arm():</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        total_2 = loop_carry[0] / rate</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">        return (None, jnp.asarray(total_2, dtype=jnp.float64))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    branch = jax.lax.cond(rate &gt; 0, then_arm, lambda: (None, jnp.asarray(loop_carry[0], dtype=jnp.float64)))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return branch[1]</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">def kernel(x: ArrayLike, rate: ArrayLike) -&gt; jax.Array:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    &quot;&quot;&quot;Compute \`value\` from the inputs.&quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    x = jnp.asarray(x)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    decay_sum_2 = decay_sum(x, rate)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    value = decay_sum_2</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">    return jnp.asarray(value, dtype=jnp.float64)</span></span></code></pre></div><p>Body-level returns, including <code>condition || return fallback(...)</code>, keep the continuation lazy. Returns inside loops remain unsupported. In device code, <code>continue</code> skips the iteration’s remaining statements and <code>break</code> exits the nearest loop. A vector loop (<code>for value in values</code>) binds a loop-local name. Bounded loops with early exits or padded lengths can stop at the active prefix and retain a guarded scan for reverse AD. Unbounded loops need an authored rule for reverse AD. On CUDA, pure scalar recurrences using supported Float64 arithmetic can execute in one Pallas Triton kernel. Other bodies use ordinary JAX control flow. Host <code>prepare_data</code> supports the same exits. Only <code>jax_source</code> reads provider source. Native execution runs the original method.</p><p><code>@inbounds</code> and <code>@simd</code> preserve the source loop. Chained comparisons evaluate a shared middle operand once and reach later operands only when earlier comparisons succeed. Plain helpers accept acyclic forward joins and static Boolean <code>Val</code> parameters. Helpers can destructure numerical arrays and use <code>firstindex</code>, <code>lastindex</code>, integer parity predicates, and floating-point <code>copysign</code>.</p><p>A numerical helper may return a scalar or <code>nothing</code>; consume that result inside the program with <code>something</code> or <code>isnothing</code>. Device branches carry a presence flag and the scalar, while host preparation keeps ordinary Python values. A numerical vector prefix <code>c[1:n]</code> carries its runtime length and checks bounds. <code>evalpoly(x, c)</code> for real vectors retains Horner&#39;s loop over that logical length, including a runtime prefix; unused coefficients contribute neither values nor derivatives.</p><p>Conditional arrays and tuple fields retain their logical lengths, including when destructured. A bound such as <code>min(n, length(row) - 1)</code> keeps a runtime range bounded by the row&#39;s capacity. Numerical matrix allocation can use a bounded dimension, such as <code>Matrix{Float64}(undef, length(row), 2)</code>, and several axes may share that logical extent. Independently padded dimensions remain unsupported. Array lengths and loop bodies are never copied from a sample dataset into emitted source.</p><p>A source <code>ntuple(j -&gt; expression, Val(N))</code> records its authored field count, with runtime captures; a splatted tuple (<code>(a, rest(...)...)</code>, <code>f(t...)</code>) passes its fields, so recursion over a record&#39;s names unrolls over them. A homogeneous numerical tuple supports computed integer reads, such as in a loop, through a bounds-checked array view; the loop stays a runtime loop.</p><p>A device helper can allocate <code>Vector{Vector{Float64}}(undef, k)</code>, fill its slots with fixed-width numerical rows, and use <code>isassigned(rows, i)</code> to skip earlier assignments. A size such as <code>isempty(ids) ? 0 : maximum(ids)</code> uses bounds computed from prepared integer data, including ragged cells. Allocation and iteration remain runtime array operations; emitted source does not grow with the data. Assignment state travels with the numerical storage through loops and branches. Missing references and invalid indices raise when read; <code>isassigned</code> returns false for either.</p><p>This nested-buffer representation accepts fresh rows with an authored fixed shape, such as <code>[a, b, c]</code> or <code>zeros(3)</code>. Different row shapes require a ragged representation, and mutation through stored row aliases or a helper&#39;s buffer or row argument requires reference writeback; these cases raise <code>JAXEmissionError</code>. Buffers are consumed inside the program by indexing or plates; direct public buffer outputs are not supported yet.</p>`,26)])])}const c=e(t,[["render",p]]);export{u as __pageData,c as default};
