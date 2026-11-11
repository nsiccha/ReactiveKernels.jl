// .vitepress/theme/index.ts
import type { DirectiveBinding } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'
import 'virtual:mathjax-styles.css';


// __DV_PLUGIN_COMPONENT_IMPORTS__

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import './style.css' // You could setup your own, or else a default will be copied.
import './docstrings.css' // You could setup your own, or else a default will be copied.
import './overrides.css' // You could setup your own, or else a default will be copied.

function exposeDagLibraries() {
  if (typeof window === 'undefined') return
  const dagWindow = window as Window & {
    __rkDagLibraries?: Promise<void>
    __rkDagLoadBundledLibraries?: () => Promise<void>
  }
  dagWindow.__rkDagLoadBundledLibraries = () => {
    if (dagWindow.__rkDagLibraries) return dagWindow.__rkDagLibraries
    dagWindow.__rkDagLibraries = Promise.all([
      import('cytoscape'),
      // cytoscape-elk does not publish TypeScript declarations.
      // @ts-expect-error missing upstream declaration
      import('cytoscape-elk'),
    ]).then(([cytoscapeModule, elkExtensionModule]) => {
      const cytoscape = cytoscapeModule.default
      const cytoscapeElk = elkExtensionModule.default
      cytoscape.use(cytoscapeElk)
      Object.assign(window, { cytoscape, cytoscapeElk })
    })
    return dagWindow.__rkDagLibraries
  }
}

// `v-exec-scripts` runs the <script> tags inside a `v-html`'d block: innerHTML never executes
// scripts, so we re-create each one. `src` scripts are awaited so order holds (bundle before
// its callers). Used on interactive text/html output (WGLMakie/Bonito, Plotly) which the writer
// wraps in <ClientOnly> + this directive.
async function activateScripts(container: Element): Promise<void> {
  for (const payload of container.querySelectorAll<HTMLElement>('[data-rk-html]')) {
    const bytes = Uint8Array.from(window.atob(payload.dataset.rkHtml!), char => char.charCodeAt(0))
    payload.innerHTML = new TextDecoder().decode(bytes)
    delete payload.dataset.rkHtml
  }
  for (const old of Array.from(container.querySelectorAll('script'))) {
    const fresh = document.createElement('script')
    for (const attr of Array.from(old.attributes)) fresh.setAttribute(attr.name, attr.value)
    fresh.textContent = old.textContent
    const hasSrc = old.hasAttribute('src')
    const ran = hasSrc
      ? new Promise<void>((resolve, reject) => {
          fresh.addEventListener('load', () => resolve(), { once: true })
          fresh.addEventListener('error', () => reject(new Error(`Could not load ${fresh.src}`)), { once: true })
        })
      : Promise.resolve()
    old.replaceWith(fresh) // inserting is what runs it; inline scripts run synchronously
    await ran
  }
}

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  enhanceApp({ app, router, siteData }) {
    // Public docs lazy-load the Vite-bundled renderer only on pages that mount
    // a DAG. The HTML component's pinned CDN fallback remains for notebooks
    // and saved standalone documents outside this host.
    exposeDagLibraries()
    enhanceAppWithTabs(app);

    // Execute the scripts inside interactive `text/html` outputs (WGLMakie/Bonito, Plotly, …)
    // once their `<ClientOnly>` wrapper has mounted on the client. `mounted` fires on the
    // initial client render and again whenever the page component is remounted by a
    // client-side navigation, so figures initialise instead of staying blank.
    app.directive('exec-scripts', {
      mounted(el: HTMLElement, binding: DirectiveBinding) {
        if (typeof binding.value === 'string') {
          const binary = window.atob(binding.value)
          const bytes = new Uint8Array(binary.length)
          for (let index = 0; index < binary.length; index += 1) {
            bytes[index] = binary.charCodeAt(index)
          }
          el.innerHTML = new TextDecoder().decode(bytes)
        }
        activateScripts(el)
      },
    })
    // __DV_PLUGIN_COMPONENT_REGISTRATIONS__
  }
}
export default Theme
