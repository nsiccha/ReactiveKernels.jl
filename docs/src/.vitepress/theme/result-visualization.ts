type VegaView = {
  finalize?: () => void
}

type VegaEmbedResult = {
  view: VegaView
}

type AoVRuntime = {
  views: Record<string, VegaView>
  embed: (
    id: string,
    spec: Record<string, unknown>,
    options?: Record<string, unknown>,
  ) => Promise<VegaEmbedResult>
  __rkDocsAdapter?: boolean
}

type ResultWindow = Window & {
  AoV?: AoVRuntime
}

let vegaEmbedModule: Promise<typeof import('vega-embed')> | undefined

function loadVegaEmbed(): Promise<typeof import('vega-embed')> {
  vegaEmbedModule ??= import('vega-embed')
  return vegaEmbedModule
}

/**
 * Install the small host adapter expected by AlgebraOfVega's generated nodes.
 * AoV remains the specification authority; VitePress owns library loading so
 * client-side navigation never races a page-local CDN script.
 */
export function setupResultVisualizations(): void {
  if (typeof window === 'undefined') return

  const resultWindow = window as ResultWindow
  if (resultWindow.AoV?.__rkDocsAdapter) return

  const views = resultWindow.AoV?.views ?? {}
  resultWindow.AoV = {
    views,
    __rkDocsAdapter: true,
    async embed(id, spec, options = {}) {
      const prior = views[id]
      prior?.finalize?.()

      const module = await loadVegaEmbed()
      const embed = module.default
      const result = await embed(`#${id}`, spec, {
        actions: false,
        renderer: 'svg',
        ...options,
      }) as VegaEmbedResult
      views[id] = result.view
      return result
    },
  }
}
