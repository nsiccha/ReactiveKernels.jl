import './kernel-example.css'

let exampleCounter = 0

type PaneName = 'source' | 'kernel' | 'dag'

const paneOrder: PaneName[] = ['source', 'kernel', 'dag']

function makeButton(label: string, className: string) {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = className
  button.textContent = label
  return button
}

function enhanceKernelExample(root: HTMLElement) {
  if (root.dataset.rkExampleReady === '1') return

  const panes = {} as Record<PaneName, HTMLElement>
  for (const name of paneOrder) {
    const pane = root.querySelector<HTMLElement>(`:scope > [data-rk-pane="${name}"]`)
    if (!pane) return
    panes[name] = pane
  }

  root.dataset.rkExampleReady = '1'
  const id = `rk-example-${++exampleCounter}`
  const labels: Record<PaneName, string> = {
    source: root.dataset.rkSourceLabel || 'Raw input',
    kernel: root.dataset.rkKernelLabel || 'Generated kernel',
    dag: root.dataset.rkDagLabel || 'Compute DAG',
  }
  let active: PaneName = 'source'

  const toolbar = document.createElement('div')
  toolbar.className = 'rk-example__toolbar'

  const tablist = document.createElement('div')
  tablist.className = 'rk-example__tabs'
  tablist.role = 'tablist'
  tablist.setAttribute('aria-label', 'Choose an executable example view')

  const tabs = {} as Record<PaneName, HTMLButtonElement>
  const selectPane = (name: PaneName, focus = false) => {
    active = name
    for (const paneName of paneOrder) {
      const selected = paneName === active
      panes[paneName].hidden = !selected
      tabs[paneName].classList.toggle('is-active', selected)
      tabs[paneName].setAttribute('aria-selected', String(selected))
      tabs[paneName].tabIndex = selected ? 0 : -1
    }
    if (focus) tabs[name].focus()
  }

  for (const name of paneOrder) {
    const tab = makeButton(labels[name], 'rk-example__tab')
    const tabId = `${id}-${name}-tab`
    const panelId = `${id}-${name}-panel`
    tab.id = tabId
    tab.role = 'tab'
    tab.setAttribute('aria-controls', panelId)
    tab.addEventListener('click', () => selectPane(name))
    tabs[name] = tab

    panes[name].id = panelId
    panes[name].classList.add('rk-example__panel')
    panes[name].role = 'tabpanel'
    panes[name].setAttribute('aria-labelledby', tabId)
    tablist.appendChild(tab)
  }

  tablist.addEventListener('keydown', (event) => {
    const index = paneOrder.indexOf(active)
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault()
      selectPane(paneOrder[(index + paneOrder.length - 1) % paneOrder.length], true)
    } else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault()
      selectPane(paneOrder[(index + 1) % paneOrder.length], true)
    } else if (event.key === 'Home') {
      event.preventDefault()
      selectPane(paneOrder[0], true)
    } else if (event.key === 'End') {
      event.preventDefault()
      selectPane(paneOrder[paneOrder.length - 1], true)
    }
  })

  const expand = makeButton('Compare all', 'rk-example__expand')
  expand.setAttribute('aria-haspopup', 'dialog')
  toolbar.append(tablist, expand)
  root.prepend(toolbar)

  const dialog = document.createElement('dialog')
  dialog.className = 'rk-example__dialog'
  dialog.setAttribute(
    'aria-label',
    `${labels.source}, ${labels.kernel}, and ${labels.dag} comparison`,
  )

  const frame = document.createElement('div')
  frame.className = 'rk-example__dialog-frame'
  const header = document.createElement('header')
  header.className = 'rk-example__dialog-header'
  const title = document.createElement('strong')
  title.textContent = `${labels.source} → ${labels.kernel} → ${labels.dag}`
  const close = makeButton('Close', 'rk-example__close')
  close.setAttribute('aria-label', 'Close executable-example comparison')
  close.addEventListener('click', () => dialog.close())
  header.append(title, close)

  const columns = document.createElement('div')
  columns.className = 'rk-example__columns'
  const columnBodies = {} as Record<PaneName, HTMLElement>
  for (const name of paneOrder) {
    const column = document.createElement('section')
    column.className = `rk-example__column rk-example__column--${name}`
    const heading = document.createElement('h3')
    heading.id = `${id}-${name}-dialog-heading`
    heading.textContent = labels[name]
    const body = document.createElement('div')
    body.className = 'rk-example__column-body'
    columnBodies[name] = body
    column.append(heading, body)
    columns.appendChild(column)
  }

  frame.append(header, columns)
  dialog.appendChild(frame)
  root.appendChild(dialog)

  expand.addEventListener('click', () => {
    for (const name of paneOrder) {
      const pane = panes[name]
      pane.hidden = false
      pane.removeAttribute('role')
      pane.removeAttribute('aria-labelledby')
      pane.setAttribute('aria-label', labels[name])
      columnBodies[name].appendChild(pane)
    }
    dialog.showModal()
    close.focus()
  })
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close()
  })
  dialog.addEventListener('close', () => {
    for (const name of paneOrder) {
      const pane = panes[name]
      pane.removeAttribute('aria-label')
      pane.role = 'tabpanel'
      pane.setAttribute('aria-labelledby', tabs[name].id)
      root.insertBefore(pane, dialog)
    }
    selectPane(active)
    expand.focus()
  })

  selectPane('source')
}

function processKernelExamples(root: ParentNode) {
  const candidates = Array.from(root.querySelectorAll<HTMLElement>('[data-rk-example]'))
  if (root instanceof HTMLElement && root.matches('[data-rk-example]')) {
    candidates.unshift(root)
  }
  candidates.forEach(enhanceKernelExample)
}

export function setupKernelExamples() {
  if (typeof window === 'undefined') return

  processKernelExamples(document.body)
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node instanceof HTMLElement) processKernelExamples(node)
      }
    }
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
