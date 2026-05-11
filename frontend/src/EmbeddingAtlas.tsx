import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { zoom as d3Zoom, zoomIdentity, type ZoomTransform } from 'd3-zoom'
import { select as d3Select } from 'd3-selection'
import { computeQueryCentroid, convexHull, type QueryCentroid } from './atlasQueryCentroid'
import ThemeToggle, { type Theme } from './ThemeToggle'
import type { Article } from './types'
import './EmbeddingAtlas.css'

export type AtlasMode = 'standalone' | 'embedded' | 'similar'
export type AtlasSource = 'minilm' | 'svd'

type AtlasMetaPayload = {
  n_docs: number
  generated_at_utc: string
  canonical_source: string
  sections: string[]
  ids: string[]
  titles: string[]
  years: number[]
  section_indices: number[]
  urls: string[]
  authors?: string[]
  keywords?: string[][]
}

export type EmbeddingAtlasProps = {
  mode: AtlasMode
  // Nav + theme (used in 'standalone' mode):
  theme?: Theme
  onToggleTheme?: () => void
  onBackToCompose?: () => void
  onOpenAbout?: () => void
  onOpenMethod?: () => void
  onOpenTutorial?: () => void
  active?: 'explore' | null
  // Highlighting:
  highlightedIds?: string[]
  highlightedArticles?: Article[]
  focalId?: string
  queryString?: string
  drawQueryLines?: boolean
  autoZoom?: 'none' | 'highlighted' | 'focal'
  // Defaults / interaction:
  defaultSource?: AtlasSource
  onPointClick?: (articleId: string) => void
  onClose?: () => void
}

const COORD_RANGE = 30000

// Spatial-grid for hover hit-testing. Cells are in data-space (int16).
const GRID_SIZE = 96
const GRID_CELL = (2 * COORD_RANGE) / GRID_SIZE

function buildSpatialGrid(coords: Int16Array): Map<number, number[]> {
  const grid = new Map<number, number[]>()
  const n = coords.length / 2
  for (let i = 0; i < n; i++) {
    const x = coords[i * 2]
    const y = coords[i * 2 + 1]
    const gx = Math.min(GRID_SIZE - 1, Math.max(0, Math.floor((x + COORD_RANGE) / GRID_CELL)))
    const gy = Math.min(GRID_SIZE - 1, Math.max(0, Math.floor((y + COORD_RANGE) / GRID_CELL)))
    const key = gy * GRID_SIZE + gx
    let bucket = grid.get(key)
    if (!bucket) {
      bucket = []
      grid.set(key, bucket)
    }
    bucket.push(i)
  }
  return grid
}

// Inline color palettes (mirror App.css `--paper`, `--ink-rgb`, `--accent-rgb`).
// Reading these via `getComputedStyle(<html>)` races with App.tsx's separate
// `useEffect` that writes `data-theme`, so we keep the canvas-paint pipeline
// keyed to plain React state (`domTheme`) instead. The CSS-var values for
// non-canvas chrome still come from `data-theme` — only the canvas needs
// this decoupled path.
type DomTheme = 'light' | 'dark'
const ATLAS_THEME_PALETTE: Record<DomTheme, {
  paper: string
  inkRgb: [number, number, number]
  accentRgb: [number, number, number]
}> = {
  light: {
    paper: '#fafaf7',
    inkRgb: [26, 26, 26],
    accentRgb: [122, 29, 29],
  },
  dark: {
    paper: '#15130f',
    inkRgb: [237, 228, 210],
    accentRgb: [217, 115, 84],
  },
}

function resolveCurrentDomTheme(): DomTheme {
  if (typeof document === 'undefined') return 'light'
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

type AtlasData = {
  coordsMinilm: Int16Array
  coordsSvd: Int16Array | null
  meta: AtlasMetaPayload
  idToIndex: Map<string, number>
  gridMinilm: Map<number, number[]>
  gridSvd: Map<number, number[]> | null
}

function useAtlasData(): {
  data: AtlasData | null
  error: string | null
  loading: boolean
} {
  const [data, setData] = useState<AtlasData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    Promise.all([
      fetch('/embedding_atlas_minilm.bin').then((r) => {
        if (!r.ok) throw new Error(`MiniLM coords ${r.status}`)
        return r.arrayBuffer()
      }),
      fetch('/embedding_atlas_svd.bin').then((r) => {
        if (!r.ok) {
          // SVD is optional — return null
          return null
        }
        return r.arrayBuffer()
      }).catch(() => null),
      fetch('/embedding_atlas_meta.json').then((r) => {
        if (!r.ok) throw new Error(`Atlas meta ${r.status}`)
        return r.json() as Promise<AtlasMetaPayload>
      }),
    ])
      .then(([minilmBuf, svdBuf, meta]) => {
        if (cancelled) return
        const coordsMinilm = new Int16Array(minilmBuf)
        const coordsSvd = svdBuf ? new Int16Array(svdBuf) : null
        const idToIndex = new Map<string, number>()
        for (let i = 0; i < meta.ids.length; i++) idToIndex.set(meta.ids[i], i)
        const gridMinilm = buildSpatialGrid(coordsMinilm)
        const gridSvd = coordsSvd ? buildSpatialGrid(coordsSvd) : null
        setData({ coordsMinilm, coordsSvd, meta, idToIndex, gridMinilm, gridSvd })
        setLoading(false)
      })
      .catch((err) => {
        if (cancelled) return
        setError(String(err?.message ?? err))
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { data, error, loading }
}

function findNearest(
  coords: Int16Array,
  grid: Map<number, number[]>,
  dataX: number,
  dataY: number,
  radiusData: number,
): number | null {
  const gxMin = Math.max(0, Math.floor((dataX - radiusData + COORD_RANGE) / GRID_CELL))
  const gxMax = Math.min(GRID_SIZE - 1, Math.floor((dataX + radiusData + COORD_RANGE) / GRID_CELL))
  const gyMin = Math.max(0, Math.floor((dataY - radiusData + COORD_RANGE) / GRID_CELL))
  const gyMax = Math.min(GRID_SIZE - 1, Math.floor((dataY + radiusData + COORD_RANGE) / GRID_CELL))
  let best = -1
  let bestDist = radiusData * radiusData
  for (let gy = gyMin; gy <= gyMax; gy++) {
    for (let gx = gxMin; gx <= gxMax; gx++) {
      const bucket = grid.get(gy * GRID_SIZE + gx)
      if (!bucket) continue
      for (const i of bucket) {
        const dx = coords[i * 2] - dataX
        const dy = coords[i * 2 + 1] - dataY
        const d2 = dx * dx + dy * dy
        if (d2 < bestDist) {
          bestDist = d2
          best = i
        }
      }
    }
  }
  return best === -1 ? null : best
}

export default function EmbeddingAtlas(props: EmbeddingAtlasProps) {
  const { data, error, loading } = useAtlasData()
  const bgCanvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const minimapCanvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const transformRef = useRef<ZoomTransform>(zoomIdentity)
  const [transformTick, setTransformTick] = useState(0)
  const [source, setSource] = useState<AtlasSource>(props.defaultSource ?? 'minilm')
  const [searchText, setSearchText] = useState('')
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null)
  const [size, setSize] = useState<{ w: number; h: number; dpr: number }>({ w: 0, h: 0, dpr: 1 })
  // `domTheme` is the source of truth for canvas colors. We initialize it
  // from `<html data-theme>` at mount, then keep it in lock-step with that
  // attribute via a MutationObserver. This deliberately avoids depending on
  // `props.theme`, which can be one phase ahead of the DOM attribute.
  const [domTheme, setDomTheme] = useState<DomTheme>(() => resolveCurrentDomTheme())
  useEffect(() => {
    const root = document.documentElement
    const sync = () => setDomTheme(resolveCurrentDomTheme())
    // Sync once on mount in case the attribute was set before mount.
    sync()
    const observer = new MutationObserver(sync)
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [])

  // Resize observer — keeps canvas matched to its container.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const update = () => {
      const rect = el.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      setSize({ w: Math.max(1, Math.floor(rect.width)), h: Math.max(1, Math.floor(rect.height)), dpr })
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [])

  const coords = useMemo(() => {
    if (!data) return null
    if (source === 'svd') return data.coordsSvd ?? data.coordsMinilm
    return data.coordsMinilm
  }, [data, source])

  const grid = useMemo(() => {
    if (!data) return null
    if (source === 'svd') return data.gridSvd ?? data.gridMinilm
    return data.gridMinilm
  }, [data, source])

  const idToIndex = data?.idToIndex ?? null
  const meta = data?.meta ?? null

  // Resolve highlighted indices once.
  const highlightedIndices = useMemo(() => {
    if (!props.highlightedIds || !idToIndex) return null
    const out = new Set<number>()
    for (const id of props.highlightedIds) {
      const idx = idToIndex.get(id)
      if (idx !== undefined) out.add(idx)
    }
    return out
  }, [props.highlightedIds, idToIndex])

  const focalIndex = useMemo(() => {
    if (!props.focalId || !idToIndex) return null
    return idToIndex.get(props.focalId) ?? null
  }, [props.focalId, idToIndex])

  // Query position in mode 'embedded'. Preferred path: call the backend
  // `/api/visualization/project_query` route, which embeds the query with
  // the matching retrieval processor and runs it through `umap_model.transform()`
  // — the exact projection, no nearest-neighbor approximation. Fallback
  // (shown immediately while the request is in flight, or if the request
  // fails) is the k-NN weighted centroid of the result articles' 2D coords.
  const fallbackCentroid: QueryCentroid | null = useMemo(() => {
    if (props.mode !== 'embedded' || !props.highlightedArticles || !coords || !idToIndex) return null
    return computeQueryCentroid(props.highlightedArticles, coords, idToIndex)
  }, [props.mode, props.highlightedArticles, coords, idToIndex])

  const [exactQueryXY, setExactQueryXY] = useState<{ x: number; y: number; source: AtlasSource; query: string } | null>(null)
  useEffect(() => {
    if (props.mode !== 'embedded') return
    const query = (props.queryString || '').trim()
    if (!query) return
    if (
      exactQueryXY &&
      exactQueryXY.query === query &&
      exactQueryXY.source === source
    ) return
    let cancelled = false
    const body = JSON.stringify({ query, source })
    fetch('/api/visualization/project_query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`status ${r.status}`))))
      .then((data) => {
        if (cancelled) return
        if (typeof data?.x !== 'number' || typeof data?.y !== 'number') return
        setExactQueryXY({ x: data.x, y: data.y, source, query })
      })
      .catch(() => {
        // silent: leave the fallback centroid in place
      })
    return () => {
      cancelled = true
    }
  }, [props.mode, props.queryString, source, exactQueryXY])

  const queryCentroid: QueryCentroid | null = useMemo(() => {
    if (
      exactQueryXY &&
      exactQueryXY.source === source &&
      exactQueryXY.query === (props.queryString || '').trim()
    ) {
      // Use the dispersion verdict from the fallback (computed from result
      // article coords) so the radial-lines / convex-hull decision stays
      // consistent — but anchor the marker at the exact transformed point.
      return {
        x: exactQueryXY.x,
        y: exactQueryXY.y,
        dispersed: fallbackCentroid?.dispersed ?? false,
      }
    }
    return fallbackCentroid
  }, [exactQueryXY, fallbackCentroid, props.queryString, source])

  // Set up d3-zoom — attach to the OVERLAY canvas (top layer) so pointer
  // events hit a single element regardless of which layer is being painted.
  useEffect(() => {
    const canvas = overlayCanvasRef.current
    if (!canvas || !size.w || !size.h) return
    const selection = d3Select(canvas as Element)
    const zoom = d3Zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.5, 40])
      .on('zoom', (ev) => {
        transformRef.current = ev.transform
        setTransformTick((t) => t + 1)
      })
    selection.call(zoom as unknown as (sel: typeof selection) => void)
    return () => {
      selection.on('.zoom', null)
    }
  }, [size.w, size.h])

  // Auto-zoom on initial data load for similar/embedded modes that pass autoZoom.
  useEffect(() => {
    if (!coords || !size.w || !size.h) return
    const indices: number[] = []
    if (props.autoZoom === 'focal' && focalIndex !== null) indices.push(focalIndex)
    if ((props.autoZoom === 'focal' || props.autoZoom === 'highlighted') && highlightedIndices) {
      for (const i of highlightedIndices) indices.push(i)
    }
    if (indices.length < 2) return
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const i of indices) {
      const x = coords[i * 2]
      const y = coords[i * 2 + 1]
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    const pad = Math.max((maxX - minX) * 0.4, (maxY - minY) * 0.4, 500)
    const dataW = maxX - minX + 2 * pad
    const dataH = maxY - minY + 2 * pad
    const baseScale = Math.min(size.w, size.h) / (2 * COORD_RANGE)
    const targetScale = Math.min(size.w / (dataW * baseScale), size.h / (dataH * baseScale))
    const k = Math.min(20, Math.max(1, targetScale))
    // Center of cluster in data space.
    const cx = (minX + maxX) / 2
    const cy = (minY + maxY) / 2
    // We render with `canvas = (w/2 + data*baseScale) * k + t`. Solve for t so
    // that the cluster centre lands at the canvas centre (w/2, h/2).
    const tx = size.w / 2 - (size.w / 2 + cx * baseScale) * k
    const ty = size.h / 2 - (size.h / 2 + cy * baseScale) * k
    transformRef.current = zoomIdentity.translate(tx, ty).scale(k)
    // Sync d3-zoom's internal transform so wheel/drag rebases off auto-zoom.
    const overlay = overlayCanvasRef.current
    if (overlay) {
      const sel = d3Select(overlay as Element)
      sel.property('__zoom', transformRef.current)
    }
    setTransformTick((t) => t + 1)
  }, [coords, size.w, size.h, focalIndex, highlightedIndices, props.autoZoom])

  // Build search-match set for standalone mode.
  const searchMatches = useMemo(() => {
    if (!searchText || !meta) return null
    const needle = searchText.toLowerCase()
    const matches = new Set<number>()
    for (let i = 0; i < meta.titles.length; i++) {
      if (meta.titles[i].toLowerCase().includes(needle)) matches.add(i)
    }
    return matches
  }, [searchText, meta])

  // Helpers shared by both render passes.
  const dataToCanvas = useMemo(() => {
    if (!size.w || !size.h) return null
    const baseScale = Math.min(size.w, size.h) / (2 * COORD_RANGE)
    return { baseScale }
  }, [size.w, size.h])

  // ===== Heavy (background) render: 66K bg points + radial lines + hull +
  // highlights + focal + query marker + search-matches.
  // Doesn't depend on hoverIdx, so hover updates do NOT trigger this.
  // Colors come from the `domTheme`-keyed palette, not getComputedStyle, so
  // the canvas never paints with stale CSS vars.
  useEffect(() => {
    const canvas = bgCanvasRef.current
    if (!canvas || !coords || !size.w || !size.h || !dataToCanvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = Math.round(size.w * size.dpr)
    canvas.height = Math.round(size.h * size.dpr)
    canvas.style.width = `${size.w}px`
    canvas.style.height = `${size.h}px`
    ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0)

    const palette = ATLAS_THEME_PALETTE[domTheme]
    const paper = palette.paper
    const inkRgb = palette.inkRgb
    const accentRgb = palette.accentRgb
    const isDark = domTheme === 'dark'

    ctx.fillStyle = paper
    ctx.fillRect(0, 0, size.w, size.h)

    const tform = transformRef.current
    const { baseScale } = dataToCanvas
    const k = tform.k
    const tx = tform.x
    const ty = tform.y
    // d3-zoom's invariant: canvas = k * preCentered + translate, where
    // preCentered = (w/2 + data*baseScale). At identity (k=1, t=0) a data point
    // (0,0) lands at the canvas centre. d3 adjusts (tx, ty) on wheel/drag so
    // the cursor stays anchored to the same data point as k changes.
    const dataToCanvasX = (dx: number) => (size.w / 2 + dx * baseScale) * k + tx
    const dataToCanvasY = (dy: number) => (size.h / 2 + dy * baseScale) * k + ty

    const n = coords.length / 2
    const hasHighlights = (highlightedIndices && highlightedIndices.size > 0) || focalIndex !== null
    const hasSearch = searchMatches && searchMatches.size > 0

    // Background dots — single neutral ink tone, alpha-dimmed when there's a focus layer.
    const bgPointSize = Math.max(0.7, Math.min(2.6, 1.0 + k * 0.05))
    const bgAlpha = hasHighlights || hasSearch ? (isDark ? 0.18 : 0.16) : isDark ? 0.55 : 0.42
    ctx.globalAlpha = bgAlpha
    ctx.fillStyle = `rgb(${inkRgb[0]},${inkRgb[1]},${inkRgb[2]})`
    for (let i = 0; i < n; i++) {
      if (hasHighlights && (highlightedIndices?.has(i) || focalIndex === i)) continue
      if (hasSearch && searchMatches?.has(i)) continue
      const cx = dataToCanvasX(coords[i * 2])
      const cy = dataToCanvasY(coords[i * 2 + 1])
      if (cx < -4 || cx > size.w + 4 || cy < -4 || cy > size.h + 4) continue
      ctx.fillRect(cx - bgPointSize / 2, cy - bgPointSize / 2, bgPointSize, bgPointSize)
    }
    ctx.globalAlpha = 1

    // Radial query lines (mode 'embedded', non-dispersed).
    if (
      props.mode === 'embedded' &&
      props.drawQueryLines &&
      queryCentroid &&
      !queryCentroid.dispersed &&
      highlightedIndices &&
      props.highlightedArticles
    ) {
      const qx = dataToCanvasX(queryCentroid.x)
      const qy = dataToCanvasY(queryCentroid.y)
      ctx.lineWidth = 1
      const articles = props.highlightedArticles
      const scoreMap = new Map<number, number>()
      for (const a of articles) {
        const idx = idToIndex?.get(String(a.id))
        if (idx === undefined) continue
        const s = a.combined_score ?? a.score ?? a.topic_score_normalized ?? a.topic_score ?? 0
        scoreMap.set(idx, typeof s === 'number' ? s : 0)
      }
      const ordered = Array.from(scoreMap.entries()).sort((a, b) => a[1] - b[1])
      const minScore = ordered.length ? ordered[0][1] : 0
      const maxScore = ordered.length ? ordered[ordered.length - 1][1] : 1
      const range = Math.max(1e-6, maxScore - minScore)
      for (const [idx, score] of ordered) {
        const t = (score - minScore) / range
        const alpha = 0.08 + 0.14 * t
        ctx.strokeStyle = `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},${alpha})`
        ctx.beginPath()
        ctx.moveTo(qx, qy)
        ctx.lineTo(dataToCanvasX(coords[idx * 2]), dataToCanvasY(coords[idx * 2 + 1]))
        ctx.stroke()
      }
    }

    // Dispersed-result convex hull fallback.
    if (
      props.mode === 'embedded' &&
      queryCentroid &&
      queryCentroid.dispersed &&
      highlightedIndices &&
      highlightedIndices.size >= 3
    ) {
      const pts: Array<[number, number]> = []
      for (const idx of highlightedIndices) {
        pts.push([dataToCanvasX(coords[idx * 2]), dataToCanvasY(coords[idx * 2 + 1])])
      }
      const hull = convexHull(pts)
      if (hull.length >= 3) {
        ctx.beginPath()
        ctx.moveTo(hull[0][0], hull[0][1])
        for (let i = 1; i < hull.length; i++) ctx.lineTo(hull[i][0], hull[i][1])
        ctx.closePath()
        ctx.fillStyle = `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},0.10)`
        ctx.fill()
        ctx.strokeStyle = `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},0.35)`
        ctx.lineWidth = 1
        ctx.setLineDash([4, 4])
        ctx.stroke()
        ctx.setLineDash([])
      }
    }

    // Highlighted result points.
    if (highlightedIndices && highlightedIndices.size > 0) {
      const ptSize = Math.max(3, Math.min(8, 3.5 + k * 0.18))
      ctx.fillStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      for (const i of highlightedIndices) {
        const cx = dataToCanvasX(coords[i * 2])
        const cy = dataToCanvasY(coords[i * 2 + 1])
        ctx.beginPath()
        ctx.arc(cx, cy, ptSize / 2, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Standalone search matches.
    if (searchMatches && !hasHighlights) {
      const ptSize = Math.max(3, Math.min(7, 3 + k * 0.14))
      ctx.fillStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      for (const i of searchMatches) {
        const cx = dataToCanvasX(coords[i * 2])
        const cy = dataToCanvasY(coords[i * 2 + 1])
        ctx.beginPath()
        ctx.arc(cx, cy, ptSize / 2, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Focal article (mode 'similar').
    if (focalIndex !== null) {
      const cx = dataToCanvasX(coords[focalIndex * 2])
      const cy = dataToCanvasY(coords[focalIndex * 2 + 1])
      const ringR = Math.max(7, Math.min(14, 7 + k * 0.3))
      ctx.strokeStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      ctx.lineWidth = 2.5
      ctx.beginPath()
      ctx.arc(cx, cy, ringR, 0, Math.PI * 2)
      ctx.stroke()
      ctx.fillStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      ctx.beginPath()
      ctx.arc(cx, cy, 3, 0, Math.PI * 2)
      ctx.fill()
    }

    // Query marker (mode 'embedded').
    if (queryCentroid && !queryCentroid.dispersed && props.mode === 'embedded') {
      const cx = dataToCanvasX(queryCentroid.x)
      const cy = dataToCanvasY(queryCentroid.y)
      const r = 8
      ctx.fillStyle = paper
      ctx.beginPath()
      ctx.arc(cx, cy, r + 1, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.stroke()
      ctx.fillStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      ctx.font = 'bold 11px "Special Elite", Georgia, serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('Q', cx, cy + 0.5)
    }
  }, [
    coords,
    size,
    dataToCanvas,
    transformTick,
    highlightedIndices,
    focalIndex,
    queryCentroid,
    props.mode,
    props.drawQueryLines,
    props.highlightedArticles,
    // `domTheme` is the canvas's authoritative theme signal (updated by the
    // MutationObserver that watches <html data-theme>).
    domTheme,
    idToIndex,
    searchMatches,
  ])

  // ===== Light overlay render: just the hover halo.
  // Cheap to redraw on every pointermove — clears + draws one arc.
  useEffect(() => {
    const canvas = overlayCanvasRef.current
    if (!canvas || !size.w || !size.h || !dataToCanvas || !coords) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    canvas.width = Math.round(size.w * size.dpr)
    canvas.height = Math.round(size.h * size.dpr)
    canvas.style.width = `${size.w}px`
    canvas.style.height = `${size.h}px`
    ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0)
    ctx.clearRect(0, 0, size.w, size.h)
    if (hoverIdx === null) return
    const accentRgb = ATLAS_THEME_PALETTE[domTheme].accentRgb
    const tform = transformRef.current
    const { baseScale } = dataToCanvas
    const k = tform.k
    const cx = (size.w / 2 + coords[hoverIdx * 2] * baseScale) * k + tform.x
    const cy = (size.h / 2 + coords[hoverIdx * 2 + 1] * baseScale) * k + tform.y
    ctx.strokeStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(cx, cy, 8, 0, Math.PI * 2)
    ctx.stroke()
  }, [hoverIdx, coords, size, dataToCanvas, transformTick, domTheme])

  // ===== Minimap (embedded + similar modes only): overview rectangle.
  useEffect(() => {
    if (props.mode === 'standalone') return
    const canvas = minimapCanvasRef.current
    if (!canvas || !coords) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const mmW = canvas.clientWidth
    const mmH = canvas.clientHeight
    if (!mmW || !mmH) return
    const dpr = size.dpr || 1
    canvas.width = Math.round(mmW * dpr)
    canvas.height = Math.round(mmH * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const palette = ATLAS_THEME_PALETTE[domTheme]
    const paper = palette.paper
    const inkRgb = palette.inkRgb
    const accentRgb = palette.accentRgb

    ctx.fillStyle = paper
    ctx.fillRect(0, 0, mmW, mmH)

    const pad = 4
    const drawableW = mmW - 2 * pad
    const drawableH = mmH - 2 * pad
    const mmBaseScale = Math.min(drawableW, drawableH) / (2 * COORD_RANGE)
    const ox = mmW / 2
    const oy = mmH / 2
    const toX = (dx: number) => ox + dx * mmBaseScale
    const toY = (dy: number) => oy + dy * mmBaseScale

    // 1) Background points — every article at 1px dimmed.
    ctx.fillStyle = `rgba(${inkRgb[0]},${inkRgb[1]},${inkRgb[2]},0.32)`
    const n = coords.length / 2
    for (let i = 0; i < n; i++) {
      ctx.fillRect(toX(coords[i * 2]), toY(coords[i * 2 + 1]), 1, 1)
    }

    // 2) Highlighted articles — accent, slightly larger.
    if (highlightedIndices && highlightedIndices.size > 0) {
      ctx.fillStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      for (const i of highlightedIndices) {
        ctx.fillRect(toX(coords[i * 2]) - 1, toY(coords[i * 2 + 1]) - 1, 2, 2)
      }
    }

    // 3) Focal article ring.
    if (focalIndex !== null) {
      const fx = toX(coords[focalIndex * 2])
      const fy = toY(coords[focalIndex * 2 + 1])
      ctx.strokeStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(fx, fy, 3.5, 0, Math.PI * 2)
      ctx.stroke()
    }

    // 4) Query marker on minimap.
    if (
      props.mode === 'embedded' &&
      queryCentroid &&
      !queryCentroid.dispersed
    ) {
      ctx.fillStyle = paper
      ctx.beginPath()
      ctx.arc(toX(queryCentroid.x), toY(queryCentroid.y), 3, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(toX(queryCentroid.x), toY(queryCentroid.y), 2.5, 0, Math.PI * 2)
      ctx.stroke()
    }

    // 5) Viewport rectangle — map the visible canvas region back into data
    // space, then onto the minimap.
    if (size.w && size.h) {
      const mainBaseScale = Math.min(size.w, size.h) / (2 * COORD_RANGE)
      const t = transformRef.current
      const k = t.k
      const dataX0 = ((0 - t.x) / k - size.w / 2) / mainBaseScale
      const dataY0 = ((0 - t.y) / k - size.h / 2) / mainBaseScale
      const dataX1 = ((size.w - t.x) / k - size.w / 2) / mainBaseScale
      const dataY1 = ((size.h - t.y) / k - size.h / 2) / mainBaseScale
      const mx0 = Math.max(0, toX(dataX0))
      const my0 = Math.max(0, toY(dataY0))
      const mx1 = Math.min(mmW, toX(dataX1))
      const my1 = Math.min(mmH, toY(dataY1))
      ctx.fillStyle = `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},0.08)`
      ctx.fillRect(mx0, my0, mx1 - mx0, my1 - my0)
      ctx.strokeStyle = `rgb(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]})`
      ctx.lineWidth = 1
      ctx.strokeRect(mx0 + 0.5, my0 + 0.5, mx1 - mx0 - 1, my1 - my0 - 1)
    }
  }, [
    coords,
    size,
    transformTick,
    highlightedIndices,
    focalIndex,
    queryCentroid,
    props.mode,
    domTheme,
  ])

  // (theme handling lives in the `domTheme` state above)

  // Pointer move → hover hit-test
  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!coords || !grid || !size.w || !size.h) return
      const rect = (event.target as HTMLCanvasElement).getBoundingClientRect()
      const px = event.clientX - rect.left
      const py = event.clientY - rect.top
      const tform = transformRef.current
      const baseScale = Math.min(size.w, size.h) / (2 * COORD_RANGE)
      const k = tform.k
      const tx = tform.x
      const ty = tform.y
      // Inverse of dataToCanvas: canvas = (w/2 + data*baseScale) * k + t
      // → data = ((canvas - t) / k - w/2) / baseScale
      const dataX = ((px - tx) / k - size.w / 2) / baseScale
      const dataY = ((py - ty) / k - size.h / 2) / baseScale
      const screenRadius = 14 // px
      const radiusData = screenRadius / (baseScale * k)
      const idx = findNearest(coords, grid, dataX, dataY, radiusData)
      setHoverIdx(idx)
      if (idx !== null) {
        setHoverPos({ x: px, y: py })
      } else {
        setHoverPos(null)
      }
    },
    [coords, grid, size],
  )

  const onPointerLeave = useCallback(() => {
    setHoverIdx(null)
    setHoverPos(null)
  }, [])

  const onClick = useCallback(() => {
    if (hoverIdx === null || !meta) return
    const id = meta.ids[hoverIdx]
    if (props.onPointClick) {
      props.onPointClick(id)
    } else {
      const url = meta.urls[hoverIdx]
      if (url) window.open(url, '_blank', 'noopener,noreferrer')
    }
  }, [hoverIdx, meta, props.onPointClick])

  // Render shell + controls
  const standalone = props.mode === 'standalone'
  const wrapperClass = `atlas-wrapper atlas-mode-${props.mode}`

  return (
    <div className={wrapperClass}>
      {standalone && (
        <>
          <div className="top-rail">
            <button type="button" className="top-rail-brand" onClick={props.onBackToCompose}>hear! hear!</button>
            <div className="top-rail-links">
              <button type="button" onClick={props.onBackToCompose}>search</button>
              <button type="button" className={props.active === 'explore' ? 'active' : ''}>explore</button>
              <button type="button" onClick={props.onOpenAbout}>about</button>
              {props.theme && props.onToggleTheme && (
                <ThemeToggle theme={props.theme} onToggle={props.onToggleTheme} />
              )}
              {props.onOpenTutorial && (
                <button
                  type="button"
                  className="help-toggle"
                  onClick={props.onOpenTutorial}
                  aria-label="Open tutorial"
                  title="Open tutorial"
                >
                  ?
                </button>
              )}
            </div>
          </div>
          <div className="top-rule" />
        </>
      )}

      <div className="atlas-controls">
        {standalone ? (
          <div className="atlas-control-group" role="tablist" aria-label="Embedding source">
            <button
              type="button"
              className={`atlas-pill ${source === 'minilm' ? 'active' : ''}`}
              onClick={() => setSource('minilm')}
            >
              MiniLM
            </button>
            <button
              type="button"
              className={`atlas-pill ${source === 'svd' ? 'active' : ''}`}
              onClick={() => setSource('svd')}
              disabled={!data?.coordsSvd}
              title={!data?.coordsSvd ? 'SVD projection not available' : undefined}
            >
              SVD
            </button>
          </div>
        ) : (
          <span className="atlas-source-label" aria-label="Embedding source">
            <span className="atlas-source-label-model">{source === 'minilm' ? 'MiniLM' : 'SVD'}</span>
            <span className="atlas-source-label-sub">atlas</span>
          </span>
        )}
        {standalone && (
          <input
            className="atlas-search"
            type="search"
            placeholder="highlight by title..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            aria-label="Highlight articles by title substring"
          />
        )}
        {props.queryString && props.mode === 'embedded' && (
          <span className="atlas-query-pill" title={`Search query: ${props.queryString}`}>
            <span className="atlas-query-q">Q</span>
            <span className="atlas-query-text">{props.queryString}</span>
          </span>
        )}
        <span className="atlas-stats">
          {meta ? `${meta.n_docs.toLocaleString()} articles` : loading ? 'loading…' : ''}
          {highlightedIndices ? ` · ${highlightedIndices.size} highlighted` : ''}
        </span>
        {props.onClose && (
          <button type="button" className="atlas-close" onClick={props.onClose} aria-label="Close graph">×</button>
        )}
      </div>

      <div className="atlas-canvas-wrap" ref={containerRef}>
        {error ? (
          <div className="atlas-empty">
            atlas couldn't load: {error}
            <br />
            <span className="atlas-empty-hint">
              Run <code>python -m backend.text_processing.embedding_projection --all</code> to generate the artifacts.
            </span>
          </div>
        ) : (
          <>
            <canvas ref={bgCanvasRef} className="atlas-canvas atlas-canvas-bg" />
            <canvas
              ref={overlayCanvasRef}
              className="atlas-canvas atlas-canvas-overlay"
              onPointerMove={onPointerMove}
              onPointerLeave={onPointerLeave}
              onClick={onClick}
            />
            {props.mode !== 'standalone' && (
              <div className="atlas-minimap" aria-label="atlas overview">
                <canvas ref={minimapCanvasRef} className="atlas-minimap-canvas" />
                <span className="atlas-minimap-label">overview</span>
              </div>
            )}
          </>
        )}
        {hoverIdx !== null && hoverPos && meta && (
          <div
            className="atlas-tooltip"
            style={{
              transform: `translate(${Math.min(size.w - 280, hoverPos.x + 12)}px, ${Math.max(8, hoverPos.y - 8)}px)`,
            }}
          >
            <div className="atlas-tooltip-title">{meta.titles[hoverIdx] || '(untitled)'}</div>
            <div className="atlas-tooltip-meta">
              {meta.years[hoverIdx] ? meta.years[hoverIdx] : ''}
              {meta.years[hoverIdx] && meta.authors?.[hoverIdx] ? ' · ' : ''}
              {meta.authors?.[hoverIdx] || ''}
            </div>
            {meta.keywords?.[hoverIdx] && meta.keywords[hoverIdx].length > 0 && (
              <div className="atlas-tooltip-keywords">
                {meta.keywords[hoverIdx].slice(0, 4).map((k) => (
                  <span key={k} className="atlas-tooltip-keyword">{k}</span>
                ))}
              </div>
            )}
          </div>
        )}
        {loading && !error && (
          <div className="atlas-empty atlas-empty-loading">loading the atlas…</div>
        )}
      </div>

      {standalone && (
        <div className="atlas-footer">
          <button type="button" onClick={props.onBackToCompose} className="btn-stamp" style={{ padding: '6px 14px' }}>
            ← compose
          </button>
        </div>
      )}
    </div>
  )
}
