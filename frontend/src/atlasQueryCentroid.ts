import type { Article } from './types'

export type QueryCentroid = {
  x: number
  y: number
  dispersed: boolean
}

const DISPERSION_THRESHOLD = 0.18 // fraction of int16 coord range

function resolveArticleId(article: Article): string {
  return String(article.id)
}

function pickWeight(article: Article): number {
  const candidates: Array<number | null | undefined> = [
    article.combined_score,
    article.score,
    article.topic_score_normalized,
    article.topic_score,
  ]
  for (const value of candidates) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      return value
    }
  }
  return 1
}

/**
 * Computes the query position in 2D as a weighted centroid of result articles'
 * coordinates. Flags `dispersed` when the standard deviation of the cluster is
 * large (relative to the int16 coord range), so the caller can fall back to
 * a convex-hull treatment instead of drawing a misleading single marker.
 */
export function computeQueryCentroid(
  results: Article[],
  coords: Int16Array | null,
  idToIndex: Map<string, number> | null,
): QueryCentroid | null {
  if (!coords || !idToIndex || results.length === 0) {
    return null
  }

  let sumX = 0
  let sumY = 0
  let sumW = 0
  const xs: number[] = []
  const ys: number[] = []

  for (const article of results) {
    const id = resolveArticleId(article)
    const idx = idToIndex.get(id)
    if (idx === undefined) continue
    const x = coords[idx * 2]
    const y = coords[idx * 2 + 1]
    const w = pickWeight(article)
    sumX += x * w
    sumY += y * w
    sumW += w
    xs.push(x)
    ys.push(y)
  }

  if (sumW <= 0 || xs.length === 0) {
    return null
  }

  const cx = sumX / sumW
  const cy = sumY / sumW

  let varX = 0
  let varY = 0
  for (let i = 0; i < xs.length; i++) {
    varX += (xs[i] - cx) ** 2
    varY += (ys[i] - cy) ** 2
  }
  const stdRadius = Math.sqrt((varX + varY) / xs.length)
  const dispersed = stdRadius / 30000 > DISPERSION_THRESHOLD

  return { x: cx, y: cy, dispersed }
}

/**
 * Convex hull of 2D points (Andrew's monotone chain). Used as the fallback
 * shape when the result cluster is too dispersed for a single query marker.
 */
export function convexHull(points: Array<[number, number]>): Array<[number, number]> {
  if (points.length <= 1) return points.slice()
  const sorted = points
    .slice()
    .sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]))

  const cross = (
    o: [number, number],
    a: [number, number],
    b: [number, number],
  ) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  const lower: Array<[number, number]> = []
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop()
    }
    lower.push(p)
  }
  const upper: Array<[number, number]> = []
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop()
    }
    upper.push(p)
  }
  lower.pop()
  upper.pop()
  return lower.concat(upper)
}
