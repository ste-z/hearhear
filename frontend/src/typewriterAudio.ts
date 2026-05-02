// Tiny synthesizer that fakes a vintage typewriter via the Web Audio API.
// We avoid shipping audio assets — the sounds are short enough that
// stitching a noise burst plus a quick oscillator decay reads as the
// characteristic "click" of a key strike, and a sine ding plus a soft
// clack reads as the carriage bell + return on Enter.

let ctx: AudioContext | null = null

type AudioCtor = typeof AudioContext

function getCtx(): AudioContext | null {
  if (typeof window === 'undefined') return null
  if (ctx) return ctx
  const Ctor: AudioCtor | undefined =
    (window.AudioContext as AudioCtor | undefined) ??
    ((window as unknown as { webkitAudioContext?: AudioCtor }).webkitAudioContext)
  if (!Ctor) return null
  try {
    ctx = new Ctor()
  } catch {
    ctx = null
  }
  return ctx
}

function ensureRunning(c: AudioContext): void {
  // Browsers suspend audio contexts until a user gesture; resume on the
  // first interaction so the very first keystroke isn't silent.
  if (c.state === 'suspended') {
    void c.resume().catch(() => undefined)
  }
}

function makeNoiseBuffer(c: AudioContext, durationSec: number, decay: 'linear' | 'square' = 'linear'): AudioBuffer {
  const length = Math.max(1, Math.floor(c.sampleRate * durationSec))
  const buffer = c.createBuffer(1, length, c.sampleRate)
  const data = buffer.getChannelData(0)
  for (let i = 0; i < length; i += 1) {
    const t = i / length
    const env = decay === 'square' ? (1 - t) * (1 - t) : 1 - t
    data[i] = (Math.random() * 2 - 1) * env
  }
  return buffer
}

export function playKeyStrike(): void {
  const c = getCtx()
  if (!c) return
  ensureRunning(c)
  const t = c.currentTime

  // Master gain so the keystroke sits comfortably under the page UI.
  const master = c.createGain()
  master.gain.value = 0.35
  master.connect(c.destination)

  // High-frequency click — the metal type-bar hitting the platen.
  const noise = c.createBufferSource()
  noise.buffer = makeNoiseBuffer(c, 0.05, 'linear')
  const noiseFilter = c.createBiquadFilter()
  noiseFilter.type = 'highpass'
  noiseFilter.frequency.value = 1600 + Math.random() * 400
  const noiseGain = c.createGain()
  noiseGain.gain.setValueAtTime(0.0001, t)
  noiseGain.gain.exponentialRampToValueAtTime(0.5, t + 0.002)
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.045)
  noise.connect(noiseFilter)
  noiseFilter.connect(noiseGain)
  noiseGain.connect(master)
  noise.start(t)
  noise.stop(t + 0.06)

  // Low thud — the key lever bottoming out on its felt rest.
  const thud = c.createOscillator()
  thud.type = 'triangle'
  // A tiny pitch jitter per keystroke so repeated clicks don't sound mechanical.
  thud.frequency.setValueAtTime(170 + Math.random() * 50, t)
  thud.frequency.exponentialRampToValueAtTime(90, t + 0.06)
  const thudGain = c.createGain()
  thudGain.gain.setValueAtTime(0.0001, t)
  thudGain.gain.exponentialRampToValueAtTime(0.4, t + 0.005)
  thudGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.07)
  thud.connect(thudGain)
  thudGain.connect(master)
  thud.start(t)
  thud.stop(t + 0.08)
}

export function playCarriageReturn(): void {
  // Carriage bell on Enter. Rendered as a "de-ding" — a small,
  // lower-pitched grace note ("de") followed quickly by the main bell
  // ("ding"). The clack of the platen snapping back closes the cue.
  const c = getCtx()
  if (!c) return
  ensureRunning(c)
  const t = c.currentTime

  const master = c.createGain()
  master.gain.value = 0.4
  master.connect(c.destination)

  // Helper: schedule a bell note. `level` is the peak gain; `decay` is
  // the time to fade to silence. A quiet harmonic an octave-and-a-fifth
  // up gives the sine its metallic shimmer.
  const scheduleBell = (start: number, freq: number, level: number, decay: number): void => {
    const osc = c.createOscillator()
    osc.type = 'sine'
    osc.frequency.value = freq
    const gain = c.createGain()
    gain.gain.setValueAtTime(0.0001, start)
    gain.gain.exponentialRampToValueAtTime(level, start + 0.005)
    gain.gain.exponentialRampToValueAtTime(0.0001, start + decay)
    osc.connect(gain)
    gain.connect(master)
    osc.start(start)
    osc.stop(start + decay + 0.02)

    const overtone = c.createOscillator()
    overtone.type = 'sine'
    overtone.frequency.value = freq * 1.5
    const overtoneGain = c.createGain()
    overtoneGain.gain.setValueAtTime(0.0001, start)
    overtoneGain.gain.exponentialRampToValueAtTime(level * 0.32, start + 0.005)
    overtoneGain.gain.exponentialRampToValueAtTime(0.0001, start + decay * 0.55)
    overtone.connect(overtoneGain)
    overtoneGain.connect(master)
    overtone.start(start)
    overtone.stop(start + decay)
  }

  // "de" — softer, lower grace note that anticipates the main bell.
  scheduleBell(t, 1480, 0.3, 0.18)
  // "ding" — the main carriage bell, a perfect fourth above the "de".
  scheduleBell(t + 0.11, 1980, 0.55, 0.5)

  // Carriage clack — the platen snapping back to the left margin a beat
  // after the bell. A short noise burst run through a band-pass filter
  // reads as a wood-on-metal "thunk".
  const clack = c.createBufferSource()
  clack.buffer = makeNoiseBuffer(c, 0.09, 'square')
  const clackFilter = c.createBiquadFilter()
  clackFilter.type = 'bandpass'
  clackFilter.frequency.value = 850
  clackFilter.Q.value = 0.6
  const clackGain = c.createGain()
  const clackStart = t + 0.27
  clackGain.gain.setValueAtTime(0.0001, clackStart)
  clackGain.gain.exponentialRampToValueAtTime(0.7, clackStart + 0.005)
  clackGain.gain.exponentialRampToValueAtTime(0.0001, clackStart + 0.09)
  clack.connect(clackFilter)
  clackFilter.connect(clackGain)
  clackGain.connect(master)
  clack.start(clackStart)
  clack.stop(clackStart + 0.1)
}

export function playSend(): void {
  // The "sent" cue that follows a carriage-return when the slip is being
  // dispatched to the press. A short paper-feed whoosh (descending
  // bandpassed noise) lands into a two-note bell chord — reads like the
  // page being pulled through the platen and the press accepting it.
  const c = getCtx()
  if (!c) return
  ensureRunning(c)
  const t = c.currentTime

  const master = c.createGain()
  master.gain.value = 0.42
  master.connect(c.destination)

  // Paper-feed whoosh: noise with bandpass sweeping down from a bright
  // top end to a warm body, mimicking the carriage feeding a sheet.
  const whoosh = c.createBufferSource()
  whoosh.buffer = makeNoiseBuffer(c, 0.45, 'square')
  const whooshFilter = c.createBiquadFilter()
  whooshFilter.type = 'bandpass'
  whooshFilter.Q.value = 1.4
  whooshFilter.frequency.setValueAtTime(3000, t)
  whooshFilter.frequency.exponentialRampToValueAtTime(500, t + 0.4)
  const whooshGain = c.createGain()
  whooshGain.gain.setValueAtTime(0.0001, t)
  whooshGain.gain.exponentialRampToValueAtTime(0.5, t + 0.05)
  whooshGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.45)
  whoosh.connect(whooshFilter)
  whooshFilter.connect(whooshGain)
  whooshGain.connect(master)
  whoosh.start(t)
  whoosh.stop(t + 0.5)

  // Resolution chord — a perfect fifth bell pair lands a beat after the
  // whoosh starts, giving the "sent" cue a confident finish.
  const chordStart = t + 0.16
  const tones: Array<{ freq: number; level: number }> = [
    { freq: 1320, level: 0.32 },
    { freq: 1980, level: 0.18 },
  ]
  for (const { freq, level } of tones) {
    const osc = c.createOscillator()
    osc.type = 'sine'
    osc.frequency.value = freq
    const gain = c.createGain()
    gain.gain.setValueAtTime(0.0001, chordStart)
    gain.gain.exponentialRampToValueAtTime(level, chordStart + 0.006)
    gain.gain.exponentialRampToValueAtTime(0.0001, chordStart + 0.55)
    osc.connect(gain)
    gain.connect(master)
    osc.start(chordStart)
    osc.stop(chordStart + 0.6)
  }
}
