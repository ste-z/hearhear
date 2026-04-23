import { useEffect, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  BaseEdge,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import './AboutMethodFlow.css'

type SearchMode = 'stance' | 'essay'
type RetrievalMode = 'lexical' | 'semantic' | 'enhanced'
type AgreementMode = 'nli' | 'llm'
type NodeKind = 'artifact' | 'process' | 'input' | 'result' | 'output'
type EdgeTone = 'neutral' | 'active' | 'support'

type AboutMethodFlowProps = {
  mode: SearchMode
  onModeChange?: (mode: SearchMode) => void
}

type DomainNode = {
  description: string
  details: string[]
  h: number
  id: string
  inputs?: string[]
  kind: NodeKind
  notes?: string[]
  outputs?: string[]
  title: string
  w: number
}

type DomainEdge = {
  id: string
  source: string
  sourceHandle: HandleId
  target: string
  targetHandle: HandleId
  tone: EdgeTone
}

type HandleId =
  | 'left-in'
  | 'left-out'
  | 'right-in'
  | 'right-out'
  | 'top-in'
  | 'top-out'
  | 'bottom-in'
  | 'bottom-out'

type MethodNodeData = {
  description: string
  details: string[]
  inputs?: string[]
  kind: NodeKind
  notes?: string[]
  outputs?: string[]
  related: boolean
  title: string
}

type SectionNodeData = {
  title: string
}

type RoutedEdgeData = {
  active: boolean
  points: Array<{ x: number; y: number }>
  tone: EdgeTone
}

type LayoutNodeRecord = {
  data: MethodNodeData
  height: number
  id: string
  position: { x: number; y: number }
  width: number
}

type LayoutEdgeRecord = {
  id: string
  points: Array<{ x: number; y: number }>
  source: string
  sourceHandle: HandleId
  target: string
  targetHandle: HandleId
  tone: EdgeTone
}

type SectionRecord = {
  height: number
  id: string
  position: { x: number; y: number }
  title: string
  width: number
}

type LayoutBox = {
  height: number
  width: number
  x: number
  y: number
}

type RoutePoint = { x: number; y: number }

const searchModes = ['stance', 'essay'] as const satisfies readonly SearchMode[]
const retrievalModes = ['lexical', 'semantic', 'enhanced'] as const satisfies readonly RetrievalMode[]
const agreementModes = ['nli', 'llm'] as const satisfies readonly AgreementMode[]

const searchModeLabels: Record<SearchMode, string> = {
  stance: 'Topic + Stance Search',
  essay: 'Essay-Guided Search',
}

const searchModeNotes: Record<SearchMode, string> = {
  stance: 'Show the compact prompt path that starts from a topic and an opinion.',
  essay: 'Show the longer writing path that starts from a full draft and can extract a thesis.',
}

const retrievalModeLabels: Record<RetrievalMode, string> = {
  lexical: 'Lexical',
  semantic: 'Semantic',
  enhanced: 'Enhanced Semantic',
}

const retrievalModeNotes: Record<RetrievalMode, string> = {
  lexical: 'Use TF-IDF vectors and an inverted index for exact-term topic matching.',
  semantic: 'Use truncated-SVD latent vectors for broader topical similarity.',
  enhanced: 'Use MiniLM embeddings for the most meaning-driven Stage 1 branch.',
}

const agreementModeLabels: Record<AgreementMode, string> = {
  nli: 'NLI Agreement',
  llm: 'LLM Agreement',
}

const agreementModeNotes: Record<AgreementMode, string> = {
  nli: 'Route the candidate set into claim pairing and DeBERTa-style stance comparison.',
  llm: 'Route the candidate set into article-context prompts for Spark-based agreement scoring.',
}

const nodeKindLabels: Record<NodeKind, string> = {
  artifact: 'Data / artifact',
  process: 'Process / method',
  input: 'User input',
  result: 'Data / result',
  output: 'User-facing output',
}

const sectionLabels = [
  'Source + Precompute',
  'Live Ranking Path',
  'Interpretation Layer',
] as const

const shapeLegend = [
  { kind: 'artifact', label: 'Data / artifact' },
  { kind: 'process', label: 'Process / method' },
  { kind: 'input', label: 'Input' },
  { kind: 'result', label: 'Score / result' },
  { kind: 'output', label: 'Output' },
] as const satisfies ReadonlyArray<{ kind: NodeKind; label: string }>

const handles: Array<{ id: HandleId; position: Position; type: 'source' | 'target' }> = [
  { id: 'left-in', position: Position.Left, type: 'target' },
  { id: 'left-out', position: Position.Left, type: 'source' },
  { id: 'right-in', position: Position.Right, type: 'target' },
  { id: 'right-out', position: Position.Right, type: 'source' },
  { id: 'top-in', position: Position.Top, type: 'target' },
  { id: 'top-out', position: Position.Top, type: 'source' },
  { id: 'bottom-in', position: Position.Bottom, type: 'target' },
  { id: 'bottom-out', position: Position.Bottom, type: 'source' },
]

const edgeToneStyles: Record<EdgeTone, { color: string; opacity: number }> = {
  neutral: { color: 'rgba(214, 221, 230, 0.72)', opacity: 0.4 },
  active: { color: 'rgba(241, 220, 197, 0.92)', opacity: 0.46 },
  support: { color: 'rgba(214, 221, 230, 0.72)', opacity: 0.4 },
}

const markerByTone = (tone: EdgeTone) => ({
  color: edgeToneStyles[tone].color,
  height: 18,
  type: MarkerType.ArrowClosed,
  width: 18,
})

function MethodNode({ data }: NodeProps<Node<MethodNodeData>>): JSX.Element {
  return (
    <div className={`about-flow-node kind-${data.kind} ${data.related ? 'related' : ''}`}>
      {handles.map((handle) => (
        <Handle
          key={handle.id}
          id={handle.id}
          type={handle.type}
          position={handle.position}
          isConnectable={false}
          className="about-flow-handle"
        />
      ))}

      <span className="about-flow-node-eyebrow">{nodeKindLabels[data.kind]}</span>
      <strong>{data.title}</strong>
      <span className="about-flow-node-description">{data.description}</span>
    </div>
  )
}

function SectionNode({ data }: NodeProps<Node<SectionNodeData>>): JSX.Element {
  return (
    <div className="about-flow-section-node">
      <span>{data.title}</span>
    </div>
  )
}

function RoutedEdge({
  data,
  id,
  markerEnd,
}: EdgeProps<Edge<RoutedEdgeData>>): JSX.Element | null {
  if (!data || data.points.length < 2) {
    return null
  }

  const path = data.points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x},${point.y}`)
    .join(' ')

  const toneStyle = edgeToneStyles[data.tone]

  return (
    <BaseEdge
      id={id}
      path={path}
      markerEnd={markerEnd}
      style={{
        opacity: data.active ? 1 : toneStyle.opacity,
        stroke: toneStyle.color,
        strokeWidth: data.active ? 3.8 : 2.8,
      }}
    />
  )
}

const nodeTypes = {
  method: MethodNode,
  section: SectionNode,
}

const edgeTypes = {
  routed: RoutedEdge,
}

const baseNodes = (): DomainNode[] => [
  {
    description: 'The Guardian opinion archive is the shared source corpus for every retrieval, ranking, and explanation step.',
    details: [
      'All branches begin from the same article collection.',
      'Metadata such as publication date can later become part of the final ranking.',
    ],
    h: 134,
    id: 'corpus',
    kind: 'artifact',
    notes: ['Guardian archive'],
    outputs: ['Article text', 'Metadata', 'Publication dates'],
    title: 'Guardian opinion archive',
    w: 320,
  },
  {
    description: 'Publication dates are normalized into a reusable recency feature that can lift newer articles during final ranking.',
    details: [
      'Recency stays separate from topic and agreement scoring.',
      'This makes freshness an explicit ranking input instead of a hidden assumption.',
    ],
    h: 122,
    id: 'recency',
    inputs: ['Publication dates'],
    kind: 'artifact',
    notes: ['Freshness signal'],
    outputs: ['Recency feature'],
    title: 'Recency signal',
    w: 300,
  },
  {
    description: 'An LLM condenses long articles into shorter claim-like snippets that can be compared and synthesized later.',
    details: [
      'This makes the processing step explicit instead of hiding it inside the summaries artifact.',
      'The same summary bank later supports NLI-style agreement and the post-ranking overview.',
    ],
    h: 138,
    id: 'summary-process',
    inputs: ['Guardian article text'],
    kind: 'process',
    notes: ['LLM'],
    outputs: ['Claim-style summary generation'],
    title: 'LLM summary generation',
    w: 330,
  },
  {
    description: 'The generated claim-like snippets become a reusable article-summary bank for later reasoning steps.',
    details: [
      'This is a stored artifact rather than a method.',
      'It is most visible when Stage 2 uses NLI and when the interpretation layer writes a debate overview.',
    ],
    h: 134,
    id: 'summaries',
    inputs: ['LLM-generated article summaries'],
    kind: 'artifact',
    notes: ['Claim bank'],
    outputs: ['Claim-style summaries', 'Short article representations'],
    title: 'Article summaries',
    w: 340,
  },
  {
    description: 'Stage 1 produces a topic-relevance score for each article in the corpus.',
    details: [
      'This is a data artifact, not a method.',
      'It becomes the input to candidate filtering before any agreement scoring runs.',
    ],
    h: 122,
    id: 'topic-scores',
    inputs: ['Cosine retrieval outputs'],
    kind: 'artifact',
    notes: ['Per-article scores'],
    outputs: ['Topic relevance scores'],
    title: 'Topic relevance scores',
    w: 300,
  },
  {
    description: 'The topic-score distribution is filtered so only promising articles continue to the more expensive second stage.',
    details: [
      'This is a process block because it decides which articles continue.',
      'It corresponds to the top-k or threshold gate in the larger method.',
    ],
    h: 136,
    id: 'candidate-selection',
    inputs: ['Topic relevance scores'],
    kind: 'process',
    notes: ['Top-k', 'Threshold'],
    outputs: ['Candidate filtering decision'],
    title: 'Candidate filtering',
    w: 320,
  },
  {
    description: 'The filtered candidate set is a reusable article subset that feeds both Stage 2 and the final merger.',
    details: [
      'This artifact separates the selected article set from the filtering process that produced it.',
      'That separation makes the ranking arrows much cleaner and more explainable.',
    ],
    h: 124,
    id: 'candidate-set',
    inputs: ['Filtered article ids'],
    kind: 'artifact',
    notes: ['Stage 1 survivors'],
    outputs: ['Candidate articles'],
    title: 'Candidate set',
    w: 320,
  },
  {
    description: 'The final ranking method merges topic relevance, agreement, and optional recency into one overall score.',
    details: [
      'This block is the score merger itself, not the ranked result artifact.',
      'The incoming arrows show the three score ingredients: topic relevance, stance agreement, and recency.',
    ],
    h: 144,
    id: 'ranking',
    inputs: ['Topic relevance scores', 'Stance agreement scores', 'Recency signal'],
    kind: 'process',
    notes: ['Weighted merge'],
    outputs: ['Merged ranking score'],
    title: 'Weighted score merger',
    w: 320,
  },
  {
    description: 'The final ranked set is turned into a concrete result artifact that downstream AI tools can read.',
    details: [
      'This block separates the ranking method from the ranked article artifact.',
      'That separation keeps the interpretation arrows out of the ranking process box.',
    ],
    h: 134,
    id: 'ranked-results',
    inputs: ['Merged ranking score'],
    kind: 'output',
    notes: ['Search results'],
    outputs: ['Ordered Guardian articles'],
    title: 'Ranked articles',
    w: 320,
  },
  {
    description: 'A post-ranking LLM reads the ranked articles and writes a user-friendly overview of support, challenge, and nuance.',
    details: [
      'This block is a process, not a stored artifact.',
      'It explains the results after ranking instead of affecting the score order.',
    ],
    h: 140,
    id: 'overview',
    inputs: ['Ranked articles', 'Article summaries'],
    kind: 'process',
    notes: ['LLM synthesis'],
    outputs: ['Narrative overview', 'Grouped viewpoints'],
    title: 'LLM results overview',
    w: 320,
  },
  {
    description: 'The ranked results can also power an interactive chat layer for follow-up questions about the retrieved articles.',
    details: [
      'This is a post-ranking process, not a ranking method.',
      'It keeps responses grounded in the already ranked article set.',
    ],
    h: 140,
    id: 'chat',
    inputs: ['Ranked articles', 'User follow-up questions'],
    kind: 'process',
    notes: ['Results chat'],
    outputs: ['Source-linked answers'],
    title: 'Follow-up chat',
    w: 320,
  },
]

const getRetrievalArtifactNode = (retrievalMode: RetrievalMode): DomainNode => {
  if (retrievalMode === 'lexical') {
    return {
      description: 'TF-IDF vectors and a lexical index capture exact-term overlap for the most literal Stage 1 branch.',
      details: [
        'This corresponds to the TF-IDF and inverted-index path in your original diagram.',
        'It is a reusable retrieval artifact rather than the retrieval method itself.',
      ],
      h: 142,
      id: 'artifact',
      inputs: ['Guardian article text'],
      kind: 'artifact',
      notes: ['TF-IDF'],
      outputs: ['TF-IDF vectors', 'Lexical index'],
      title: 'Lexical retrieval index',
      w: 340,
    }
  }

  if (retrievalMode === 'semantic') {
    return {
      description: 'Truncated-SVD projections compress the corpus into a latent semantic space for broader topic matching.',
      details: [
        'This is the semantic artifact branch derived from the TF-IDF representation.',
        'It stores the searchable latent space rather than performing the search itself.',
      ],
      h: 142,
      id: 'artifact',
      inputs: ['TF-IDF term-document matrix'],
      kind: 'artifact',
      notes: ['SVD'],
      outputs: ['Latent semantic vectors', 'SVD index'],
      title: 'Semantic latent space',
      w: 340,
    }
  }

  return {
    description: 'MiniLM embeddings map articles into a dense meaning space that can be searched semantically at query time.',
    details: [
      'This is the strongest meaning-based retrieval artifact in the chart.',
      'It stores dense vectors rather than being the retrieval method itself.',
    ],
    h: 142,
    id: 'artifact',
    inputs: ['Guardian article text or semantic chunks'],
    kind: 'artifact',
    notes: ['MiniLM'],
    outputs: ['Dense embeddings', 'Embedding index'],
    title: 'Embedding index',
    w: 340,
  }
}

const getSearchNodes = (
  searchMode: SearchMode,
  agreementMode: AgreementMode,
): DomainNode[] => {
  if (searchMode === 'stance') {
    return [
      {
        description: 'The stance workflow starts from the compact prompt already used in the interface: a topic plus a position statement.',
        details: [
          'This is the main user-provided input artifact.',
          'It gives the system both a topical anchor and a stance signal from the start.',
        ],
        h: 132,
        id: 'input',
        kind: 'input',
        notes: ['User prompt'],
        outputs: ['Topic + stance query'],
        title: 'Topic + stance prompt',
        w: 320,
      },
      {
        description: agreementMode === 'llm'
          ? 'The prompt can be cleaned and then reused both for retrieval and for Stage 2 LLM scoring.'
          : 'The prompt can be cleaned before retrieval, correcting typos or sharpening wording without taking control away from the user.',
        details: [
          'This is a process block because it transforms the query.',
          'The cleaned prompt can continue into Stage 1 retrieval and Stage 2 setup.',
        ],
        h: 126,
        id: 'refine',
        inputs: ['Topic + stance query'],
        kind: 'process',
        notes: ['Query cleanup'],
        outputs: ['Cleaned stance query'],
        title: 'Query cleanup',
        w: 320,
      },
    ]
  }

  if (agreementMode === 'llm') {
    return [
      {
        description: 'The essay workflow begins with the full draft, which can directly drive both retrieval and Stage 2 LLM scoring.',
        details: [
          'This is the user-provided input artifact.',
          'When LLM agreement is selected, the full essay can remain the Stage 2 context without extracting a thesis sentence.',
        ],
        h: 132,
        id: 'input',
        kind: 'input',
        notes: ['Full essay'],
        outputs: ['Essay draft'],
        title: 'Essay intake',
        w: 320,
      },
    ]
  }

  return [
    {
      description: 'The essay workflow begins with the full draft, which drives topic retrieval before a compact thesis is selected.',
      details: [
        'This is the main essay input artifact.',
        'The whole draft stays useful for Stage 1 even though Stage 2 NLI will later use a thesis sentence.',
      ],
      h: 132,
      id: 'input',
      kind: 'input',
      notes: ['Full essay'],
      outputs: ['Essay draft'],
      title: 'Essay intake',
      w: 320,
    },
    {
      description: 'A thesis sentence is proposed from the draft and then confirmed or overridden by the user before NLI agreement scoring.',
      details: [
        'This is a processing block because it transforms the essay into a compact claim representation.',
        'It condenses the claimness-scoring and user-choice loop from the original chart.',
      ],
      h: 126,
      id: 'refine',
      inputs: ['Essay draft'],
      kind: 'process',
      notes: ['Thesis selection'],
      outputs: ['Selected thesis sentence'],
      title: 'Thesis selection',
      w: 320,
    },
  ]
}

const getRetrievalProcessNode = (
  searchMode: SearchMode,
  retrievalMode: RetrievalMode,
): DomainNode => {
  const title = (
    retrievalMode === 'lexical'
      ? 'Cosine retrieval over TF-IDF'
      : retrievalMode === 'semantic'
        ? 'Cosine retrieval over SVD space'
        : 'Cosine retrieval over embeddings'
  )

  const description = (
    retrievalMode === 'lexical'
      ? 'The cleaned query is compared against TF-IDF article vectors so exact terms and distinctive vocabulary drive topic relevance.'
      : retrievalMode === 'semantic'
        ? 'The query is compared inside a latent semantic space so related topics can surface even without exact phrase overlap.'
        : 'The query is compared inside a dense embedding space so semantic closeness can drive topic retrieval.'
  )

  const details = (
    retrievalMode === 'lexical'
      ? [
          'This is the most literal Stage 1 method.',
          'The retrieval artifact stays separate from the cosine-similarity method used to score it.',
        ]
      : retrievalMode === 'semantic'
        ? [
            'This is the semantic Stage 1 method over an SVD-based artifact.',
            'The method is still cosine-style comparison even though the artifact is different.',
          ]
        : [
            'This is the embedding-based Stage 1 method.',
            'The method compares dense vectors rather than exact term weights.',
          ]
  )

  return {
    description,
    details,
    h: 150,
    id: 'retrieval',
    inputs: [
      searchMode === 'stance' ? 'Cleaned stance query' : 'Essay draft',
      retrievalMode === 'lexical'
        ? 'TF-IDF lexical index'
        : retrievalMode === 'semantic'
          ? 'SVD semantic space'
          : 'Embedding index',
    ],
    kind: 'process',
    notes: ['Cosine similarity'],
    outputs: ['Topic similarity computation'],
    title,
    w: 380,
  }
}

const getStageTwoPreparationNode = (
  searchMode: SearchMode,
  agreementMode: AgreementMode,
): DomainNode => {
  if (agreementMode === 'nli') {
    return {
      description: searchMode === 'stance'
        ? 'The candidate set is paired with the user stance and article-summary claims to prepare premise-hypothesis comparisons.'
        : 'The candidate set is paired with the selected thesis sentence and article-summary claims to prepare premise-hypothesis comparisons.',
      details: [
        'This is a process block that constructs Stage 2 inputs.',
        'The summaries remain a separate artifact, while the pairing itself is the method.',
      ],
      h: 140,
      id: 'stage-two-prep',
      inputs: [
        'Candidate set',
        searchMode === 'stance' ? 'Cleaned stance query' : 'Selected thesis sentence',
        'Article summaries',
      ],
      kind: 'process',
      notes: ['Pair construction'],
      outputs: ['Premise / hypothesis pairs'],
      title: 'Claim pairing',
      w: 340,
    }
  }

  return {
    description: searchMode === 'stance'
      ? 'The candidate set is packaged with the cleaned stance prompt so an LLM can judge support, contradiction, and nuance.'
      : 'The candidate set is packaged with the full essay so an LLM can judge support, contradiction, and nuance in context.',
    details: [
      'This is a process block that assembles the LLM comparison context.',
      'It makes the prompt-building step explicit instead of hiding it inside the scorer.',
    ],
    h: 140,
    id: 'stage-two-prep',
    inputs: [
      'Candidate set',
      searchMode === 'stance' ? 'Cleaned stance query' : 'Essay draft',
      'Retrieved article context',
    ],
    kind: 'process',
    notes: ['Prompt assembly'],
    outputs: ['LLM comparison prompts'],
    title: 'Prompt assembly',
    w: 340,
  }
}

const getAgreementNode = (
  searchMode: SearchMode,
  agreementMode: AgreementMode,
): DomainNode => {
  if (agreementMode === 'nli') {
    return {
      description: searchMode === 'stance'
        ? 'An NLI model rescales candidate articles by whether they support, challenge, or stay neutral toward the user stance.'
        : 'An NLI model rescales candidate articles by whether they support, challenge, or stay neutral toward the selected thesis.',
      details: [
        'This block is the Stage 2 method itself.',
        'It consumes paired claims and produces agreement judgments.',
      ],
      h: 140,
      id: 'agreement',
      inputs: ['Premise / hypothesis pairs'],
      kind: 'process',
      notes: ['NLI', 'DeBERTa'],
      outputs: ['Per-candidate stance judgments'],
      title: 'NLI agreement scorer',
      w: 300,
    }
  }

  return {
    description: searchMode === 'stance'
      ? 'An LLM rescales candidate articles by interpreting their relationship to the user’s stated position.'
      : 'An LLM rescales candidate articles by interpreting their relationship to the full essay argument and context.',
    details: [
      'This block is the Stage 2 method itself.',
      'It uses richer context than NLI, which can help capture nuance beyond short claim pairs.',
    ],
    h: 140,
    id: 'agreement',
    inputs: ['LLM comparison prompts'],
    kind: 'process',
    notes: ['LLM', 'Spark'],
    outputs: ['Per-candidate stance judgments'],
    title: 'LLM agreement scorer',
    w: 300,
  }
}

const getAgreementScoresNode = (agreementMode: AgreementMode): DomainNode => ({
  description: agreementMode === 'nli'
    ? 'The NLI judgments become per-candidate stance agreement scores for the final weighted merge.'
    : 'The LLM judgments become per-candidate stance agreement scores for the final weighted merge.',
  details: [
    'This is a scoring result, not the scoring method.',
    'It makes the final score inputs explicit beside topic relevance and recency.',
  ],
  h: 112,
  id: 'agreement-scores',
  inputs: ['Per-candidate stance judgments'],
  kind: 'result',
  notes: [agreementMode === 'nli' ? 'NLI output' : 'LLM output'],
  outputs: ['Stance agreement scores'],
  title: 'Stance agreement scores',
  w: 280,
})

const buildDomainGraph = (
  searchMode: SearchMode,
  retrievalMode: RetrievalMode,
  agreementMode: AgreementMode,
): { edges: DomainEdge[]; nodes: DomainNode[] } => {
  const nodes = [
    ...baseNodes(),
    getRetrievalArtifactNode(retrievalMode),
    ...getSearchNodes(searchMode, agreementMode),
    getRetrievalProcessNode(searchMode, retrievalMode),
    getStageTwoPreparationNode(searchMode, agreementMode),
    getAgreementNode(searchMode, agreementMode),
    getAgreementScoresNode(agreementMode),
  ]

  const edges: DomainEdge[] = [
    {
      id: 'corpus-to-artifact',
      source: 'corpus',
      sourceHandle: 'bottom-out',
      target: 'artifact',
      targetHandle: 'top-in',
      tone: 'active',
    },
    {
      id: 'corpus-to-summary-process',
      source: 'corpus',
      sourceHandle: 'left-out',
      target: 'summary-process',
      targetHandle: 'left-in',
      tone: 'support',
    },
    {
      id: 'summary-process-to-summaries',
      source: 'summary-process',
      sourceHandle: 'bottom-out',
      target: 'summaries',
      targetHandle: 'top-in',
      tone: 'support',
    },
    {
      id: 'artifact-to-retrieval',
      source: 'artifact',
      sourceHandle: 'right-out',
      target: 'retrieval',
      targetHandle: 'left-in',
      tone: 'active',
    },
    {
      id: 'retrieval-to-topic-scores',
      source: 'retrieval',
      sourceHandle: 'right-out',
      target: 'topic-scores',
      targetHandle: 'left-in',
      tone: 'active',
    },
    {
      id: 'topic-scores-to-candidate-selection',
      source: 'topic-scores',
      sourceHandle: 'bottom-out',
      target: 'candidate-selection',
      targetHandle: 'top-in',
      tone: 'neutral',
    },
    {
      id: 'candidate-selection-to-candidate-set',
      source: 'candidate-selection',
      sourceHandle: 'bottom-out',
      target: 'candidate-set',
      targetHandle: 'top-in',
      tone: 'neutral',
    },
    {
      id: 'candidate-set-to-stage-two-prep',
      source: 'candidate-set',
      sourceHandle: 'right-out',
      target: 'stage-two-prep',
      targetHandle: 'left-in',
      tone: 'neutral',
    },
    {
      id: 'stage-two-prep-to-agreement',
      source: 'stage-two-prep',
      sourceHandle: 'bottom-out',
      target: 'agreement',
      targetHandle: 'top-in',
      tone: 'active',
    },
    {
      id: 'agreement-to-agreement-scores',
      source: 'agreement',
      sourceHandle: 'bottom-out',
      target: 'agreement-scores',
      targetHandle: 'top-in',
      tone: 'active',
    },
    {
      id: 'topic-scores-to-ranking',
      source: 'topic-scores',
      sourceHandle: 'right-out',
      target: 'ranking',
      targetHandle: 'top-in',
      tone: 'active',
    },
    {
      id: 'agreement-scores-to-ranking',
      source: 'agreement-scores',
      sourceHandle: 'right-out',
      target: 'ranking',
      targetHandle: 'left-in',
      tone: 'active',
    },
    {
      id: 'recency-to-ranking',
      source: 'recency',
      sourceHandle: 'right-out',
      target: 'ranking',
      targetHandle: 'bottom-in',
      tone: 'support',
    },
    {
      id: 'ranking-to-ranked-results',
      source: 'ranking',
      sourceHandle: 'right-out',
      target: 'ranked-results',
      targetHandle: 'left-in',
      tone: 'active',
    },
    {
      id: 'ranked-results-to-overview',
      source: 'ranked-results',
      sourceHandle: 'top-out',
      target: 'overview',
      targetHandle: 'bottom-in',
      tone: 'support',
    },
    {
      id: 'ranked-results-to-chat',
      source: 'ranked-results',
      sourceHandle: 'bottom-out',
      target: 'chat',
      targetHandle: 'top-in',
      tone: 'support',
    },
  ]

  if (searchMode === 'stance') {
    edges.push(
      {
        id: 'input-to-refine',
        source: 'input',
        sourceHandle: 'bottom-out',
        target: 'refine',
        targetHandle: 'top-in',
        tone: 'support',
      },
      {
        id: 'refine-to-retrieval',
        source: 'refine',
        sourceHandle: 'bottom-out',
        target: 'retrieval',
        targetHandle: 'top-in',
        tone: 'neutral',
      },
    )
  } else if (agreementMode === 'nli') {
    edges.push(
      {
        id: 'input-to-retrieval',
        source: 'input',
        sourceHandle: 'left-out',
        target: 'retrieval',
        targetHandle: 'left-in',
        tone: 'neutral',
      },
      {
        id: 'input-to-refine',
        source: 'input',
        sourceHandle: 'bottom-out',
        target: 'refine',
        targetHandle: 'top-in',
        tone: 'support',
      },
    )
  } else {
    edges.push(
      {
        id: 'input-to-retrieval',
        source: 'input',
        sourceHandle: 'left-out',
        target: 'retrieval',
        targetHandle: 'left-in',
        tone: 'neutral',
      },
    )
  }
  return { nodes, edges }
}

const sectionLayout: SectionRecord[] = [
  {
    height: 940,
    id: 'section-source',
    position: { x: 40, y: 40 },
    title: sectionLabels[0],
    width: 320,
  },
  {
    height: 940,
    id: 'section-live',
    position: { x: 400, y: 40 },
    title: sectionLabels[1],
    width: 1420,
  },
  {
    height: 940,
    id: 'section-interpretation',
    position: { x: 1860, y: 40 },
    title: sectionLabels[2],
    width: 360,
  },
]

const methodNodeLayout: Record<string, LayoutBox> = {
  agreement: { height: 124, width: 260, x: 1200, y: 545 },
  'agreement-scores': { height: 112, width: 260, x: 1200, y: 735 },
  artifact: { height: 116, width: 250, x: 75, y: 260 },
  'candidate-selection': { height: 118, width: 250, x: 795, y: 435 },
  'candidate-set': { height: 110, width: 250, x: 795, y: 650 },
  chat: { height: 124, width: 270, x: 1910, y: 745 },
  corpus: { height: 108, width: 250, x: 75, y: 105 },
  input: { height: 112, width: 250, x: 445, y: 105 },
  overview: { height: 124, width: 270, x: 1910, y: 250 },
  'ranked-results': { height: 118, width: 270, x: 1910, y: 545 },
  ranking: { height: 126, width: 280, x: 1530, y: 735 },
  recency: { height: 110, width: 250, x: 75, y: 770 },
  refine: { height: 110, width: 250, x: 445, y: 275 },
  retrieval: { height: 124, width: 250, x: 445, y: 470 },
  'stage-two-prep': { height: 124, width: 260, x: 1200, y: 330 },
  summaries: { height: 110, width: 250, x: 75, y: 620 },
  'summary-process': { height: 110, width: 250, x: 75, y: 450 },
  'topic-scores': { height: 110, width: 250, x: 795, y: 275 },
}

const manualEdgeWaypoints: Record<string, (start: RoutePoint, end: RoutePoint) => RoutePoint[]> = {
  'corpus-to-summary-process': (start, end) => [
    { x: 56, y: start.y },
    { x: 56, y: end.y },
  ],
  'candidate-set-to-stage-two-prep': (start, end) => [
    { x: 1110, y: start.y },
    { x: 1110, y: end.y },
  ],
  'input-to-retrieval': (start, end) => [
    { x: 415, y: start.y },
    { x: 415, y: end.y },
  ],
  'topic-scores-to-ranking': (start, end) => [
    { x: end.x, y: start.y },
  ],
  'recency-to-ranking': (start, end) => [
    { x: 380, y: start.y },
    { x: 380, y: 925 },
    { x: end.x, y: 925 },
  ],
  'topic-scores-to-candidate-selection': (start, end) => [
    { x: start.x, y: 405 },
    { x: end.x, y: 405 },
  ],
}

const isHorizontalHandle = (handleId: HandleId): boolean => (
  handleId === 'left-in'
  || handleId === 'left-out'
  || handleId === 'right-in'
  || handleId === 'right-out'
)

const getHandlePoint = (node: LayoutNodeRecord, handleId: HandleId): RoutePoint => {
  const { x, y } = node.position
  const centerX = x + node.width / 2
  const centerY = y + node.height / 2

  if (handleId === 'left-in' || handleId === 'left-out') {
    return { x, y: centerY }
  }

  if (handleId === 'right-in' || handleId === 'right-out') {
    return { x: x + node.width, y: centerY }
  }

  if (handleId === 'top-in' || handleId === 'top-out') {
    return { x: centerX, y }
  }

  return { x: centerX, y: y + node.height }
}

const isSamePoint = (first: RoutePoint, second: RoutePoint): boolean => (
  first.x === second.x && first.y === second.y
)

const isCollinear = (previous: RoutePoint, current: RoutePoint, next: RoutePoint): boolean => (
  (previous.x === current.x && current.x === next.x)
  || (previous.y === current.y && current.y === next.y)
)

const cleanRoutePoints = (points: RoutePoint[]): RoutePoint[] => (
  points.reduce<RoutePoint[]>((cleaned, point) => {
    const previous = cleaned[cleaned.length - 1]
    if (previous && isSamePoint(previous, point)) {
      return cleaned
    }

    const beforePrevious = cleaned[cleaned.length - 2]
    if (beforePrevious && previous && isCollinear(beforePrevious, previous, point)) {
      return [...cleaned.slice(0, -1), point]
    }

    return [...cleaned, point]
  }, [])
)

const defaultEdgeWaypoints = (
  edge: DomainEdge,
  start: RoutePoint,
  end: RoutePoint,
): RoutePoint[] => {
  const sourceIsHorizontal = isHorizontalHandle(edge.sourceHandle)
  const targetIsHorizontal = isHorizontalHandle(edge.targetHandle)

  if (sourceIsHorizontal && targetIsHorizontal) {
    const midX = (start.x + end.x) / 2
    return [
      { x: midX, y: start.y },
      { x: midX, y: end.y },
    ]
  }

  if (!sourceIsHorizontal && !targetIsHorizontal) {
    const midY = (start.y + end.y) / 2
    return [
      { x: start.x, y: midY },
      { x: end.x, y: midY },
    ]
  }

  if (sourceIsHorizontal) {
    return [{ x: end.x, y: start.y }]
  }

  return [{ x: start.x, y: end.y }]
}

const routeEdge = (
  edge: DomainEdge,
  nodesById: Map<string, LayoutNodeRecord>,
): RoutePoint[] => {
  const sourceNode = nodesById.get(edge.source)
  const targetNode = nodesById.get(edge.target)

  if (!sourceNode || !targetNode) {
    return []
  }

  const start = getHandlePoint(sourceNode, edge.sourceHandle)
  const end = getHandlePoint(targetNode, edge.targetHandle)
  const waypoints = manualEdgeWaypoints[edge.id]?.(start, end) ?? defaultEdgeWaypoints(edge, start, end)

  return cleanRoutePoints([start, ...waypoints, end])
}

const decorateFlow = (
  layoutNodes: LayoutNodeRecord[],
  layoutEdges: LayoutEdgeRecord[],
  layoutSections: SectionRecord[],
  activeNodeId: string,
): {
  edges: Edge<RoutedEdgeData>[]
  methodNodes: Node<MethodNodeData>[]
  nodes: Array<Node<MethodNodeData> | Node<SectionNodeData>>
} => {
  const relatedNodeIds = new Set<string>([activeNodeId])
  const relatedEdgeIds = new Set<string>()

  layoutEdges.forEach((edge) => {
    if (edge.source === activeNodeId || edge.target === activeNodeId) {
      relatedNodeIds.add(edge.source)
      relatedNodeIds.add(edge.target)
      relatedEdgeIds.add(edge.id)
    }
  })

  const sectionNodes: Array<Node<SectionNodeData>> = layoutSections.map((section) => ({
    data: {
      title: section.title,
    },
    draggable: false,
    id: section.id,
    position: section.position,
    selectable: false,
    style: {
      height: section.height,
      width: section.width,
    },
    type: 'section',
    zIndex: 0,
  }))

  const methodNodes: Array<Node<MethodNodeData>> = layoutNodes.map((node) => ({
    data: {
      ...node.data,
      kind: node.data.kind,
      related: relatedNodeIds.has(node.id),
    },
    draggable: false,
    id: node.id,
    position: node.position,
    selectable: false,
    style: {
      height: node.height,
      width: node.width,
    },
    type: 'method',
    zIndex: 2,
  }))

  return {
    edges: layoutEdges.map((edge) => ({
      animated: false,
      data: {
        active: relatedEdgeIds.has(edge.id),
        points: edge.points,
        tone: edge.tone,
      },
      id: edge.id,
      markerEnd: markerByTone(edge.tone),
      source: edge.source,
      sourceHandle: edge.sourceHandle,
      target: edge.target,
      targetHandle: edge.targetHandle,
      type: 'routed',
    })),
    methodNodes,
    nodes: [...sectionNodes, ...methodNodes],
  }
}

const layoutGraph = async (
  searchMode: SearchMode,
  retrievalMode: RetrievalMode,
  agreementMode: AgreementMode,
): Promise<{ edges: LayoutEdgeRecord[]; nodes: LayoutNodeRecord[]; sections: SectionRecord[] }> => {
  const domain = buildDomainGraph(searchMode, retrievalMode, agreementMode)

  const nodes = domain.nodes.map((node) => {
    const layoutNode = methodNodeLayout[node.id] ?? {
      height: node.h,
      width: node.w,
      x: 0,
      y: 0,
    }

    return {
      data: {
        description: node.description,
        details: node.details,
        inputs: node.inputs,
        kind: node.kind,
        notes: node.notes,
        outputs: node.outputs,
        related: false,
        title: node.title,
      },
      height: layoutNode.height,
      id: node.id,
      position: {
        x: layoutNode.x,
        y: layoutNode.y,
      },
      width: layoutNode.width,
    }
  })

  const nodesById = new Map(nodes.map((node) => [node.id, node]))

  const edges = domain.edges.map((edge) => (
    {
      id: edge.id,
      points: routeEdge(edge, nodesById),
      source: edge.source,
      sourceHandle: edge.sourceHandle,
      target: edge.target,
      targetHandle: edge.targetHandle,
      tone: edge.tone,
    }
  ))

  return { edges, nodes, sections: sectionLayout }
}

const renderControlGroup = <T extends string>(
  label: string,
  options: readonly T[],
  activeOption: T,
  onChange: (option: T) => void,
  labels: Record<T, string>,
  note: string,
): JSX.Element => (
  <section className="about-system-control-group">
    <div className="about-system-control-copy">
      <span>{label}</span>
      <p>{note}</p>
    </div>

    <div className="about-system-selector" role="tablist" aria-label={label}>
      {options.map((option) => (
        <button
          key={option}
          type="button"
          role="tab"
          aria-selected={activeOption === option}
          className={`about-system-selector-button ${activeOption === option ? 'active' : ''}`}
          onClick={() => onChange(option)}
        >
          {labels[option]}
        </button>
      ))}
    </div>
  </section>
)

function AboutMethodFlow({ mode, onModeChange }: AboutMethodFlowProps): JSX.Element {
  const [searchMode, setSearchMode] = useState<SearchMode>(mode)
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('semantic')
  const [agreementMode, setAgreementMode] = useState<AgreementMode>('nli')
  const [activeNodeId, setActiveNodeId] = useState<string>('retrieval')
  const [layoutNodes, setLayoutNodes] = useState<LayoutNodeRecord[]>([])
  const [layoutEdges, setLayoutEdges] = useState<LayoutEdgeRecord[]>([])
  const [layoutSections, setLayoutSections] = useState<SectionRecord[]>([])

  useEffect(() => {
    setSearchMode(mode)
  }, [mode])

  useEffect(() => {
    let cancelled = false

    layoutGraph(searchMode, retrievalMode, agreementMode)
      .then(({ nodes, edges, sections }) => {
        if (cancelled) {
          return
        }

        setLayoutNodes(nodes)
        setLayoutEdges(edges)
        setLayoutSections(sections)
      })
      .catch((error) => {
        console.error('Failed to layout method flow', error)
      })

    return () => {
      cancelled = true
    }
  }, [searchMode, retrievalMode, agreementMode])

  useEffect(() => {
    if (layoutNodes.length === 0) {
      return
    }

    if (!layoutNodes.some((node) => node.id === activeNodeId)) {
      setActiveNodeId('retrieval')
    }
  }, [activeNodeId, layoutNodes])

  const { nodes, methodNodes, edges } = decorateFlow(layoutNodes, layoutEdges, layoutSections, activeNodeId)
  const activeNode = methodNodes.find((node) => node.id === activeNodeId) ?? methodNodes[0]

  const handleSearchModeChange = (nextMode: SearchMode): void => {
    setSearchMode(nextMode)
    onModeChange?.(nextMode)
  }

  const handleNodeFocus = (node: Node): void => {
    if (node.type === 'method') {
      setActiveNodeId(node.id)
    }
  }

  return (
    <section className="about-system-shell">
      <div className="about-system-header">
        <div>
          <p className="about-system-eyebrow">Search Method</p>
          <h3>How hear! hear! ranks articles</h3>
        </div>

        <p className="about-system-intro">
          Follow how your topic or essay moves from prepared Guardian data into relevance scoring, agreement
          ranking, and the explanations shown with your results.
        </p>
      </div>

      <div className="about-system-controls">
        {renderControlGroup(
          'Search mode',
          searchModes,
          searchMode,
          handleSearchModeChange,
          searchModeLabels,
          searchModeNotes[searchMode],
        )}

        {renderControlGroup(
          'Stage 1',
          retrievalModes,
          retrievalMode,
          setRetrievalMode,
          retrievalModeLabels,
          retrievalModeNotes[retrievalMode],
        )}

        {renderControlGroup(
          'Stage 2',
          agreementModes,
          agreementMode,
          setAgreementMode,
          agreementModeLabels,
          agreementModeNotes[agreementMode],
        )}
      </div>

      <div className="about-flow-meta">
        <div className="about-flow-legend-card" aria-label="Block type legend">
          <span className="about-flow-legend-title">Block types</span>
          <div className="about-flow-legend">
            {shapeLegend.map((item) => (
              <div key={item.label} className="about-flow-legend-item">
                <span className={`about-flow-legend-shape kind-${item.kind}`} aria-hidden="true" />
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <p className="about-system-selection-summary">
        Showing <strong>{searchModeLabels[searchMode]}</strong> with <strong>{retrievalModeLabels[retrievalMode]}</strong> retrieval and{' '}
        <strong>{agreementModeLabels[agreementMode]}</strong>.
      </p>

      <div className="about-flow-shell">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          edgeTypes={edgeTypes}
          nodeTypes={nodeTypes}
          onNodeClick={(_, node) => handleNodeFocus(node)}
          onNodeMouseEnter={(_, node) => handleNodeFocus(node)}
          defaultViewport={{ x: 0, y: 0, zoom: 1 }}
          minZoom={0.72}
          maxZoom={1.08}
          nodesDraggable={false}
          nodesConnectable={false}
          nodesFocusable={false}
          edgesFocusable={false}
          elementsSelectable={false}
          panOnDrag
          panOnScroll={false}
          preventScrolling={false}
          zoomOnDoubleClick={false}
          zoomOnScroll={false}
          defaultEdgeOptions={{ type: 'routed' }}
          className="about-flow-canvas"
          proOptions={{ hideAttribution: true }}
        >
          <Background
            color="rgba(255, 255, 255, 0.08)"
            gap={22}
            size={1.2}
            variant={BackgroundVariant.Dots}
          />
        </ReactFlow>
      </div>

      {activeNode && (
        <section className={`about-system-detail kind-${activeNode.data.kind}`} aria-live="polite">
          <div className="about-system-detail-header">
            <div>
              <p className="about-system-detail-eyebrow">Focused block</p>
              <h4>{activeNode.data.title}</h4>
            </div>
          </div>

          <p className="about-system-detail-summary">{activeNode.data.description}</p>

          <div className="about-system-detail-grid">
            <div className="about-system-detail-card">
              <span>Type</span>
              <ul>
                <li>{nodeKindLabels[activeNode.data.kind]}</li>
              </ul>
            </div>

            {Array.isArray(activeNode.data.inputs) && activeNode.data.inputs.length > 0 && (
              <div className="about-system-detail-card">
                <span>Inputs</span>
                <ul>
                  {activeNode.data.inputs.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="about-system-detail-card">
              <span>Details</span>
              <ul>
                {activeNode.data.details.map((detail) => (
                  <li key={detail}>{detail}</li>
                ))}
              </ul>
            </div>

            {Array.isArray(activeNode.data.outputs) && activeNode.data.outputs.length > 0 && (
              <div className="about-system-detail-card">
                <span>Outputs</span>
                <ul>
                  {activeNode.data.outputs.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}

            {Array.isArray(activeNode.data.notes) && activeNode.data.notes.length > 0 && (
              <div className="about-system-detail-card">
                <span>Notes</span>
                <ul>
                  {activeNode.data.notes.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}
    </section>
  )
}

export default AboutMethodFlow
