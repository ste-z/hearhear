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
type SectionTone = 'primary' | 'stage'

type AboutMethodFlowProps = {
  mode: SearchMode
  onModeChange?: (mode: SearchMode) => void
}

type ModelReference = {
  href?: string
  label: string
}

type DomainNode = {
  description: string
  details: string[]
  h: number
  id: string
  inputs?: string[]
  kind: NodeKind
  outputs?: string[]
  references?: ModelReference[]
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
  outputs?: string[]
  references?: ModelReference[]
  related: boolean
  title: string
}

type SectionNodeData = {
  tone: SectionTone
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
  tone?: SectionTone
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

const retrievalModeLabels: Record<RetrievalMode, string> = {
  lexical: 'TF-IDF',
  semantic: 'SVD',
  enhanced: 'MiniLM',
}

const agreementModeLabels: Record<AgreementMode, string> = {
  nli: 'NLI Agreement',
  llm: 'LLM Agreement',
}

const nodeKindLabels: Record<NodeKind, string> = {
  artifact: 'Prepared data',
  process: 'Processing step',
  input: 'User input',
  result: 'Score / result',
  output: 'Shown result',
}

const sectionLabels = [
  'Source + Precompute',
  'Live Ranking Path',
  'Interpretation Layer',
] as const

const shapeLegend = [
  { kind: 'artifact', label: 'Prepared data' },
  { kind: 'process', label: 'Processing step' },
  { kind: 'input', label: 'Input' },
  { kind: 'result', label: 'Score / result' },
  { kind: 'output', label: 'Shown result' },
] as const satisfies ReadonlyArray<{ kind: NodeKind; label: string }>

const modelReferences = {
  gpt5Nano: {
    href: 'https://developers.openai.com/api/docs/models/gpt-5-nano',
    label: 'GPT-5 nano',
  },
  gptOss20b: {
    label: 'gpt-oss-20b',
  },
  minilm: {
    href: 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
    label: 'all-MiniLM-L6-v2',
  },
  nliDeberta: {
    href: 'https://huggingface.co/cross-encoder/nli-deberta-v3-small',
    label: 'NLI DeBERTa v3 small',
  },
} as const satisfies Record<string, ModelReference>

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

/* Vintage palette: edges read as ink lines on paper. Inactive paths are
   dim grey but solid; active paths are oxblood with full strength so the
   chosen route is unambiguous against the newsprint backdrop. */
const edgeToneStyles: Record<EdgeTone, { color: string; opacity: number }> = {
  neutral: { color: 'rgba(26, 26, 26, 0.55)', opacity: 1 },
  active: { color: '#7a1d1d', opacity: 1 },
  support: { color: 'rgba(26, 26, 26, 0.45)', opacity: 0.9 },
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
    <div className={`about-flow-section-node tone-${data.tone}`}>
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
    description: 'The app starts with Guardian opinion articles and keeps their text, links, dates, and article metadata together.',
    details: [
      'Article text is used for topic matching and summary generation.',
      'Publication dates stay attached so freshness can be used later in the final rank.',
    ],
    h: 134,
    id: 'corpus',
    kind: 'artifact',
    outputs: ['Article text', 'Metadata', 'Publication dates'],
    title: 'Guardian opinion archive',
    w: 320,
  },
  {
    description: "Each article's publication date becomes a freshness score that can lift newer pieces in the final ranking.",
    details: [
      'The score comes from the article date, not from the user query.',
      'It stays separate from topic relevance and stance agreement so the ranking weights can treat freshness explicitly.',
    ],
    h: 122,
    id: 'recency',
    inputs: ['Article publication date'],
    kind: 'result',
    outputs: ['Recency score'],
    title: 'Recency score',
    w: 300,
  },
  {
    description: 'Before live searching, GPT-5 nano condenses long articles into short summaries of their main claims.',
    details: [
      'Article summaries can be generated or recomputed with GPT-5 nano.',
      'The summaries make it easier to compare a user stance or thesis with each article.',
      'The same summary bank helps the agreement scorer and the AI overview after results are ranked.',
    ],
    h: 138,
    id: 'summary-process',
    inputs: ['Guardian article text'],
    kind: 'process',
    outputs: ['Claim-style summary generation'],
    references: [modelReferences.gpt5Nano],
    title: 'LLM summary generation',
    w: 330,
  },
  {
    description: 'The saved article summaries give each article a shorter claim-focused version for later steps.',
    details: [
      'NLI agreement uses these summaries when it compares article claims with the user position.',
      'The overview can also use them to describe patterns across the retrieved articles.',
    ],
    h: 134,
    id: 'summaries',
    inputs: ['LLM-generated article summaries'],
    kind: 'artifact',
    outputs: ['Claim-style summaries', 'Short article representations'],
    references: [modelReferences.gpt5Nano],
    title: 'Article summaries',
    w: 340,
  },
  {
    description: 'Stage 1 gives articles or chunks a topic relevance score based on the selected retrieval method.',
    details: [
      'A higher score means the article is closer to the topic query.',
      'The score helps choose which articles continue and later contributes to the overall score.',
    ],
    h: 122,
    id: 'topic-scores',
    inputs: ['Cosine retrieval outputs'],
    kind: 'result',
    outputs: ['Topic relevance scores'],
    title: 'Topic relevance scores',
    w: 300,
  },
  {
    description: 'The app narrows the topic matches before agreement scoring using the Fixed Number or Smart Filter setting.',
    details: [
      'Fixed Number always uses a set number of top matches.',
      'Smart Filter only sends articles or chunks that are at least the selected relevance level.',
    ],
    h: 136,
    id: 'candidate-selection',
    inputs: ['Topic relevance scores'],
    kind: 'process',
    outputs: ['Candidate articles'],
    title: 'Candidate filtering',
    w: 320,
  },
  {
    description: 'The articles that pass candidate filtering move forward as the candidate set for deeper scoring.',
    details: [
      'This keeps the more expensive agreement step focused on likely matches.',
      'Article-level retrieval and chunk retrieval both end here as article candidates.',
    ],
    h: 124,
    id: 'candidate-set',
    inputs: ['Filtered article ids'],
    kind: 'artifact',
    outputs: ['Candidate articles'],
    title: 'Candidate set',
    w: 320,
  },
  {
    description: 'The final ranking weights combine topic relevance, stance agreement, and recency into one score.',
    details: [
      'Topic relevance answers how closely the article matches the issue.',
      'Stance agreement answers how the article relates to the user position or thesis.',
      'Recency can lift newer articles when its ranking weight is enabled.',
    ],
    h: 144,
    id: 'ranking',
    inputs: ['Topic relevance scores', 'Stance agreement scores', 'Recency score'],
    kind: 'process',
    outputs: ['Final weighted score'],
    title: 'Weighted score merger',
    w: 320,
  },
  {
    description: 'Each candidate article receives one overall score after the weighted merge.',
    details: [
      'This score is what the result list uses for ordering.',
      'Keeping it visible makes clear where the final rank comes from.',
    ],
    h: 122,
    id: 'overall-score',
    inputs: ['Final weighted score'],
    kind: 'result',
    outputs: ['Overall scores'],
    title: 'Overall score',
    w: 300,
  },
  {
    description: 'Articles are sorted by overall score and shown as the final results list.',
    details: [
      'The ranked list keeps score details, article summaries, dates, and links available for inspection.',
      'The same results feed the AI overview, per-result explanation, and results chat.',
    ],
    h: 134,
    id: 'ranked-results',
    inputs: ['Overall scores'],
    kind: 'output',
    outputs: ['Ordered Guardian articles'],
    title: 'Ranked articles',
    w: 320,
  },
  {
    description: 'gpt-oss-20b reads the top ranked articles and summarizes the main pattern of support, challenge, and nuance.',
    details: [
      'The overview describes the result set as a whole instead of listing one article at a time.',
      'It uses only the retrieved results and cites the source results it relies on.',
    ],
    h: 140,
    id: 'overview',
    inputs: ['Ranked articles', 'Article summaries'],
    kind: 'process',
    outputs: ['Narrative overview', 'Grouped viewpoints'],
    references: [modelReferences.gptOss20b],
    title: 'LLM results overview',
    w: 320,
  },
  {
    description: 'gpt-oss-20b can explain why an individual article ranked where it did.',
    details: [
      'The explanation uses the query, score breakdown, retrieved evidence, and article metadata.',
      'It helps users interpret a result after ranking and does not change the article order.',
    ],
    h: 140,
    id: 'ranking-explanation',
    inputs: ['Ranked articles', 'Score breakdown', 'Retrieved evidence'],
    kind: 'process',
    outputs: ['Plain-language rank explanation'],
    references: [modelReferences.gptOss20b],
    title: 'LLM ranking explanation',
    w: 320,
  },
  {
    description: 'LLM results chat lets the user ask follow-up questions grounded in the ranked Guardian articles.',
    details: [
      'The chat can answer from the full result set or from selected attached articles.',
      'It uses the retrieved snippets and article text available to the results page.',
    ],
    h: 140,
    id: 'chat',
    inputs: ['Ranked articles', 'User follow-up questions'],
    kind: 'process',
    outputs: ['Source-linked answers'],
    references: [modelReferences.gptOss20b],
    title: 'LLM results chat',
    w: 320,
  },
]

const getRetrievalArtifactNode = (retrievalMode: RetrievalMode): DomainNode => {
  if (retrievalMode === 'lexical') {
    return {
      description: 'A TF-IDF index records which words are distinctive across Guardian articles for exact-term matching.',
      details: [
        'This works best when the topic uses concrete names, phrases, policies, or events likely to appear in article text.',
        'The live search compares the query against this index to produce topic relevance scores.',
      ],
      h: 142,
      id: 'artifact',
      inputs: ['Guardian article text'],
      kind: 'artifact',
      outputs: ['TF-IDF vectors', 'Lexical index'],
      title: 'Lexical retrieval index',
      w: 340,
    }
  }

  if (retrievalMode === 'semantic') {
    return {
      description: 'A truncated-SVD topic map helps the app match articles by broader themes, not only exact words.',
      details: [
        'The map is built from the article text before the user searches.',
        'It lets related wording land near the same topic area when Stage 1 scores relevance.',
      ],
      h: 142,
      id: 'artifact',
      inputs: ['TF-IDF term-document matrix'],
      kind: 'artifact',
      outputs: ['Latent semantic vectors', 'SVD index'],
      title: 'Semantic latent space',
      w: 340,
    }
  }

  return {
    description: 'MiniLM embeddings store article or chunk meaning in a dense semantic index.',
    details: [
      'This is useful when the user and an article talk about the same issue with different wording.',
      'When chunk search is on, the embedding path can search smaller article passages before grouping them back into articles.',
    ],
    h: 142,
    id: 'artifact',
    inputs: ['Guardian article text or semantic chunks'],
    kind: 'artifact',
    outputs: ['Dense embeddings', 'Embedding index'],
    references: [modelReferences.minilm],
    title: 'Embedding index',
    w: 340,
  }
}

const getSearchNodes = (searchMode: SearchMode, agreementMode: AgreementMode): DomainNode[] => {
  if (searchMode === 'stance') {
    return [
      {
        description: 'The user enters a topic to search for and a position to compare articles against.',
        details: [
          'The topic field drives Stage 1 topic retrieval.',
          'The stance field is used later to decide whether candidate articles support, challenge, or complicate the position.',
        ],
        h: 132,
        id: 'input',
        kind: 'input',
        outputs: ['Topic + stance query'],
        title: 'Topic + stance prompt',
        w: 320,
      },
      {
        description: 'The app can flag likely typos in the topic using the retrieval vocabulary, then lets the user choose a correction or search anyway.',
        details: [
          'The correction is spelling-focused; it does not try to change the user stance.',
          'It is useful when a misspelled topic would otherwise miss relevant articles.',
        ],
        h: 126,
        id: 'refine',
        inputs: ['Topic + stance query'],
        kind: 'process',
        outputs: ['Corrected topic option'],
        title: 'Query typo correction',
        w: 320,
      },
      {
        description: 'gpt-oss-20b can propose clearer topic and stance alternatives while preserving the user position.',
        details: [
          'The rewritten topic is tuned to the selected Stage 1 retrieval method.',
          'The rewritten stance should make agreement scoring clearer without changing the underlying belief.',
        ],
        h: 136,
        id: 'query-rewrite',
        inputs: ['Corrected topic option', 'Original stance'],
        kind: 'process',
        outputs: ['Rewritten topic + stance options'],
        references: [modelReferences.gptOss20b],
        title: 'Query rewrite with LLM',
        w: 320,
      },
    ]
  }

  if (agreementMode === 'llm') {
    return [
      {
        description: 'The user provides an essay draft that can drive both article retrieval and LLM agreement scoring.',
        details: [
          'The full draft is used as the topic query for Stage 1.',
          'With LLM Agreement, the full essay can also provide context for the agreement judgment.',
        ],
        h: 132,
        id: 'input',
        kind: 'input',
        outputs: ['Essay draft'],
        title: 'Essay intake',
        w: 320,
      },
    ]
  }

  return [
    {
      description: 'The user provides an essay draft that drives topic retrieval before a thesis is selected for NLI scoring.',
      details: [
        'The full draft is used as the topic query for Stage 1.',
        'For NLI Agreement, the app also needs a shorter thesis-style statement for the stance comparison.',
      ],
      h: 132,
      id: 'input',
      kind: 'input',
      outputs: ['Essay draft'],
      title: 'Essay intake',
      w: 320,
    },
    {
      description: 'The app proposes a thesis sentence from the essay, and the user can accept or override it before NLI scoring.',
      details: [
        'The selected thesis becomes the statement that candidate articles are compared against.',
        'The full essay still remains useful for Stage 1 topic retrieval.',
      ],
      h: 126,
      id: 'refine',
      inputs: ['Essay draft'],
      kind: 'process',
      outputs: ['Selected thesis sentence'],
      title: 'Thesis selection',
      w: 320,
    },
  ]
}

const getRetrievalProcessNode = (searchMode: SearchMode, retrievalMode: RetrievalMode): DomainNode => {
  const title = (
    retrievalMode === 'lexical'
      ? 'Cosine retrieval over TF-IDF'
      : retrievalMode === 'semantic'
        ? 'Cosine retrieval over SVD space'
        : 'Cosine retrieval over embeddings'
  )

  const description = (
    retrievalMode === 'lexical'
      ? 'The topic query is compared with TF-IDF article vectors so exact terms and distinctive vocabulary drive relevance.'
      : retrievalMode === 'semantic'
        ? 'The topic query is compared inside the SVD topic map so related themes can surface even without exact phrase overlap.'
        : 'The topic query is compared inside the MiniLM embedding index so semantic closeness can drive retrieval.'
  )

  const details = (
    retrievalMode === 'lexical'
      ? [
          'Lexical retrieval is strongest when the query wording matches article wording.',
          'It produces the first topic relevance scores before agreement scoring begins.',
        ]
      : retrievalMode === 'semantic'
        ? [
            'Semantic retrieval is useful when articles use related terms rather than the exact same words.',
            'It produces the first topic relevance scores before agreement scoring begins.',
          ]
        : [
            'Enhanced semantic retrieval uses all-MiniLM-L6-v2 over article-level embeddings or semantic chunks.',
            'It produces the first topic relevance scores before agreement scoring begins.',
          ]
  )

  return {
    description,
    details,
    h: 150,
    id: 'retrieval',
    inputs: [
      searchMode === 'stance' ? 'Topic query' : 'Essay draft',
      retrievalMode === 'lexical'
        ? 'TF-IDF lexical index'
        : retrievalMode === 'semantic'
          ? 'SVD semantic space'
          : 'Embedding index',
    ],
    kind: 'process',
    outputs: ['Topic similarity computation'],
    references: retrievalMode === 'enhanced' ? [modelReferences.minilm] : undefined,
    title,
    w: 380,
  }
}

const getStageTwoPreparationNode = (searchMode: SearchMode, agreementMode: AgreementMode): DomainNode => {
  if (agreementMode === 'nli') {
    return {
      description: searchMode === 'stance'
        ? 'Candidate articles are paired with the user stance and article summaries so NLI can compare claims.'
        : 'Candidate articles are paired with the selected thesis and article summaries so NLI can compare claims.',
      details: [
        'The article summary becomes one side of the comparison.',
        searchMode === 'stance'
          ? 'The user stance becomes the other side.'
          : 'The selected thesis becomes the other side.',
      ],
      h: 140,
      id: 'stage-two-prep',
      inputs: [
        'Candidate set',
        searchMode === 'stance' ? 'User stance' : 'Selected thesis sentence',
        'Article summaries',
      ],
      kind: 'process',
      outputs: ['Premise / hypothesis pairs'],
      title: 'Claim pairing',
      w: 340,
    }
  }

  return {
    description: searchMode === 'stance'
      ? 'Candidate articles are packaged with the user stance so an LLM can judge support, contradiction, and nuance.'
      : 'Candidate articles are packaged with the full essay so an LLM can judge support, contradiction, and nuance in context.',
    details: [
      'The LLM sees the user argument and the article context together.',
      'This is where the app prepares the comparison prompt before asking for an agreement judgment.',
    ],
    h: 140,
    id: 'stage-two-prep',
    inputs: [
      'Candidate set',
      searchMode === 'stance' ? 'User stance' : 'Essay draft',
      'Retrieved article context',
    ],
    kind: 'process',
    outputs: ['LLM comparison prompts'],
    title: 'Prompt assembly',
    w: 340,
  }
}

const getAgreementNode = (searchMode: SearchMode, agreementMode: AgreementMode): DomainNode => {
  if (agreementMode === 'nli') {
    return {
      description: searchMode === 'stance'
        ? 'The NLI DeBERTa model scores whether each article supports, challenges, or stays neutral toward the user stance.'
        : 'The NLI DeBERTa model scores whether each article supports, challenges, or stays neutral toward the selected thesis.',
      details: [
        'It compares short claim pairs instead of reading the whole essay or article.',
        'The output becomes a stance agreement score for each candidate article.',
      ],
      h: 140,
      id: 'agreement',
      inputs: ['Premise / hypothesis pairs'],
      kind: 'process',
      outputs: ['Per-candidate stance judgments'],
      references: [modelReferences.nliDeberta],
      title: 'NLI agreement scorer',
      w: 300,
    }
  }

  return {
    description: searchMode === 'stance'
      ? 'gpt-oss-20b scores how each candidate article relates to the user position.'
      : 'gpt-oss-20b scores how each candidate article relates to the full essay argument and context.',
    details: [
      'The LLM can use richer article context than the short NLI claim pairs.',
      'The output becomes a stance agreement score for each candidate article.',
    ],
    h: 140,
    id: 'agreement',
    inputs: ['LLM comparison prompts'],
    kind: 'process',
    outputs: ['Per-candidate stance judgments'],
    references: [modelReferences.gptOss20b],
    title: 'LLM agreement scorer',
    w: 300,
  }
}

const getAgreementScoresNode = (agreementMode: AgreementMode): DomainNode => ({
  description: agreementMode === 'nli'
    ? 'The NLI judgments become stance agreement scores for the candidate articles.'
    : 'The LLM judgments become stance agreement scores for the candidate articles.',
  details: [
    'These scores say how strongly each candidate aligns with the user stance or thesis.',
    'They join topic relevance and recency in the final weighted score.',
  ],
  h: 112,
  id: 'agreement-scores',
  inputs: ['Per-candidate stance judgments'],
  kind: 'result',
  outputs: ['Stance agreement scores'],
  references: agreementMode === 'nli' ? [modelReferences.nliDeberta] : [modelReferences.gptOss20b],
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
      sourceHandle: 'right-out',
      target: 'artifact',
      targetHandle: 'left-in',
      tone: 'active',
    },
    {
      id: 'corpus-to-summary-process',
      source: 'corpus',
      sourceHandle: 'right-out',
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
      sourceHandle: 'right-out',
      target: 'agreement',
      targetHandle: 'left-in',
      tone: 'active',
    },
    {
      id: 'agreement-to-agreement-scores',
      source: 'agreement',
      sourceHandle: 'right-out',
      target: 'agreement-scores',
      targetHandle: 'left-in',
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
      sourceHandle: 'top-out',
      target: 'ranking',
      targetHandle: 'bottom-in',
      tone: 'support',
    },
    {
      id: 'ranking-to-overall-score',
      source: 'ranking',
      sourceHandle: 'right-out',
      target: 'overall-score',
      targetHandle: 'left-in',
      tone: 'active',
    },
    {
      id: 'overall-score-to-ranked-results',
      source: 'overall-score',
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
      targetHandle: 'left-in',
      tone: 'support',
    },
    {
      id: 'ranked-results-to-ranking-explanation',
      source: 'ranked-results',
      sourceHandle: 'right-out',
      target: 'ranking-explanation',
      targetHandle: 'left-in',
      tone: 'support',
    },
    {
      id: 'ranked-results-to-chat',
      source: 'ranked-results',
      sourceHandle: 'bottom-out',
      target: 'chat',
      targetHandle: 'left-in',
      tone: 'support',
    },
  ]

  if (searchMode === 'stance') {
    edges.push(
      {
        id: 'input-to-refine',
        source: 'input',
        sourceHandle: 'top-out',
        target: 'refine',
        targetHandle: 'bottom-in',
        tone: 'support',
      },
      {
        id: 'refine-to-query-rewrite',
        source: 'refine',
        sourceHandle: 'top-out',
        target: 'query-rewrite',
        targetHandle: 'bottom-in',
        tone: 'support',
      },
      {
        id: 'query-rewrite-to-retrieval',
        source: 'query-rewrite',
        sourceHandle: 'top-out',
        target: 'retrieval',
        targetHandle: 'bottom-in',
        tone: 'neutral',
      },
    )
  } else if (agreementMode === 'nli') {
    edges.push(
      {
        id: 'input-to-retrieval',
        source: 'input',
        sourceHandle: 'top-out',
        target: 'retrieval',
        targetHandle: 'bottom-in',
        tone: 'neutral',
      },
      {
        id: 'input-to-refine',
        source: 'input',
        sourceHandle: 'top-out',
        target: 'refine',
        targetHandle: 'bottom-in',
        tone: 'support',
      },
    )
  } else {
    edges.push({
      id: 'input-to-retrieval',
      source: 'input',
      sourceHandle: 'top-out',
      target: 'retrieval',
      targetHandle: 'bottom-in',
      tone: 'neutral',
    })
  }

  return { edges, nodes }
}

const sectionLayout: SectionRecord[] = [
  {
    height: 940,
    id: 'section-source',
    position: { x: 40, y: 40 },
    tone: 'primary',
    title: sectionLabels[0],
    width: 700,
  },
  {
    height: 940,
    id: 'section-live',
    position: { x: 760, y: 40 },
    tone: 'primary',
    title: sectionLabels[1],
    width: 2760,
  },
  {
    height: 240,
    id: 'section-stage-one',
    position: { x: 840, y: 130 },
    tone: 'stage',
    title: 'Stage 1: Topic relevance',
    width: 820,
  },
  {
    height: 250,
    id: 'section-stage-two',
    position: { x: 1840, y: 620 },
    tone: 'stage',
    title: 'Stage 2: Stance agreement',
    width: 1000,
  },
  {
    height: 940,
    id: 'section-interpretation',
    position: { x: 3570, y: 40 },
    tone: 'primary',
    title: sectionLabels[2],
    width: 800,
  },
]

const methodNodeLayout: Record<string, LayoutBox> = {
  agreement: { height: 118, width: 280, x: 2220, y: 686 },
  'agreement-scores': { height: 112, width: 280, x: 2540, y: 689 },
  artifact: { height: 126, width: 280, x: 430, y: 185 },
  'candidate-selection': { height: 124, width: 300, x: 1280, y: 410 },
  'candidate-set': { height: 118, width: 300, x: 1280, y: 686 },
  chat: { height: 124, width: 320, x: 3970, y: 850 },
  corpus: { height: 110, width: 250, x: 85, y: 300 },
  input: { height: 118, width: 300, x: 880, y: 730 },
  overview: { height: 124, width: 320, x: 3970, y: 490 },
  'overall-score': { height: 118, width: 300, x: 3225, y: 686 },
  'query-rewrite': { height: 124, width: 300, x: 880, y: 410 },
  'ranked-results': { height: 118, width: 300, x: 3610, y: 686 },
  ranking: { height: 126, width: 300, x: 2880, y: 682 },
  'ranking-explanation': { height: 124, width: 320, x: 3970, y: 683 },
  recency: { height: 110, width: 300, x: 2880, y: 860 },
  refine: { height: 118, width: 300, x: 880, y: 570 },
  retrieval: { height: 126, width: 300, x: 880, y: 185 },
  'stage-two-prep': { height: 118, width: 280, x: 1900, y: 686 },
  summaries: { height: 118, width: 280, x: 430, y: 585 },
  'summary-process': { height: 124, width: 280, x: 430, y: 385 },
  'topic-scores': { height: 112, width: 300, x: 1280, y: 192 },
}

const manualEdgeWaypoints: Record<string, (start: RoutePoint, end: RoutePoint) => RoutePoint[]> = {
  'corpus-to-artifact': (start, end) => [
    { x: start.x + 45, y: start.y },
    { x: start.x + 45, y: end.y },
  ],
  'corpus-to-summary-process': (start, end) => [
    { x: start.x + 45, y: start.y },
    { x: start.x + 45, y: end.y },
  ],
  'topic-scores-to-ranking': (start, end) => [
    { x: end.x, y: start.y },
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

const defaultEdgeWaypoints = (edge: DomainEdge, start: RoutePoint, end: RoutePoint): RoutePoint[] => {
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

const routeEdge = (edge: DomainEdge, nodesById: Map<string, LayoutNodeRecord>): RoutePoint[] => {
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

const layoutGraph = (
  searchMode: SearchMode,
  retrievalMode: RetrievalMode,
  agreementMode: AgreementMode,
): { edges: LayoutEdgeRecord[]; nodes: LayoutNodeRecord[]; sections: SectionRecord[] } => {
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
        outputs: node.outputs,
        references: node.references,
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
  const edges = domain.edges.map((edge) => ({
    id: edge.id,
    points: routeEdge(edge, nodesById),
    source: edge.source,
    sourceHandle: edge.sourceHandle,
    target: edge.target,
    targetHandle: edge.targetHandle,
    tone: edge.tone,
  }))

  return { edges, nodes, sections: sectionLayout }
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
      tone: section.tone ?? 'primary',
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
    zIndex: section.tone === 'stage' ? 1 : 0,
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

const renderControlGroup = <T extends string>(
  label: string,
  options: readonly T[],
  activeOption: T,
  onChange: (option: T) => void,
  labels: Record<T, string>,
): JSX.Element => (
  <section className="about-system-control-group">
    <div className="about-system-control-copy">
      <span>{label}</span>
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
    const layout = layoutGraph(searchMode, retrievalMode, agreementMode)
    setLayoutNodes(layout.nodes)
    setLayoutEdges(layout.edges)
    setLayoutSections(layout.sections)
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
        {renderControlGroup('Search mode', searchModes, searchMode, handleSearchModeChange, searchModeLabels)}
        {renderControlGroup('Stage 1', retrievalModes, retrievalMode, setRetrievalMode, retrievalModeLabels)}
        {renderControlGroup('Stage 2', agreementModes, agreementMode, setAgreementMode, agreementModeLabels)}
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
        <div className="about-flow-drag-hint" aria-hidden="true">
          <span className="about-flow-drag-icon">DRAG</span>
          <span>Drag to move the chart</span>
        </div>
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
              <p className="about-system-detail-eyebrow">Selected step</p>
              <h4>{activeNode.data.title}</h4>
            </div>
            <span className={`about-system-type-tag kind-${activeNode.data.kind}`}>
              {nodeKindLabels[activeNode.data.kind]}
            </span>
          </div>

          <p className="about-system-detail-summary">{activeNode.data.description}</p>

          <div className="about-system-detail-grid">
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
              <span>How it works</span>
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

            {Array.isArray(activeNode.data.references) && activeNode.data.references.length > 0 && (
              <div className="about-system-detail-card about-system-reference-card">
                <span>Models and references</span>
                <div className="about-system-detail-tags">
                  {activeNode.data.references.map((reference) => (
                    reference.href ? (
                      <a
                        key={`${reference.label}-${reference.href}`}
                        className="about-system-detail-tag"
                        href={reference.href}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {reference.label}
                      </a>
                    ) : (
                      <span key={reference.label} className="about-system-detail-tag">
                        {reference.label}
                      </span>
                    )
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      )}
    </section>
  )
}

export default AboutMethodFlow
