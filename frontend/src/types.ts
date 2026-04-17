export type SvdLatentDimension = {
  dimension_index: number
  dimension_label: number
  value: number
  magnitude: number
  pole: 'positive' | 'negative'
  label_terms: string[]
  label_text: string
}

export type Article = {
  id: string | number
  title: string
  url: string
  summary: string
  date: string | null
  year?: number | null
  author_display?: string | null
  author_raw?: string | null
  score?: number | null
  keywords?: string[] | null
  central_claim_summary?: string | null
  combined_score?: number | null
  topic_score?: number | null
  stance_score_normalized?: number | null
  topic_score_normalized?: number | null
  topic_score_display?: number | null
  topic_score_is_normalized?: boolean | null
  recency_score_normalized?: number | null
  recency_weight?: number | null
  stance_method?: string | null
  stance_label?: string | null
  stance_entailment_prob?: number | null
  stance_neutral_prob?: number | null
  stance_contradiction_prob?: number | null
  llm_agreement_score?: number | null
  llm_irrelevant?: boolean | null
  llm_chunking_enabled?: boolean | null
  llm_chunking_mode?: string | null
  llm_chunk_count?: number | null
  llm_related_chunk_count?: number | null
  llm_relevant_paragraphs?: LlmRelevantParagraph[] | null
  thesis_sentence?: string | null
  support_sentences?: string[] | null
  svd_query_chart_dimensions?: SvdLatentDimension[] | null
  svd_chart_dimensions?: SvdLatentDimension[] | null
  svd_positive_dimensions?: SvdLatentDimension[] | null
  svd_negative_dimensions?: SvdLatentDimension[] | null
  svd_dimensions?: SvdLatentDimension[] | null
}

export type LlmRelevantParagraph = {
  paragraph_id?: string | null
  paragraph_index?: number | null
  text: string
  agreement_score?: number | null
  sentence_start_index?: number | null
  sentence_end_index?: number | null
}

export type ArticleSearchResponse = {
  results?: Article[] | null
  query_svd_corpus_chart_dimensions?: SvdLatentDimension[] | null
  query_svd_dimensions?: SvdLatentDimension[] | null
  empty_results_message?: string | null
}

export type RetrievalModel = 'tfidf' | 'svd'

export type EssayClaimCandidate = {
  sentence_id: string
  sentence: string
  score?: number | null
}

export type EssayClaimCandidateResponse = {
  essay_text?: string
  candidates?: EssayClaimCandidate[]
}

export type EssayTextExtractionResponse = {
  essay_text?: string
}
