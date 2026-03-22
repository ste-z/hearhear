export type Article = {
  id: string | number
  title: string
  url: string
  summary: string
  date: string | null
  author_display?: string | null
  author_raw?: string | null
  score?: number | null
  keywords?: string[] | null
  central_claim_summary?: string | null
  combined_score?: number | null
  stance_score_normalized?: number | null
  topic_score_normalized?: number | null
  stance_label?: string | null
  stance_entailment_prob?: number | null
  stance_neutral_prob?: number | null
  stance_contradiction_prob?: number | null
  thesis_sentence?: string | null
  support_sentences?: string[] | null
}

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
