export interface Article {
  id: string;
  title: string;
  summary: string;
  date: string | null;
  url: string;
  authors: string[];
  author_display: string;
  author_raw: string;
  year: number;
  n_contributors: number;
  keywords: string[];
  score?: number;
  central_claim_summary?: string | null;
  has_clear_central_thesis?: boolean | null;
  thesis_sentence_id?: string | null;
  thesis_sentence?: string | null;
  support_sentence_ids?: string[];
  support_sentences?: string[];
  secondary_claim_ids?: string[];
  secondary_claim_sentences?: string[];
  claim_available?: boolean;
  topic_statement?: string;
  topic_score?: number | null;
  topic_score_normalized?: number | null;
  stance_entailment_prob?: number | null;
  stance_neutral_prob?: number | null;
  stance_contradiction_prob?: number | null;
  stance_score?: number | null;
  stance_score_normalized?: number | null;
  stance_label?: string | null;
  combined_score?: number | null;
  topic_weight?: number | null;
  stance_weight?: number | null;
  rerank_position?: number | null;
  selected_thesis_sentence?: string | null;
  selected_thesis_id?: string | null;
  essay_query_text?: string | null;
}

export interface EssayClaimCandidate {
  sentence_id: string;
  sentence: string;
  entailment_prob: number;
  neutral_prob: number;
  contradiction_prob: number;
  claim_score: number;
  claim_score_normalized: number;
  claim_label: string;
}

export interface EssayClaimCandidateResponse {
  essay_text: string;
  sentence_count: number;
  candidates: EssayClaimCandidate[];
}
