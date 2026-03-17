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
}
