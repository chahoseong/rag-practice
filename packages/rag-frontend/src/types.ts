export type DocumentChunk = {
  chunk_text: string;
  chunk_index: number;
  title: string;
  url: string;
  source_type: string;
  score?: number; // Optional score field
};

export type GenerateResponse = {
  response: string;
  reference_documents: DocumentChunk[] | null;
  prompt: string;
  question: string;
  elapsed_ms: number;
};

export type GenerateRequest = {
  query: string;
  use_rag?: boolean;
  max_tokens?: number;
};
