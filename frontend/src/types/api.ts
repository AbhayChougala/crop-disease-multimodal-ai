export interface TopPrediction {
  label: string;
  confidence: number;
}

export interface DiagnosisResponse {
  session_id: string;
  is_mock: boolean;
  prediction: {
    crop: string | null;
    disease: string | null;
    architecture: string | null;
    checkpoint_id: string | null;
    inference_mode: string;
    abstained: boolean;
    abstention_reason: string | null;
    confidence: number;
    top_k: TopPrediction[];
  };
  severity: {
    method: string;
    affected_area_percent: number | null;
    category: string;
    status: string;
    is_experimental: boolean;
    warning: string | null;
  };
  artifacts: {
    original_image: string;
    preprocessed_image: string;
    leaf_mask: string | null;
    lesion_mask: string | null;
    leaf_isolated: string | null;
    lesion_overlay: string | null;
    gradcam: string | null;
  };
  evidence: {
    model_id: string;
    model_mode: "DEMO" | "REAL" | "UNAVAILABLE";
    uncertainty_flags: string[];
  };
  advisory: {
    summary: string;
    symptoms: string[];
    possible_causes: string[];
    prevention: string[];
    management: string[];
    when_to_seek_expert_help: string;
    sources: Array<{ source_id: string; title: string; url?: string }>;
    is_mock: boolean;
  };
  timings_ms: Record<string, number>;
  runtime: Record<string, string | boolean>;
  warnings: string[];
}

export interface LivePrediction {
  is_mock: boolean;
  crop: string;
  disease: string;
  confidence: number;
  preprocessing_ms: number;
  inference_ms: number;
  total_ms: number;
  warning: string;
  error?: string;
}

export interface ChatCitation {
  source_id: string;
  chunk_id: string;
  document_id: string;
  title: string;
  organization: string | null;
  page: number | null;
  section_name: string | null;
  url: string | null;
  excerpt: string;
  score: number;
}

export interface ChatMessage {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  citations: ChatCitation[];
  metadata: Record<string, string | number | boolean>;
  created_at: string;
}

export interface ChatSession {
  session_id: string;
  mode: "general" | "diagnosis";
  diagnosis_id: string | null;
  provider: string;
  model: string;
  preferred_language: "en" | "kn" | "hi";
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

export interface ChatResponse {
  session_id: string;
  message_id: string;
  answer: string;
  sources: ChatCitation[];
  warnings: string[];
  follow_up_suggestions: string[];
  is_mock: boolean;
  insufficient_evidence: boolean;
  status: string;
  preferred_language: string;
  created_at: string;
  metadata: {
    provider: string;
    model: string;
    retriever: string;
    retrieved_chunks: number;
    context_characters: number;
    retrieval_ms: number;
    llm_ms: number;
    total_ms: number;
  };
}
