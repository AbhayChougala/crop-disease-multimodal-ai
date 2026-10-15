export interface TopPrediction {
  label: string;
  confidence: number;
}

export interface DiagnosisResponse {
  session_id: string;
  is_mock: boolean;
  prediction: {
    crop: string;
    disease: string;
    confidence: number;
    top_k: TopPrediction[];
  };
  severity: {
    method: string;
    affected_area_percent: number | null;
    category: string;
    is_experimental: boolean;
  };
  artifacts: {
    original_image: string;
    preprocessed_image: string;
    leaf_mask: string | null;
    lesion_mask: string | null;
    gradcam: string | null;
  };
  evidence: {
    segmentation_valid: boolean;
    model_id: string;
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
