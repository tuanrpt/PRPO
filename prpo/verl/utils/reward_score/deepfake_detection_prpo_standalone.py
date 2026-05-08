# Original Copyright (c) 2023 PRIME-RL (TTRL)
# Modifications Copyright (c) 2025 Tuan Nguyen
#
# This file is modified from TTRL: https://github.com/PRIME-RL/TTRL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Standalone PRPO deepfake detection reward function.
All dependencies included inline to avoid import issues with Ray.
"""

import re
import traceback
import os
import random
from typing import List, Dict, Tuple, Any

NEGATORS = [
    "no", "not", "without", "lack of", "absence of", "lacks", "lacking",
    "neither", "doesn't", "isn't", "aren't", "can't", "won't", "never", "none", "missing"
]

REAL_TERMS = [
    "real", "authentic", "genuine", "unmanipulated", "photograph", "original",
    "natural", "realistic", "plausible", "typical", "normal", "expected", "appropriate", "consistent", "coherent",
    "anatomically", "well-defined", "aligned", "properly", "correctly",
    "pores", "micro-variations", "subtle variations", "specular highlight", "catchlight",
    "depth of field", "3d appearance", "underlying bone", "muscle structure", "asymmetry",
    "lack of artifacts", "absence of artifacts", "no artifacts", "no manipulation",
]

FAKE_TERMS = [
    "fake", "manipulated", "manipulation", "deepfake", "synthetic", "generated", "artificial", "composite",
    "artifact", "artifacts", "anomaly", "anomalies", "inconsistent", "inconsistencies", "unnatural", "uniform",
    "overly smooth", "plastic-like", "pixelation", "smudging", "distortion", "blending", "seams", "glitches", "warping",
    "incorrect", "missing", "misaligned", "rigid"
]

def make_patterns(terms: List[str]) -> List:
    """Create regex patterns from term lists."""
    pats = []
    for t in terms:
        t_esc = re.escape(t).replace("\\ ", r"\s+")
        pats.append(re.compile(r'\b' + t_esc + r'\b', re.IGNORECASE))
    return pats

REAL_PATS = make_patterns(REAL_TERMS)
FAKE_PATS = make_patterns(FAKE_TERMS)
NEGATOR_PATS = make_patterns(NEGATORS)

def has_negation_before(text: str, start_idx: int, window: int = 80) -> bool:
    """Check if there's negation before a match within a window."""
    left = max(0, start_idx - window)
    span = text[left:start_idx]
    return any(p.search(span) for p in NEGATOR_PATS)

def score_paragraph(paragraph: str) -> Dict[str, float]:
    """Score a paragraph for real/fake indicators with negation handling."""
    txt = paragraph.lower()
    real_score = 0.0
    fake_score = 0.0

    def weight(pattern) -> float:
        return 1.5 if "\\s+" in pattern.pattern else 1.0

    for pat in REAL_PATS:
        for m in pat.finditer(txt):
            w = weight(pat)
            if has_negation_before(txt, m.start()):
                fake_score += w
            else:
                real_score += w

    for pat in FAKE_PATS:
        for m in pat.finditer(txt):
            w = weight(pat)
            if has_negation_before(txt, m.start()):
                real_score += w
            else:
                fake_score += w

    return {"real_score": real_score + 1e-6, "fake_score": fake_score + 1e-6}

def predict_label(scores: Dict[str, float]) -> Tuple[str, float]:
    """Predict label from scores."""
    real_s, fake_s = scores["real_score"], scores["fake_score"]
    total_s = real_s + fake_s
    if real_s >= fake_s:
        confidence = (real_s - fake_s) / total_s if total_s > 0 else 0.0
        return "real", min(1.0, max(0.0, confidence))
    else:
        confidence = (fake_s - real_s) / total_s if total_s > 0 else 0.0
        return "fake", min(1.0, max(0.0, confidence))

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs and clean them."""
    paragraphs = re.split(r'\n\s*\n', text)

    cleaned = []
    for para in paragraphs:
        para = para.strip()

        para = re.sub(r'\s+', ' ', para)

        is_final_answer = bool(re.search(r'###.*[Aa]nswer.*:|[Aa]nswer.*:|[Ff]inal.*[Aa]nswer.*:|[Cc]onclusion.*:', para))

        if para and (len(para) > 20 or is_final_answer):
            cleaned.append(para)

    return cleaned

def extract_final_answer(response_text: str) -> str:
    """Extract the final answer (real/fake) from response text."""
    lines = response_text.strip().split('\n')

    answer_patterns = [
        r'###\s*[Aa]nswer\s*:',
        r'##\s*[Aa]nswer\s*:',
        r'#\s*[Aa]nswer\s*:',
        r'[Aa]nswer\s*:',
        r'[Ff]inal\s*[Aa]nswer\s*:',
        r'[Cc]onclusion\s*:',
    ]

    for line in reversed(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        for pattern in answer_patterns:
            if re.search(pattern, line_stripped):
                answer_lower = line_stripped.lower()
                if 'real' in answer_lower and 'fake' not in answer_lower:
                    return 'real'
                elif 'fake' in answer_lower and 'real' not in answer_lower:
                    return 'fake'
                elif answer_lower.strip().endswith('real'):
                    return 'real'
                elif answer_lower.strip().endswith('fake'):
                    return 'fake'

    return 'unknown'

def extract_answer_deepfake(response: str) -> str:
    """Extract 'real' or 'fake' prediction from model response."""
    if not isinstance(response, str):
        return None

    prediction = None
    if '### Answer:' in response:
        answer_part = response.split('### Answer:')[-1].strip().lower()
        if 'real' in answer_part and 'fake' not in answer_part:
            prediction = 'real'
        elif 'fake' in answer_part and 'real' not in answer_part:
            prediction = 'fake'
        elif 'fake' in answer_part:
            prediction = 'fake'
        elif 'real' in answer_part:
            prediction = 'real'
    else:
        # Fallback: use enhanced extraction
        prediction = extract_final_answer(response)
        if prediction == 'unknown':
            prediction = None

    return prediction

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import yake
    import open_clip

    _clip_models = {}
    _clip_preprocess = None
    _image_feature_cache = {}
    _clip_tokenizer = None
    _available_gpus = []
    _gpu_round_robin = 0

    _debug_counter = 0
    _debug_samples_seen_this_batch = 0
    _debug_max_per_batch = 1

    _master_yake_extractor = None
    _yake_extractors = {}
    _yake_initialized = False

    def get_available_gpus():
        """Get list of available GPU devices respecting CUDA_VISIBLE_DEVICES."""
        if not torch.cuda.is_available():
            return []

        import os
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)

        if cuda_visible_devices:
            visible_gpu_count = len(cuda_visible_devices.split(','))
            return list(range(visible_gpu_count))
        else:
            return list(range(torch.cuda.device_count()))

    def initialize_clip_model():
        """Initialize single ConvNeXt CLIP model on primary GPU with distributed processing."""
        global _clip_models, _clip_preprocess, _clip_tokenizer, _available_gpus, _image_feature_cache

        import os
        ray_worker_id = os.environ.get('RAY_WORKER_ID', None)
        ray_node_rank = os.environ.get('RAY_NODE_RANK', '0')

        _available_gpus = get_available_gpus()
        if not _available_gpus:
            return False

        if _clip_models:
            return True

        primary_gpu = _available_gpus[0]

        try:
            model_name = "convnext_large_d_320"
            pretrained = "laion2b_s29b_b131k_ft_soup"


            primary_device = torch.device(f"cuda:{primary_gpu}")
            model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=primary_device
            )
            _clip_tokenizer = open_clip.get_tokenizer(model_name)

            model = model.eval().half()

            _clip_models[primary_gpu] = {
                'model': model,
                'device': primary_device,
                'dtype': torch.float16,
                'is_primary': True
            }

            _image_feature_cache = {}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return True

        except Exception as e:
            print(f"ConvNeXt initialization failed: {e}")

    def get_next_gpu():
        """Round-robin GPU selection for load balancing."""
        global _gpu_round_robin
        if not _available_gpus:
            return None
        gpu_id = _available_gpus[_gpu_round_robin]
        _gpu_round_robin = (_gpu_round_robin + 1) % len(_available_gpus)
        return gpu_id

    def initialize_yake_extractor():
        """Initialize master YAKE keyword extractor - Ray distributed training aware."""
        global _master_yake_extractor, _yake_initialized

        import os
        ray_worker_id = os.environ.get('RAY_WORKER_ID', None)
        if not _yake_initialized:

            _master_yake_extractor = yake.KeywordExtractor(
                lan="en",
                n=10,
                dedupLim=0.7,
                top=1,
                features=None
            )

            _yake_initialized = True
        return True

    def get_yake_extractor():
        """Get YAKE extractor for current process (creates clone if needed for multiprocessing)."""
        global _master_yake_extractor, _yake_extractors

        import os
        process_id = os.getpid()
        main_process_id = getattr(get_yake_extractor, '_main_pid', None)

        if main_process_id is None:
            get_yake_extractor._main_pid = process_id
            main_process_id = process_id

        if process_id == main_process_id and _master_yake_extractor is not None:
            return _master_yake_extractor

        if process_id not in _yake_extractors:
            if _master_yake_extractor is not None:
                _yake_extractors[process_id] = yake.KeywordExtractor(
                    lan="en",
                    n=10,
                    dedupLim=0.7,
                    top=1,
                    features=None
                )
            else:
                _yake_extractors[process_id] = yake.KeywordExtractor(
                    lan="en",
                    n=10,
                    dedupLim=0.7,
                    top=1,
                    features=None
                )

        return _yake_extractors[process_id]

    def setup_clip_model():
        """Get already initialized ConvNeXt CLIP model for multi-GPU processing."""
        global _clip_models, _clip_preprocess, _clip_tokenizer, _available_gpus
        if not _clip_models:
            success = initialize_clip_model()
            if not success:
                return None, None, None, None

        if _available_gpus and _clip_models:
            gpu_id = _available_gpus[0]
            model_info = _clip_models[gpu_id]
            return model_info['model'], _clip_preprocess, _clip_tokenizer, model_info['device']

        return None, None, None, None

    def extract_keyword_yake_single(text: str, top_k: int = 1) -> str:
        """Extract top keyword from text using YAKE (single-threaded version)."""
        try:
            kw_extractor = get_yake_extractor()
            keywords = kw_extractor.extract_keywords(text)
            if keywords:
                keyword_tuple = keywords[0]
                if isinstance(keyword_tuple, tuple) and len(keyword_tuple) >= 2:
                    keyword_str = keyword_tuple[0]
                    if isinstance(keyword_str, str) and keyword_str.strip():
                        return keyword_str.strip()
                elif isinstance(keyword_tuple, str):
                    return keyword_tuple.strip()

            words = text.split()
            meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
            if meaningful_words:
                return ' '.join(meaningful_words[:2])
            else:
                return ' '.join(words[:3]) if words else "content"

        except Exception as e:
            print(f"YAKE extraction failed: {e}")
            words = text.split()
            meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
            if meaningful_words:
                return ' '.join(meaningful_words[:2])
            else:
                return ' '.join(words[:3]) if words else "content"

    def extract_keywords_yake_parallel(paragraphs: List[str]) -> List[str]:
        """Extract keywords from multiple paragraphs in parallel using threading (avoids pickling issues)."""
        if not paragraphs:
            return []

        if len(paragraphs) <= 3:
            return [extract_keyword_yake_single(para) for para in paragraphs]

        try:
            import threading

            n_threads = min(4, len(paragraphs))
            results = [None] * len(paragraphs)
            threads = []

            def extract_keyword_thread(idx, paragraph):
                try:
                    results[idx] = extract_keyword_yake_single(paragraph)
                except Exception as e:
                    print(f"Thread {idx} YAKE extraction failed: {e}")
                    results[idx] = "content"

            for i, para in enumerate(paragraphs):
                thread = threading.Thread(target=extract_keyword_thread, args=(i, para))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            return results

        except Exception as e:
            print(f"Parallel YAKE processing failed: {e}, falling back to sequential")
            return [extract_keyword_yake_single(para) for para in paragraphs]

    def extract_keyword_yake(text: str, top_k: int = 1) -> str:
        """Extract top keyword from text using YAKE (backward compatibility)."""
        return extract_keyword_yake_single(text, top_k)

    def compute_clip_similarity_batch_with_model(image_path: str, keywords: List[str],
                                               model, preprocess, tokenizer, device) -> List[float]:
        """Compute ConvNeXt CLIP similarity with pre-loaded model (most efficient)."""
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                return [0.5] * len(keywords)

            clean_keywords = []
            for keyword in keywords:
                if not isinstance(keyword, str):
                    keyword = str(keyword) if keyword is not None else "content"
                if not keyword.strip():
                    keyword = "content"
                clean_keywords.append(keyword.strip())

            image = Image.open(image_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device, dtype=torch.float16)

            text_input = tokenizer(clean_keywords).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarities = torch.matmul(image_features, text_features.t()).squeeze(0)

                similarities = (similarities + 1) / 2

            return similarities.cpu().numpy().tolist()

        except Exception as e:
            print(f"ConvNeXt CLIP batch similarity computation failed: {e}")
            return [0.5] * len(keywords)

    def compute_clip_similarity_batch_multi_gpu(image_path: str, keywords: List[str]) -> List[float]:
        """Compute ConvNeXt CLIP similarity using multiple GPUs - Ray distributed training aware."""
        global _clip_models, _clip_preprocess, _clip_tokenizer, _available_gpus

        import os
        ray_worker_id = os.environ.get('RAY_WORKER_ID', None)

        if not _clip_models or not _available_gpus:
            return compute_clip_similarity_batch_single_gpu(image_path, keywords)

        if not keywords:
            return []

        try:
            available_model_ids = list(_clip_models.keys())
            n_gpus = len(available_model_ids)

            if len(keywords) < n_gpus:
                n_gpus = min(len(keywords), n_gpus)
                available_model_ids = available_model_ids[:n_gpus]

            keywords_per_gpu = len(keywords) // n_gpus
            extra_keywords = len(keywords) % n_gpus


            keyword_chunks = []
            start_idx = 0
            for i in range(n_gpus):
                current_chunk_size = keywords_per_gpu + (1 if i < extra_keywords else 0)
                end_idx = start_idx + current_chunk_size

                if start_idx < len(keywords):
                    chunk = keywords[start_idx:end_idx]
                    if chunk:
                        gpu_id = available_model_ids[i]
                        keyword_chunks.append((gpu_id, chunk))
                        start_idx = end_idx

            import threading
            results = {}
            threads = []

            def process_chunk(gpu_id, chunk, chunk_idx):
                try:
                    model_info = _clip_models[gpu_id]
                    model = model_info['model']
                    device = model_info['device']

                    with torch.cuda.device(device):
                        chunk_results = compute_clip_similarity_batch_with_model(
                            image_path, chunk, model, _clip_preprocess, _clip_tokenizer, device
                        )
                        results[chunk_idx] = chunk_results

                except Exception as e:
                    results[chunk_idx] = [0.5] * len(chunk)
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            for chunk_idx, (gpu_id, chunk) in enumerate(keyword_chunks):
                thread = threading.Thread(target=process_chunk, args=(gpu_id, chunk, chunk_idx))
                threads.append(thread)
                thread.start()

            timeout_per_thread = 30
            for thread in threads:
                thread.join(timeout=timeout_per_thread)
                if thread.is_alive():
                    print("⚠️ Thread timeout - some GPU processing may be incomplete")

            combined_results = []
            for chunk_idx in range(len(keyword_chunks)):
                if chunk_idx in results:
                    combined_results.extend(results[chunk_idx])
                else:
                    chunk_size = len(keyword_chunks[chunk_idx][1])
                    combined_results.extend([0.5] * chunk_size)

            if len(combined_results) != len(keywords):
                return [0.5] * len(keywords)

            return combined_results

        except Exception as e:
            if ray_worker_id:
                print(f"Ray Worker {ray_worker_id}: Multi-GPU CLIP processing failed: {e}, falling back to single GPU")
            else:
                print(f"Multi-GPU CLIP processing failed: {e}, falling back to single GPU")
            return compute_clip_similarity_batch_single_gpu(image_path, keywords)

    def compute_clip_similarity_batch_single_gpu(image_path: str, keywords: List[str]) -> List[float]:
        """Compute ConvNeXt CLIP similarity on single GPU (fallback)."""
        try:
            model, preprocess, tokenizer, device = setup_clip_model()
            if model is None:
                return [0.5] * len(keywords)
            return compute_clip_similarity_batch_with_model(image_path, keywords, model, preprocess, tokenizer, device)

        except Exception as e:
            print(f"Single GPU CLIP batch similarity computation failed: {e}")
            return [0.5] * len(keywords)

    def compute_clip_similarity_batch(image_path: str, keywords: List[str]) -> List[float]:
        """Compute ConvNeXt CLIP similarity between image and multiple keywords in batch."""
        if not keywords:
            return []

        if len(_clip_models) > 1 and len(keywords) >= len(_clip_models):
            return compute_clip_similarity_batch_multi_gpu(image_path, keywords)
        else:
            return compute_clip_similarity_batch_single_gpu(image_path, keywords)

    def compute_clip_similarity_hybrid(image_path: str, keywords: List[str]) -> List[float]:
        """
        HYBRID APPROACH: Single GPU CLIP + Distributed Keyword Processing

        1. Single ConvNeXt model on primary GPU (no deep copying)
        2. Cache image features (compute once per image)
        3. Distribute keyword encoding across available GPUs
        4. Batch similarity computation
        """
        global _clip_models, _image_feature_cache, _available_gpus

        if not _clip_models:
            return [0.5] * len(keywords)

        primary_gpu_id = next(iter(_clip_models.keys()))
        clip_data = _clip_models[primary_gpu_id]
        model = clip_data['model']
        device = clip_data['device']

        try:
            image_features = _image_feature_cache.get(image_path)

            if image_features is None:
                if not os.path.exists(image_path):
                    return [0.5] * len(keywords)

                image = Image.open(image_path).convert('RGB')
                image_input = _clip_preprocess(image).unsqueeze(0).to(device, dtype=torch.float16)

                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features = F.normalize(image_features, p=2, dim=1)


                _image_feature_cache[image_path] = image_features.clone()

            n_available_gpus = len(_available_gpus)
            if n_available_gpus > 1:
                keyword_chunks = [keywords[i::n_available_gpus] for i in range(n_available_gpus)]
                keyword_chunks = [chunk for chunk in keyword_chunks if chunk]
            else:
                keyword_chunks = [keywords]

            all_similarities = []

            for chunk_idx, keyword_chunk in enumerate(keyword_chunks):
                if not keyword_chunk:
                    continue


                try:
                    text_tokens = [_clip_tokenizer(kw) for kw in keyword_chunk]

                    if len(text_tokens) == 1 and text_tokens[0].dim() > 1:
                        text_input = text_tokens[0].to(device, dtype=torch.int64)
                    else:
                        text_input = torch.stack(text_tokens).to(device, dtype=torch.int64)

                    if text_input.dim() == 3:
                        text_input = text_input.squeeze(1)


                    with torch.no_grad():
                        text_features = model.encode_text(text_input)
                        text_features = F.normalize(text_features, p=2, dim=1)


                        if image_features.dim() == 1:
                            image_features = image_features.unsqueeze(0)
                        elif image_features.dim() > 2:
                            image_features = image_features.view(image_features.size(0), -1)

                        if text_features.dim() == 1:
                            text_features = text_features.unsqueeze(0)
                        elif text_features.dim() > 2:
                            text_features = text_features.view(text_features.size(0), -1)

                        raw_similarities = torch.mm(image_features, text_features.T).squeeze(0)

                        normalized_similarities = (raw_similarities + 1.0) / 2.0

                        if normalized_similarities.dim() == 0:
                            chunk_similarities = [normalized_similarities.item()]
                        else:
                            chunk_similarities = normalized_similarities.cpu().tolist()

                        all_similarities.extend(chunk_similarities)

                except Exception as chunk_error:
                    chunk_similarities = [0.5] * len(keyword_chunk)
                    all_similarities.extend(chunk_similarities)

            return all_similarities[:len(keywords)]

        except Exception as e:
            return [0.5] * len(keywords)

    def compute_clip_similarity(image_path: str, keyword: str) -> float:
        """Compute ConvNeXt CLIP similarity between image and keyword."""
        try:
            model, preprocess, tokenizer, device = setup_clip_model()

            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                return 0.5

            if not isinstance(keyword, str):
                print(f"Warning: keyword is not a string: {type(keyword)}, value: {keyword}")
                keyword = str(keyword) if keyword is not None else "content"

            if not keyword.strip():
                print(f"Warning: empty keyword, using fallback")
                keyword = "content"

            image = Image.open(image_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device, dtype=torch.float16)

            text_input = tokenizer([keyword.strip()]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = torch.matmul(image_features, text_features.t()).item()

                similarity = (similarity + 1) / 2

            return float(similarity)

        except Exception as e:
            print(f"ConvNeXt CLIP similarity computation failed: {e}")
            return 0.5

    CLIP_AVAILABLE = True

except ImportError as e:
    print(f"Warning: CLIP functionality not available: {e}")
    CLIP_AVAILABLE = False

    def initialize_clip_model():
        return None, None, None

    def initialize_yake_extractor():
        return None

    def setup_clip_model():
        return None, None, None

    def extract_keyword_yake(text):
        return "unknown"

    def compute_clip_similarity(image_path, keyword):
        return 0.5


def _compute_validation_metrics(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Compute supervised validation metrics (accuracy, precision, recall, F1)."""
    pred = prediction.lower().strip() if prediction else ""
    gt = ground_truth.lower().strip() if ground_truth else ""

    if not pred or pred == "unknown":
        return {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "extracted_gt": gt,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0
        }

    if not gt or gt == "unknown":
        return {
            "acc": -1.0,
            "precision": -1.0,
            "recall": -1.0,
            "f1": -1.0,
            "extracted_gt": "unknown",
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0
        }

    pred_binary = 1 if pred == "fake" else 0
    gt_binary = 1 if gt == "fake" else 0

    tp = 1 if pred_binary == 1 and gt_binary == 1 else 0
    fp = 1 if pred_binary == 1 and gt_binary == 0 else 0
    tn = 1 if pred_binary == 0 and gt_binary == 0 else 0
    fn = 1 if pred_binary == 0 and gt_binary == 1 else 0

    accuracy = float(tp + tn)

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "acc": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "extracted_gt": gt,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn)
    }


def init_prpo_models():
    """
    Initialize BOTH CLIP model and YAKE extractor for PRPO training.

    RAY DISTRIBUTED TRAINING AWARE:
    - Each Ray worker initializes its own models for the GPUs it can access
    - Main process initialization is coordinated via training script

    Returns:
        dict: Status of initialization for each component
    """
    import os
    ray_worker_id = os.environ.get('RAY_WORKER_ID', None)

    results = {"clip": False, "yake": False}
    if not CLIP_AVAILABLE:
        results["clip"] = False
    else:
        try:
            success = initialize_clip_model()
            if success:
                results["clip"] = True
            else:
                results["clip"] = False
        except Exception as e:
            print(f"Failed to initialize CLIP model for PRPO: {e}")
            import traceback
            traceback.print_exc()
            results["clip"] = False

    try:
        initialize_yake_extractor()
        results["yake"] = True
    except Exception as e:
        print(f"Failed to initialize YAKE extractor for PRPO: {e}")
        import traceback
        traceback.print_exc()
        results["yake"] = False

    return results

def init_prpo_clip_model():
    """Backward compatibility wrapper."""
    results = init_prpo_models()
    return results["clip"]

def compute_enhanced_paragraph_rewards(response_text: str, image_path: str = None) -> Dict[str, Any]:
    """Compute enhanced paragraph-level rewards combining CLIP similarity and consistency."""
    import os

    paragraphs = split_into_paragraphs(response_text)
    if not paragraphs:
        paragraphs = [response_text.strip()]


    paragraph_rewards = []
    paragraph_metadata = []

    extracted_answer = extract_final_answer(response_text)

    has_valid_image = CLIP_AVAILABLE and image_path and os.path.exists(image_path)

    all_keywords = []
    all_clip_scores = [0.5] * len(paragraphs)

    if has_valid_image:
        try:
            cleaned_paragraphs_for_yake = []
            for paragraph in paragraphs:
                cleaned_para = re.sub(r'\*\*[^*]+\*\*:\s*', '', paragraph)
                cleaned_para = re.sub(r'#+\s*', '', cleaned_para)
                cleaned_para = re.sub(r'\s+', ' ', cleaned_para).strip()
                cleaned_paragraphs_for_yake.append(cleaned_para)

            all_keywords = extract_keywords_yake_parallel(cleaned_paragraphs_for_yake)
            all_clip_scores = compute_clip_similarity_hybrid(image_path, all_keywords)


        except Exception as e:
            all_keywords = []
            for paragraph in paragraphs:
                cleaned_para = re.sub(r'\*\*[^*]+\*\*:\s*', '', paragraph)
                cleaned_para = re.sub(r'#+\s*', '', cleaned_para)
                cleaned_para = re.sub(r'\s+', ' ', cleaned_para).strip()

                keyword = extract_keyword_yake_single(cleaned_para)
                all_keywords.append(keyword)
            all_clip_scores = [0.5] * len(paragraphs)
    else:

        cleaned_paragraphs_for_yake = []
        for paragraph in paragraphs:
            cleaned_para = re.sub(r'\*\*[^*]+\*\*:\s*', '', paragraph)
            cleaned_para = re.sub(r'#+\s*', '', cleaned_para)
            cleaned_para = re.sub(r'\s+', ' ', cleaned_para).strip()
            cleaned_paragraphs_for_yake.append(cleaned_para)

        all_keywords = extract_keywords_yake_parallel(cleaned_paragraphs_for_yake)
        all_clip_scores = [0.5] * len(paragraphs)

    for i, paragraph in enumerate(paragraphs):
        reward_components = {}

        is_final_answer = bool(re.search(r'###.*[Aa]nswer.*:|[Aa]nswer.*:|[Ff]inal.*[Aa]nswer.*:|[Cc]onclusion.*:', paragraph))

        if is_final_answer and extracted_answer in ['real', 'fake']:
            keyword = extracted_answer
        else:
            keyword = all_keywords[i] if i < len(all_keywords) else "unknown"

        if is_final_answer and extracted_answer in ['real', 'fake']:
            if image_path and os.path.exists(image_path):
                clip_score = compute_clip_similarity_batch_multi_gpu(image_path, [keyword])[0]
            else:
                clip_score = 0.5
        else:
            clip_score = all_clip_scores[i] if i < len(all_clip_scores) else 0.5

        clip_reward = clip_score

        reward_components['clip_score'] = clip_score
        reward_components['clip_reward'] = clip_reward
        reward_components['keyword'] = keyword

        para_scores = score_paragraph(paragraph)
        para_pred, confidence = predict_label(para_scores)

        if is_final_answer:
            if i == 0:
                consistency_score = 1.0
            else:
                other_paragraph_predictions = []
                for j, other_paragraph in enumerate(paragraphs[:i]):
                    other_para_scores = score_paragraph(other_paragraph)
                    other_para_pred, _ = predict_label(other_para_scores)
                    other_paragraph_predictions.append(other_para_pred)

                if other_paragraph_predictions:
                    real_votes = other_paragraph_predictions.count('real')
                    fake_votes = other_paragraph_predictions.count('fake')

                    if real_votes > fake_votes:
                        majority_prediction = 'real'
                    elif fake_votes > real_votes:
                        majority_prediction = 'fake'
                    else:
                        majority_prediction = extracted_answer if extracted_answer in ['real', 'fake'] else 'real'

                    consistency_score = 1.0 if majority_prediction == extracted_answer else 0.0

                else:
                    consistency_score = 1.0
        else:
            consistency_score = 1.0

        reward_components['consistency_score'] = consistency_score
        reward_components['paragraph_prediction'] = para_pred
        reward_components['paragraph_confidence'] = confidence
        reward_components['matches_final_answer'] = (para_pred == extracted_answer)
        reward_components['is_final_answer'] = is_final_answer

        alpha_clip = float(os.environ.get('ALPHA_CLIP', '0.5'))
        beta_consistency = float(os.environ.get('BETA_CONSISTENCY', '0.5'))

        if has_valid_image:
            raw_combined_reward = (alpha_clip * clip_reward + beta_consistency * consistency_score)
            reward_components['has_clip'] = True
            reward_components['clip_weight'] = alpha_clip
            reward_components['consistency_weight'] = beta_consistency
        else:
            raw_combined_reward = consistency_score
            reward_components['has_clip'] = False
            reward_components['clip_weight'] = 0.0
            reward_components['consistency_weight'] = 1.0

        combined_reward = raw_combined_reward

        reward_components['raw_combined_reward'] = raw_combined_reward
        reward_components['combined_reward'] = combined_reward
        paragraph_rewards.append(combined_reward)
        paragraph_metadata.append([
            float(i),
            float(combined_reward),
            float(reward_components.get('clip_score', 0.5)),
            float(reward_components.get('clip_reward', 0.0)),
            float(consistency_score),
            float(raw_combined_reward)
        ])

    import os
    MAX_PARAGRAPHS = int(os.environ.get('PRPO_MAX_PARAGRAPHS', 8))
    while len(paragraph_rewards) < MAX_PARAGRAPHS:
        paragraph_rewards.append(0.0)
        paragraph_metadata.append([0.0, 0.0, 0.5, 0.0, 0.5, 0.5])

    if len(paragraph_rewards) > MAX_PARAGRAPHS:
        paragraph_rewards = paragraph_rewards[:MAX_PARAGRAPHS]
        paragraph_metadata = paragraph_metadata[:MAX_PARAGRAPHS]

    return {
        'paragraph_rewards': paragraph_rewards,
        'paragraph_metadata': paragraph_metadata,
        'num_paragraphs': len(paragraphs),
        'extracted_answer': extracted_answer,
        'image_path': image_path
    }

def compute_prpo_score(model_response: str, image_path: str = None, ground_truth: str = None,
                      is_validation: bool = False, fast: bool = True) -> Dict[str, Any]:
    """Enhanced compute score for PRPO with paragraph-level rewards and supervised validation."""
    model_answer = extract_answer_deepfake(model_response)

    if model_answer is None:
        import os
        MAX_PARAGRAPHS = int(os.environ.get('PRPO_MAX_PARAGRAPHS', 8))
        paragraph_rewards = [0.0] * MAX_PARAGRAPHS
        paragraph_metadata = [[0.0, 0.0, 0.5, 0.0, 0.5, 0.5]] * MAX_PARAGRAPHS
        base_result = {
            "format_score": 0.0,
            "acc": -1.0,
            "extracted_gt": "unsupervised",
            "pred": "",
            "paragraph_rewards": paragraph_rewards,
            "paragraph_metadata": paragraph_metadata,
            "num_paragraphs": 1,
            "prpo_enabled": True,
            "image_path": image_path
        }

        if is_validation and ground_truth:
            base_result.update(_compute_validation_metrics("", ground_truth))
        else:
            base_result.update({
                "precision": -1.0,
                "recall": -1.0,
                "f1": -1.0,
                "extracted_gt": "unsupervised",
                "true_positive": 0,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0
            })

        return base_result

    paragraph_data = compute_enhanced_paragraph_rewards(model_response, image_path)

    result = {
        "format_score": 1.0 if model_answer else 0.0,
        "pred": model_answer,
        "paragraph_rewards": paragraph_data['paragraph_rewards'],
        "paragraph_metadata": paragraph_data['paragraph_metadata'],
        "num_paragraphs": paragraph_data['num_paragraphs'],
        "prpo_enabled": True,
        "image_path": image_path
    }

    if is_validation and ground_truth:
        validation_metrics = _compute_validation_metrics(model_answer, ground_truth)
        result.update(validation_metrics)
    else:
        result.update({
            "acc": -1.0,
            "extracted_gt": "unsupervised",
            "precision": -1.0,
            "recall": -1.0,
            "f1": -1.0,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0
        })

    return result


def get_paragraph_rewards_for_prpo(response_text: str, image_path: str = None) -> Dict[str, Any]:
    """
    Get paragraph-level rewards optimized for PRPO training.

    This function extracts paragraph rewards from the enhanced reward computation
    and returns them in a format suitable for PRPO training.

    Args:
        response_text: Model response text to analyze
        image_path: Path to associated image for CLIP similarity

    Returns:
        Dict with paragraph rewards and metadata for PRPO
    """
    try:
        paragraph_data = compute_enhanced_paragraph_rewards(response_text, image_path)

        return {
            'paragraph_rewards': paragraph_data['paragraph_rewards'],
            'paragraph_metadata': paragraph_data['paragraph_metadata'],
            'num_paragraphs': paragraph_data['num_paragraphs'],
            'extracted_answer': paragraph_data.get('extracted_answer', 'unknown'),
            'prpo_compatible': True
        }

    except Exception as e:
        print(f"[ERROR] Error in get_paragraph_rewards_for_prpo: {str(e)}")
        import os
        MAX_PARAGRAPHS = int(os.environ.get('PRPO_MAX_PARAGRAPHS', 8))
        return {
            'paragraph_rewards': [0.0] * MAX_PARAGRAPHS,
            'paragraph_metadata': [[0.0, 0.0, 0.5, 0.0, 0.5, 0.5]] * MAX_PARAGRAPHS,
            'num_paragraphs': 1,
            'extracted_answer': 'unknown',
            'prpo_compatible': True,
            'error': str(e)
        }


def reward_func(
    data_source, solution_str, ground_truth=None, extra_info=None,
    sandbox_fusion_url=None, concurrent_semaphore=None
):
    """PRPO reward function with paragraph-level CLIP similarity scoring."""
    try:
        image_path = None
        if extra_info and isinstance(extra_info, dict):
            image_path = extra_info.get('image_path')
            if not image_path and 'meta_info' in extra_info:
                meta_info = extra_info['meta_info']
                if isinstance(meta_info, dict):
                    image_path = meta_info.get('image_path')

        is_validation = False
        if extra_info:
            if isinstance(extra_info, dict):
                is_validation = extra_info.get('validate', False)
                if not is_validation and 'meta_info' in extra_info:
                    meta_info = extra_info['meta_info']
                    if isinstance(meta_info, dict):
                        is_validation = meta_info.get('validate', False)
            if hasattr(extra_info, 'get'):
                is_validation = is_validation or extra_info.get('validate', False)
                meta_info = extra_info.get('meta_info', {})
                if isinstance(meta_info, dict):
                    is_validation = is_validation or meta_info.get('validate', False)

        if not is_validation and ground_truth is not None and str(ground_truth).lower() not in ['none', 'unknown', '']:
            is_validation = True

        recovered_ground_truth = ground_truth
        if ground_truth is None or str(ground_truth).lower() == 'none':
            if image_path:
                if '/real/' in image_path:
                    recovered_ground_truth = 'real'
                elif '/fake/' in image_path:
                    recovered_ground_truth = 'fake'

        res = compute_prpo_score(solution_str, image_path, recovered_ground_truth, is_validation)
        mode_str = "PRPO-VAL" if is_validation else "PRPO"


        return res

    except Exception as e:
        print(f"[ERROR] Error in PRPO reward function: {str(e)}")
        traceback.print_exc()
        import os
        MAX_PARAGRAPHS = int(os.environ.get('PRPO_MAX_PARAGRAPHS', 8))
        paragraph_rewards = [0.0] * MAX_PARAGRAPHS
        paragraph_metadata = [[0.0, 0.0, 0.5, 0.0, 0.5, 0.5]] * MAX_PARAGRAPHS
        return {
            "format_score": 0.0,
            "acc": 0.0,
            "extracted_gt": "error",
            "pred": "",
            "paragraph_rewards": paragraph_rewards,
            "paragraph_metadata": paragraph_metadata,
            "num_paragraphs": 1,
            "prpo_enabled": True,
            "image_path": image_path,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
            "error": str(e)
        }
