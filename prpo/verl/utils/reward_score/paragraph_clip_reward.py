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

# paragraph_clip_reward.py
# Paragraph-level reward function using CLIP image-text similarity
# Integrates with PRPO (Paragraph-level GRPO) for fine-grained optimization

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import re
import yake
import os
from typing import List, Dict, Any, Tuple
import numpy as np

# Cache for CLIP model to avoid reloading
_clip_model = None
_clip_processor = None
_device = None

# Cache for YAKE extractor to avoid reloading
_yake_extractor = None

def setup_clip_model():
    """Setup CLIP model for similarity computation."""
    global _clip_model, _clip_processor, _device
    if _clip_model is None:
        print("Loading CLIP model for paragraph-level rewards...")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        _clip_model.eval()
        print(f"CLIP model loaded on {_device}")
    return _clip_model, _clip_processor, _device

def setup_yake_extractor():
    """Setup YAKE keyword extractor for text processing."""
    global _yake_extractor
    if _yake_extractor is None:
        print("🚀 Initializing YAKE keyword extractor...")
        _yake_extractor = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.7, top=1, features=None
        )
        print("✅ YAKE keyword extractor initialized!")
    return _yake_extractor

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs and clean them.
    Args:
        text: Input text response
    Returns:
        List of cleaned paragraphs
    """
    # Split by double newlines, markdown headers, or reasoning markers
    paragraphs = re.split(r'\n\s*\n|\*\*[^*]+\*\*:|\d+\.\s+', text)
    
    cleaned = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:  # Skip very short paragraphs
            continue
            
        # Remove markdown formatting
        para = re.sub(r'\*\*([^*]+)\*\*', r'\1', para)
        para = re.sub(r'\*([^*]+)\*', r'\1', para)
        
        # Remove excessive whitespace
        para = ' '.join(para.split())
        
        if len(para) >= 20:  # Only keep meaningful paragraphs
            cleaned.append(para)
    
    return cleaned if cleaned else [text.strip()]  # Fallback to original text

def extract_keyword_yake(text: str, top_k: int = 1) -> str:
    """
    Extract top keyword from text using YAKE.
    Args:
        text: Input paragraph text
        top_k: Number of top keywords to extract
    Returns:
        Top keyword as string
    """
    try:
        # Use cached YAKE keyword extractor (initialized once)
        kw_extractor = setup_yake_extractor()
        
        # Extract keywords
        keywords = kw_extractor.extract_keywords(text)
        if keywords:
            return keywords[0][1]  # Return the keyword string
        else:
            # Fallback: extract noun phrases or important words
            words = text.split()
            return ' '.join(words[:3]) if words else "content"
    except Exception as e:
        print(f"YAKE extraction failed: {e}")
        # Simple fallback: return first few words
        words = text.split()
        return ' '.join(words[:3]) if words else "content"

def compute_clip_similarity(image_path: str, keyword: str) -> float:
    """
    Compute CLIP similarity between image and keyword.
    Args:
        image_path: Path to image file
        keyword: Text keyword/phrase
    Returns:
        Similarity score in [0, 1] range
    """
    try:
        model, processor, device = setup_clip_model()
        
        # Load and process image
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return 0.5  # Neutral score for missing images
            
        image = Image.open(image_path).convert('RGB')
        
        # Process inputs
        inputs = processor(
            text=[keyword], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        # Compute similarity
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.matmul(image_embeds, text_embeds.t()).item()
            
            # Convert from [-1, 1] to [0, 1] range
            similarity = (similarity + 1) / 2
            
        return float(similarity)
        
    except Exception as e:
        print(f"CLIP similarity computation failed: {e}")
        return 0.5  # Neutral fallback score

def reward_func(responses: List[str], **kwargs) -> List[Dict[str, Any]]:
    """
    Compute paragraph-level rewards using CLIP image-text similarity.
    
    Args:
        responses: List of generated responses
        **kwargs: Additional arguments including:
            - image_paths: List of image paths corresponding to responses
            - prompts: List of prompts (optional)
            - sample_indices: List of sample indices (optional)
    
    Returns:
        List of dictionaries containing paragraph-level rewards and metadata
    """
    image_paths = kwargs.get('image_paths', [])
    prompts = kwargs.get('prompts', [])
    sample_indices = kwargs.get('sample_indices', list(range(len(responses))))
    
    if len(image_paths) != len(responses):
        print(f"Warning: Mismatch between responses ({len(responses)}) and images ({len(image_paths)})")
        # Repeat the first image if needed
        if image_paths:
            image_paths = image_paths * (len(responses) // len(image_paths) + 1)
            image_paths = image_paths[:len(responses)]
        else:
            # No images provided - use dummy rewards
            return [{"total_reward": 0.5, "paragraph_rewards": [0.5], "group_id": i} 
                   for i in range(len(responses))]
    
    results = []
    
    for i, (response, image_path) in enumerate(zip(responses, image_paths)):
        # Split response into paragraphs
        paragraphs = split_into_paragraphs(response)
        
        paragraph_rewards = []
        paragraph_metadata = []
        
        for j, paragraph in enumerate(paragraphs):
            # Extract keyword from paragraph
            keyword = extract_keyword_yake(paragraph)
            
            # Compute CLIP similarity
            similarity_score = compute_clip_similarity(image_path, keyword)
            
            paragraph_rewards.append(similarity_score)
            paragraph_metadata.append({
                'paragraph_index': j,
                'paragraph_text': paragraph[:100] + "..." if len(paragraph) > 100 else paragraph,
                'keyword': keyword,
                'similarity_score': similarity_score
            })
        
        # Compute total reward (mean of paragraph rewards)
        total_reward = np.mean(paragraph_rewards) if paragraph_rewards else 0.5
        
        # Group ID for PRPO (can be sample index or prompt-based grouping)
        group_id = sample_indices[i] if i < len(sample_indices) else i
        
        results.append({
            'total_reward': float(total_reward),
            'paragraph_rewards': paragraph_rewards,
            'paragraph_metadata': paragraph_metadata,
            'group_id': group_id,
            'num_paragraphs': len(paragraphs),
            'image_path': image_path
        })
    
    return results

# Compatibility function for existing TTRL reward interface
def compute_rewards(responses: List[str], **kwargs) -> List[float]:
    """
    Compatibility wrapper that returns simple reward scores.
    Used when PRPO is not enabled.
    """
    paragraph_results = reward_func(responses, **kwargs)
    return [result['total_reward'] for result in paragraph_results]

# PRPO-specific function for paragraph-level rewards
def get_paragraph_level_data(responses: List[str], **kwargs) -> Dict[str, List]:
    """
    Get paragraph-level data for PRPO training.
    
    Returns:
        Dictionary with:
            - paragraph_rewards: List of all paragraph rewards
            - group_ids: List of group IDs for each paragraph
            - paragraph_metadata: List of metadata for each paragraph
            - response_to_paragraph_mapping: Mapping from response index to paragraph indices
    """
    paragraph_results = reward_func(responses, **kwargs)
    
    all_paragraph_rewards = []
    all_group_ids = []
    all_paragraph_metadata = []
    response_to_paragraph_mapping = []
    
    paragraph_idx = 0
    
    for response_idx, result in enumerate(paragraph_results):
        paragraph_rewards = result['paragraph_rewards']
        group_id = result['group_id']
        paragraph_metadata = result['paragraph_metadata']
        
        # Track which paragraphs belong to this response
        response_paragraphs = []
        
        for para_reward, para_metadata in zip(paragraph_rewards, paragraph_metadata):
            all_paragraph_rewards.append(para_reward)
            all_group_ids.append(group_id)
            all_paragraph_metadata.append(para_metadata)
            response_paragraphs.append(paragraph_idx)
            paragraph_idx += 1
        
        response_to_paragraph_mapping.append(response_paragraphs)
    
    return {
        'paragraph_rewards': all_paragraph_rewards,
        'group_ids': all_group_ids,
        'paragraph_metadata': all_paragraph_metadata,
        'response_to_paragraph_mapping': response_to_paragraph_mapping
    }

# Test function
if __name__ == "__main__":
    # Simple test
    test_responses = [
        "This image shows a beautiful landscape. The mountains are visible in the background. The colors are vibrant and natural.",
        "The photograph depicts urban architecture. Modern buildings dominate the skyline. The lighting creates interesting shadows."
    ]
    test_image_paths = ["/tmp/test1.jpg", "/tmp/test2.jpg"]  # Dummy paths for testing
    
    results = reward_func(test_responses, image_paths=test_image_paths)
    print("Test results:")
    for i, result in enumerate(results):
        print(f"Response {i}: {result['total_reward']:.3f} ({result['num_paragraphs']} paragraphs)")