"""
LLM Benchmarking Script for Quranic Similar Verse Identification (Local HuggingFace Edition)
This script evaluates multiple LLMs on their ability to identify verses similar to ground truth verses.
Features:
- Local HuggingFace model inference
- JSONL dataset support with similar_verses evaluation
- Separate result files for each model
- Checkpoint saving  
- Progress tracking
- Memory optimization for MacBook
- Similarity scoring between predicted and ground truth verses
"""

import json
import re
import time
import logging
import os
import gc
import importlib.util
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path

# HuggingFace imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    HF_AVAILABLE = True
    print("✅ HuggingFace transformers available")
except ImportError as e:
    HF_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    print(f"❌ HuggingFace transformers not available: {e}")

class LLMBenchmark:
    def __init__(self, config_path: str = "config.py"):
        """Initialize the benchmark with configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.quran_data = self.load_json(self.config['QURAN_FILE'])
        self.questions_data = self.load_jsonl(self.config['QUESTIONS_FILE'])
        self.quran_lookup = {}
        self.surah_name_mapping = self.create_surah_mapping()
        self.create_verse_lookup()
        
        # Model cache for loaded models
        self._loaded_models = {}
        self._loaded_tokenizers = {}
        
        # Ensure output directory exists
        os.makedirs(self.config['OUTPUT_DIR'], exist_ok=True)
        
    def setup_logging(self):
        """Setup logging with both file and console output"""
        log_file = os.path.join(self.config['OUTPUT_DIR'], 'benchmark.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            return {
                # Local HuggingFace settings
                'USE_LOCAL_MODELS': getattr(config_module, 'USE_LOCAL_MODELS', True),
                'DEVICE': getattr(config_module, 'DEVICE', 'auto'),
                'TORCH_DTYPE': getattr(config_module, 'TORCH_DTYPE', 'auto'),
                'USE_QUANTIZATION': getattr(config_module, 'USE_QUANTIZATION', False),
                'QUANTIZATION_BITS': getattr(config_module, 'QUANTIZATION_BITS', 4),
                
                # Model configuration - HARDCODED SPECIFIC MODEL
                'MODELS': {
                    'deepseek': 'suayptalha/DeepSeek-R1-Distill-Llama-3B',
                },
                
                # Generation parameters
                'GENERATION_PARAMS': getattr(config_module, 'GENERATION_PARAMS', {
                    'temperature': 0.1,
                    'max_new_tokens': 1000,  # Reduced for faster inference
                    'do_sample': True,
                    'top_p': 0.9,
                    'pad_token_id': None,  # Will be set based on tokenizer
                }),
                
                # Request settings
                'DEFAULT_TEMPERATURE': getattr(config_module, 'DEFAULT_TEMPERATURE', 0.1),
                'DEFAULT_MAX_TOKENS': getattr(config_module, 'DEFAULT_MAX_TOKENS', 1000),
                'DELAY_BETWEEN_REQUESTS': getattr(config_module, 'DELAY_BETWEEN_REQUESTS', 1),
                
                # File paths - Updated for JSONL format
                'QUESTIONS_FILE': getattr(config_module, 'QUESTIONS_FILE', 'data.jsonl'),
                'QURAN_FILE': getattr(config_module, 'QURAN_FILE', '.quran_cleaned.json'),
                'OUTPUT_DIR': getattr(config_module, 'OUTPUT_DIR', 'results')
            }
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
            raise

    def load_json(self, file_path: str) -> Any:
        """Load JSON data from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL data from file and validate similar_verses format"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        
                        # Validate and normalize similar_verses if present
                        if 'similar_verses' in item:
                            validated_verses = []
                            for verse in item['similar_verses']:
                                # Normalize the verse reference
                                normalized = self.normalize_verse_reference(verse)
                                
                                # Validate the format is X:Y
                                if re.match(r'^\d+:\d+$', normalized):
                                    validated_verses.append(normalized)
                                else:
                                    logging.warning(f"Line {line_num}: Invalid similar_verse format '{verse}' -> '{normalized}' in question ID {item.get('id', 'unknown')}")
                            
                            item['similar_verses'] = validated_verses
                            logging.debug(f"Line {line_num}: Normalized {len(item['similar_verses'])} similar verses for question {item.get('id', 'unknown')}")
                        
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logging.error(f"Line {line_num}: Failed to parse JSON - {e}")
                        continue
                        
        logging.info(f"Loaded {len(data)} questions from {file_path}")
        return data
    
    def save_json(self, data: Any, file_path: str):
        """Save data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved data to {file_path}")
    
    def create_surah_mapping(self) -> Dict[int, str]:
        """Create mapping from surah number to English name"""
        mapping = {}
        for verse in self.quran_data:
            surah_num = verse['surah_number']
            if surah_num not in mapping:
                mapping[surah_num] = verse['surah_name_english']
        return mapping
    
    def create_verse_lookup(self):
        """Create lookup dictionary for verse validation"""
        for verse in self.quran_data:
            surah_num = verse['surah_number']
            aya_num = verse['aya_number']
            surah_name_en = verse['surah_name_english']
            
            # Key by surah:verse format
            key = f"{surah_num}:{aya_num}"
            self.quran_lookup[key] = {
                'surah_number': surah_num,
                'aya_number': aya_num,
                'surah_name_english': surah_name_en,
                'arabic_clean': verse['arabic_clean'],
                'english_translation': verse['english_translation']
            }
        
        logging.info(f"Loaded {len(self.quran_lookup)} verses for validation")
    
    def get_device(self) -> str:
        """Determine the best device to use"""
        if self.config['DEVICE'] == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return self.config['DEVICE']
    
    def get_torch_dtype(self):
        """Determine the appropriate torch dtype"""
        if self.config['TORCH_DTYPE'] == 'auto':
            device = self.get_device()
            if device in ['cuda', 'mps']:
                return torch.float16
            else:
                return torch.float32
        elif self.config['TORCH_DTYPE'] == 'float16':
            return torch.float16
        elif self.config['TORCH_DTYPE'] == 'bfloat16':
            return torch.bfloat16
        else:
            return torch.float32
    
    def create_quantization_config(self):
        """Create quantization config if enabled"""
        if not self.config.get('USE_QUANTIZATION', False):
            return None
        
        try:
            bits = self.config.get('QUANTIZATION_BITS', 4)
            if bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif bits == 8:
                return BitsAndBytesConfig(load_in_8bit=True)
            else:
                logging.warning(f"Unsupported quantization bits: {bits}")
                return None
        except Exception as e:
            logging.warning(f"Failed to create quantization config: {e}")
            return None
    
    def load_model_and_tokenizer(self, model_name: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model and tokenizer for local inference"""
        if not HF_AVAILABLE:
            logging.error("HuggingFace transformers not available")
            return None, None
        
        # Check if already loaded
        model_id = self.config['MODELS'].get(model_name.lower())
        if not model_id:
            logging.error(f"Model {model_name} not found in config")
            return None, None
        
        if model_id in self._loaded_models:
            logging.info(f"Using cached model: {model_id}")
            return self._loaded_models[model_id], self._loaded_tokenizers[model_id]
        
        try:
            device = self.get_device()
            torch_dtype = self.get_torch_dtype()
            quantization_config = self.create_quantization_config()
            
            logging.info(f"Loading model: {model_id}")
            logging.info(f"Device: {device}, Dtype: {torch_dtype}, Quantization: {quantization_config is not None}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # Set pad token if not available
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch_dtype,
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
                model_kwargs['device_map'] = 'auto'
            else:
                # For non-quantized models, we'll move to device after loading
                pass
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # Move to device if not using quantization
            if not quantization_config:
                model = model.to(device)
            
            # Set generation config
            generation_params = self.config['GENERATION_PARAMS'].copy()
            generation_params['pad_token_id'] = tokenizer.pad_token_id
            
            # Cache the loaded model and tokenizer
            self._loaded_models[model_id] = model
            self._loaded_tokenizers[model_id] = tokenizer
            
            logging.info(f"Successfully loaded {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logging.error(f"Failed to load model {model_id}: {e}")
            return None, None
    
    def create_prompt(self, question: str) -> str:
        """Create the prompt for LLMs to output JSON with verse references for similar verse identification"""
        prompt = f"""
        Given the following question about the Quran, identify up to 5 relevant Quranic verses that best answer or relate to the question.

        Question: {question}

        Please provide your answer as a JSON object with the following structure:
        {{
        "verses": [
            {{
            "surah_name_format": "Al-Baqarah:255",
            "surah_number_format": "2:255",
            "relevance_score": 10
            }},
            {{
            "surah_name_format": "Al-A'raf:113", 
            "surah_number_format": "7:113",
            "relevance_score": 9
            }}
        ]
        }}

        Requirements:
        - List up to 5 most relevant verses that address or relate to the question
        - Order by relevance (highest score first, 1-10 scale)
        - Use exact Surah names (e.g., "Al-Baqarah", "An-Nisa", "Al-A'raf")
        - Provide both name and number formats
        - Focus on verses that have similar themes, concepts, or messages to what the question is asking about
        - Return ONLY the JSON object, no additional text, do NOT add any comments or extra details
        """
        return prompt
    
    def query_local_model(self, model_name: str, prompt: str) -> Optional[str]:
        """Query local HuggingFace model for response"""
        model, tokenizer = self.load_model_and_tokenizer(model_name)
        
        if model is None or tokenizer is None:
            logging.error(f"Failed to load model {model_name}")
            return None
        
        try:
            device = self.get_device()
            generation_params = self.config['GENERATION_PARAMS'].copy()
            
            # Format prompt for instruction-tuned models
            if 'instruct' in model_name.lower() or 'chat' in model_name.lower():
                # Try to use chat template if available
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    # Fallback formatting
                    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = prompt
            
            # Tokenize input
            inputs = tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048  # Limit input length
            ).to(device)
            
            # Set generation parameters
            generation_params['pad_token_id'] = tokenizer.pad_token_id
            generation_params['eos_token_id'] = tokenizer.eos_token_id
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_params
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Remove any remaining chat template artifacts
            for stop_phrase in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                if stop_phrase in response:
                    response = response.split(stop_phrase)[0]
            
            print(f"ARWA before clean up: {response} END")
            # Extract JSON from ```json``` code blocks if present
            if "```json" in response.lower():
                # Find the start of the JSON block
                json_start = response.lower().find("```json")
                if json_start != -1:
                    # Find the content after ```json
                    content_start = response.find("\n", json_start) + 1
                    if content_start > 0:
                        # Find the closing ```
                        json_end = response.find("```", content_start)
                        if json_end != -1:
                            # Extract just the JSON content
                            json_content = response[content_start:json_end].strip()
                            logging.info(f"Extracted JSON from code block: {len(json_content)} characters")
                            return json_content
                        else:
                            logging.warning("Found ```json but no closing ```, using content after ```json")
                            return response[content_start:].strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Error during inference with {model_name}: {e}")
            return None
    
    def extract_verses_from_response(self, response: str) -> Tuple[List[str], List[str]]:
        """Extract verse references from LLM JSON response and sort by relevance"""
        name_format_verses = []
        number_format_verses = []
        
        try:
            # Try to parse as JSON first
            response_clean = response.strip()
            
            # If response starts with { and ends with }, it's likely already clean JSON
            if response_clean.startswith('{') and response_clean.endswith('}'):
                json_str = response_clean
            else:
                # Find JSON object in the response
                start_idx = response_clean.find('{')
                end_idx = response_clean.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_clean[start_idx:end_idx]
                else:
                    raise ValueError("No JSON object found")
            
            parsed_response = json.loads(json_str)
            
            # Extract verses from JSON structure
            verses = parsed_response.get('verses', [])
            
            # Sort verses by relevance score (highest first)
            verses_sorted = sorted(verses, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            for verse in verses_sorted:
                if isinstance(verse, dict):
                    # Extract name format
                    name_format = verse.get('surah_name_format', '')
                    if name_format:
                        # Ensure proper format with brackets
                        if not name_format.startswith('['):
                            name_format = f"[{name_format}]"
                        if name_format not in name_format_verses:
                            name_format_verses.append(name_format)
                    
                    # Extract number format
                    number_format = verse.get('surah_number_format', '')
                    if number_format:
                        # Ensure proper format with brackets
                        if not number_format.startswith('['):
                            number_format = f"[{number_format}]"
                        if number_format not in number_format_verses:
                            number_format_verses.append(number_format)
            
            logging.info(f"Parsed JSON response: {len(verses_sorted)} verses sorted by relevance")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logging.warning(f"Failed to parse JSON response: {e}, falling back to regex parsing")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logging.warning(f"Failed to parse JSON response: {e}, falling back to regex parsing")
            
            # Fallback to original regex-based parsing
            name_patterns = [
                r'\[([A-Za-z\'-]+)-(\d+)\]',
                r'\[([A-Za-z\s\'-]+?)-(\d+)\]',
                r'([A-Za-z\'-]+)-(\d+)',
                r'([A-Za-z\s\'-]+?)-(\d+)',
            ]
            
            number_patterns = [
                r'\[(\d+)-(\d+)\]',
                r'(\d+)-(\d+)',
            ]
            
            # Extract name format verses
            for pattern in name_patterns:
                name_matches = re.findall(pattern, response, re.IGNORECASE)
                for surah_name, verse_num in name_matches:
                    surah_name_clean = surah_name.strip().replace(' ', '-')
                    verse_ref = f"[{surah_name_clean}-{verse_num}]"
                    if verse_ref not in name_format_verses:
                        name_format_verses.append(verse_ref)
            
            # Extract number format verses
            for pattern in number_patterns:
                number_matches = re.findall(pattern, response)
                for surah_num, verse_num in number_matches:
                    verse_ref = f"[{surah_num}-{verse_num}]"
                    if verse_ref not in number_format_verses:
                        number_format_verses.append(verse_ref)
            
            logging.info(f"Regex fallback: {len(name_format_verses)} name format, {len(number_format_verses)} number format verses")
        
        # Limit to 10 verses maximum
        return name_format_verses[:10], number_format_verses[:10]
    
    def normalize_verse_reference(self, verse_ref: str) -> str:
        """Normalize verse reference to standard format (surah:verse)"""
        if not verse_ref:
            return verse_ref
            
        # Remove brackets and clean whitespace
        clean_ref = verse_ref.strip('[]').strip()
        
        # If already in X:Y format, validate and return
        if ':' in clean_ref:
            parts = clean_ref.split(':')
            if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
                surah_num = parts[0].strip()
                verse_num = parts[1].strip()
                return f"{surah_num}:{verse_num}"
            else:
                logging.warning(f"Invalid colon format: '{verse_ref}' -> '{clean_ref}'")
                return clean_ref
        
        # Handle dash format (surah-verse)
        elif '-' in clean_ref:
            parts = clean_ref.split('-')
            if len(parts) == 2:
                surah_part = parts[0].strip()
                verse_part = parts[1].strip()
                
                # If surah_part is numeric, use it directly
                if surah_part.isdigit() and verse_part.isdigit():
                    return f"{surah_part}:{verse_part}"
                else:
                    # Convert surah name to number if possible
                    surah_num = self.get_surah_number_from_name(surah_part)
                    if surah_num and verse_part.isdigit():
                        return f"{surah_num}:{verse_part}"
                    else:
                        logging.warning(f"Could not normalize verse reference: '{verse_ref}'")
        
        # If no recognizable format, return as-is but log warning
        if clean_ref != verse_ref:
            logging.warning(f"Unrecognized verse format: '{verse_ref}' -> '{clean_ref}'")
        
        return clean_ref
    
    def get_surah_number_from_name(self, surah_name: str) -> Optional[int]:
        """Get surah number from English name"""
        surah_name_clean = surah_name.lower().replace('-', ' ').replace('_', ' ')
        
        for surah_num, name in self.surah_name_mapping.items():
            if name.lower().replace('-', ' ').replace('_', ' ') == surah_name_clean:
                return surah_num
        
        return None
    
    def validate_verse_reference(self, verse_ref: str) -> bool:
        """Validate if a verse reference exists in the Quran"""
        normalized = self.normalize_verse_reference(verse_ref)
        return normalized in self.quran_lookup
    
    def calculate_similarity_metrics(self, predicted_verses: List[str], ground_truth_verses: List[str]) -> Dict[str, Any]:
        """Calculate similarity metrics between predicted and ground truth verses"""
        if not ground_truth_verses:
            return {
                'exact_matches': 0,
                'partial_matches': 0,
                'total_predictions': len(predicted_verses),
                'valid_predictions': sum(1 for v in predicted_verses if self.validate_verse_reference(v)),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'jaccard_similarity': 0.0,
                'matched_verses': [],
                'missed_verses': ground_truth_verses,
                'extra_verses': predicted_verses
            }
        
        # Normalize all verse references
        pred_normalized = set()
        for verse in predicted_verses:
            normalized = self.normalize_verse_reference(verse)
            if self.validate_verse_reference(verse):
                pred_normalized.add(normalized)
                logging.debug(f"Predicted verse normalized: '{verse}' -> '{normalized}'")
        
        truth_normalized = set()
        for verse in ground_truth_verses:
            normalized = self.normalize_verse_reference(verse)
            truth_normalized.add(normalized)
            logging.debug(f"Ground truth verse normalized: '{verse}' -> '{normalized}'")
        
        # Validate that all normalized verses are in the same format (X:Y)
        def validate_format(verse_set, set_name):
            for verse in verse_set:
                if not re.match(r'^\d+:\d+$', verse):
                    logging.warning(f"Inconsistent format in {set_name}: '{verse}' - expected format X:Y")
                    return False
            return True
        
        validate_format(pred_normalized, "predicted verses")
        validate_format(truth_normalized, "ground truth verses")
        
        logging.info(f"Normalized predicted verses: {sorted(pred_normalized)}")
        logging.info(f"Normalized ground truth verses: {sorted(truth_normalized)}")
        
        # Calculate metrics
        exact_matches = len(pred_normalized.intersection(truth_normalized))
        
        # Calculate precision, recall, F1
        precision = exact_matches / len(pred_normalized) if pred_normalized else 0.0
        recall = exact_matches / len(truth_normalized) if truth_normalized else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Jaccard similarity
        union_size = len(pred_normalized.union(truth_normalized))
        jaccard = exact_matches / union_size if union_size > 0 else 0.0
        
        # Identify matched, missed, and extra verses
        matched_verses = list(pred_normalized.intersection(truth_normalized))
        missed_verses = list(truth_normalized - pred_normalized)
        extra_verses = list(pred_normalized - truth_normalized)
        
        return {
            'exact_matches': exact_matches,
            'total_predictions': len(predicted_verses),
            'valid_predictions': len(pred_normalized),
            'ground_truth_count': len(truth_normalized),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'jaccard_similarity': jaccard,
            'matched_verses': matched_verses,
            'missed_verses': missed_verses,
            'extra_verses': extra_verses
        }
    
    def clear_model_cache(self):
        """Clear loaded models from memory"""
        for model_id in list(self._loaded_models.keys()):
            del self._loaded_models[model_id]
            del self._loaded_tokenizers[model_id]
        
        self._loaded_models.clear()
        self._loaded_tokenizers.clear()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logging.info("Cleared model cache and GPU memory")
    
    def save_checkpoint(self, model_name: str, results: List[Dict], question_index: int):
        """Save checkpoint for a model"""
        checkpoint_file = os.path.join(self.config['OUTPUT_DIR'], f'{model_name}_checkpoint_{question_index}.json')
        checkpoint_data = {
            'model': model_name,
            'questions_completed': question_index + 1,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        self.save_json(checkpoint_data, checkpoint_file)
        logging.info(f"Checkpoint saved for {model_name} at question {question_index + 1}")
    
    def load_checkpoint(self, model_name: str) -> Tuple[List[Dict], int]:
        """Load checkpoint for a model if it exists"""
        checkpoint_files = []
        for file in os.listdir(self.config['OUTPUT_DIR']):
            if file.startswith(f'{model_name}_checkpoint_') and file.endswith('.json'):
                checkpoint_files.append(file)
        
        if not checkpoint_files:
            return [], 0
        
        # Sort by question number and get the latest
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
        
        checkpoint_path = os.path.join(self.config['OUTPUT_DIR'], latest_checkpoint)
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            last_question = data.get('questions_completed', 0)
            
            logging.info(f"Loaded checkpoint for {model_name}: {last_question} questions completed")
            return results, last_question
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return [], 0
    
    def check_local_requirements(self):
        """Check if local model requirements are met"""
        if not HF_AVAILABLE:
            logging.error("HuggingFace transformers not available. Install with: pip install transformers torch")
            return False
        
        device = self.get_device()
        logging.info(f"Using device: {device}")
        
        if device == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        elif device == 'mps' and torch.backends.mps.is_available():
            logging.info("Using Apple Silicon MPS backend")
        else:
            logging.info("Using CPU (will be slower)")
        
        return True
    
    def run_model_benchmark(self, model_name: str, max_questions: int = None, resume: bool = True):
        """Run benchmark for a single model with checkpointing"""
        logging.info(f"Starting benchmark for {model_name.upper()}")
        
        # Check local requirements
        if not self.check_local_requirements():
            logging.error("Local model requirements not met.")
            return []
        
        # Load checkpoint if resuming
        if resume:
            model_results, start_question = self.load_checkpoint(model_name)
        else:
            model_results, start_question = [], 0
        
        questions_to_test = self.questions_data[:max_questions] if max_questions else self.questions_data
        total_questions = len(questions_to_test)
        
        # Skip already completed questions
        questions_to_process = questions_to_test[start_question:]
        
        if not questions_to_process:
            logging.info(f"All questions already completed for {model_name}")
            return model_results
        
        logging.info(f"Processing {len(questions_to_process)} questions for {model_name} (starting from {start_question + 1})")
        
        successful_responses = len([r for r in model_results if r.get('llm_response')])
        total_f1_score = sum(r.get('similarity_metrics', {}).get('f1_score', 0) for r in model_results)
        total_precision = sum(r.get('similarity_metrics', {}).get('precision', 0) for r in model_results)
        total_recall = sum(r.get('similarity_metrics', {}).get('recall', 0) for r in model_results)
        
        for i, question_data in enumerate(questions_to_process):
            current_question_index = start_question + i
            question_id = question_data['id']
            question = question_data['query']
            ground_truth_verses = question_data.get('similar_verses', [])
            bloom_level = question_data.get('Bloom Taxonomy Level', 'Unknown')
            
            logging.info(f"Processing question {current_question_index + 1}/{total_questions} (ID: {question_id}) for {model_name}")
            logging.info(f"  Ground truth verses: {ground_truth_verses}")
            
            # Create prompt and get response
            prompt = self.create_prompt(question)
            start_time = time.time()
            response = self.query_local_model(model_name, prompt)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response:
                print(f"ARWA response: {response} END ARWA")
                successful_responses += 1
                
                # Parse response for basic format (for backward compatibility)
                name_format_verses, number_format_verses = self.extract_verses_from_response(response)
                all_predicted_verses = name_format_verses + number_format_verses
                
                # Get detailed verse information with relevance scores
                detailed_verses = self.extract_detailed_verses_from_response(response)
                
                # Calculate similarity metrics with ground truth
                similarity_metrics = self.calculate_similarity_metrics(all_predicted_verses, ground_truth_verses)
                
                # Update running totals
                total_f1_score += similarity_metrics['f1_score']
                total_precision += similarity_metrics['precision']
                total_recall += similarity_metrics['recall']
                
                logging.info(f"  Response: {len(response)} chars, Found {len(detailed_verses)} verses")
                logging.info(f"  Similarity metrics - F1: {similarity_metrics['f1_score']:.3f}, Precision: {similarity_metrics['precision']:.3f}, Recall: {similarity_metrics['recall']:.3f}")
                logging.info(f"  Matched verses: {similarity_metrics['matched_verses']}")
            else:
                logging.warning(f"  Failed to get response for question {question_id}")
                similarity_metrics = {
                    'exact_matches': 0,
                    'total_predictions': 0,
                    'valid_predictions': 0,
                    'ground_truth_count': len(ground_truth_verses),
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'jaccard_similarity': 0.0,
                    'matched_verses': [],
                    'missed_verses': ground_truth_verses,
                    'extra_verses': []
                }
                name_format_verses = []
                number_format_verses = []
                detailed_verses = []
            
            question_result = {
                'question_id': question_id,
                'query': question,
                'bloom_level': bloom_level,
                'ground_truth_verses': ground_truth_verses,
                'predicted_verses_name_format': name_format_verses,
                'predicted_verses_number_format': number_format_verses,
                'detailed_verse_predictions': detailed_verses,
                'similarity_metrics': similarity_metrics,
                'llm_response': response,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            model_results.append(question_result)
            
            # Save checkpoint every 5 questions (more frequent for local testing)
            if (current_question_index + 1) % 5 == 0:
                self.save_checkpoint(model_name, model_results, current_question_index)
                
                # Print progress
                current_avg_f1 = total_f1_score / successful_responses if successful_responses > 0 else 0
                current_avg_precision = total_precision / successful_responses if successful_responses > 0 else 0
                current_avg_recall = total_recall / successful_responses if successful_responses > 0 else 0
                logging.info(f"  Progress: {current_question_index + 1}/{total_questions}")
                logging.info(f"  Running averages - F1: {current_avg_f1:.3f}, Precision: {current_avg_precision:.3f}, Recall: {current_avg_recall:.3f}")
            
            # Small delay between requests to prevent overheating
            time.sleep(self.config['DELAY_BETWEEN_REQUESTS'])
        
        # Clear model from memory after completion
        self.clear_model_cache()
        
        # Save final results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_file = os.path.join(self.config['OUTPUT_DIR'], f'{model_name}_final_{timestamp}.json')
        
        # Calculate final metrics
        final_avg_f1 = total_f1_score / successful_responses if successful_responses > 0 else 0
        final_avg_precision = total_precision / successful_responses if successful_responses > 0 else 0
        final_avg_recall = total_recall / successful_responses if successful_responses > 0 else 0
        
        # Calculate aggregate metrics
        total_exact_matches = sum(r['similarity_metrics']['exact_matches'] for r in model_results)
        total_predictions = sum(r['similarity_metrics']['total_predictions'] for r in model_results)
        total_valid_predictions = sum(r['similarity_metrics']['valid_predictions'] for r in model_results)
        total_ground_truth = sum(r['similarity_metrics']['ground_truth_count'] for r in model_results)
        
        # Overall precision, recall, F1
        overall_precision = total_exact_matches / total_valid_predictions if total_valid_predictions > 0 else 0
        overall_recall = total_exact_matches / total_ground_truth if total_ground_truth > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        final_results = {
            'model': model_name,
            'model_id': self.config['MODELS'].get(model_name.lower(), ''),
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(model_results),
            'successful_responses': successful_responses,
            'aggregate_metrics': {
                'total_exact_matches': total_exact_matches,
                'total_predictions': total_predictions,
                'total_valid_predictions': total_valid_predictions,
                'total_ground_truth_verses': total_ground_truth,
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1_score': overall_f1
            },
            'average_metrics': {
                'avg_f1_score': final_avg_f1,
                'avg_precision': final_avg_precision,
                'avg_recall': final_avg_recall,
                'avg_response_time': sum(r['response_time'] for r in model_results if r['response_time']) / len([r for r in model_results if r['response_time']]) if any(r['response_time'] for r in model_results) else 0
            },
            'questions': model_results
        }
        
        self.save_json(final_results, final_file)
        
        logging.info(f"Final results for {model_name}:")
        logging.info(f"  Total questions: {len(model_results)}")
        logging.info(f"  Successful responses: {successful_responses}")
        logging.info(f"  Overall F1 score: {overall_f1:.3f}")
        logging.info(f"  Overall Precision: {overall_precision:.3f}")
        logging.info(f"  Overall Recall: {overall_recall:.3f}")
        logging.info(f"  Average F1 score: {final_avg_f1:.3f}")
        logging.info(f"  Average response time: {final_results['average_metrics']['avg_response_time']:.2f}s")
        
        return model_results
    
    def run_all_benchmarks(self, models: List[str], max_questions: int = None, resume: bool = True):
        """Run benchmarks for all specified models"""
        logging.info("Starting LLM Benchmark for Quranic Similar Verse Identification (Local Models)")
        logging.info(f"Models to test: {', '.join(models)}")
        logging.info(f"Questions limit: {max_questions or 'All'}")
        
        # Check local requirements
        if not self.check_local_requirements():
            logging.error("Local model requirements not met. Please install required dependencies.")
            return
        
        # Check available models
        available_models = [m for m in models if m.lower() in self.config['MODELS']]
        if len(available_models) != len(models):
            skipped = [m for m in models if m.lower() not in [model.lower() for model in available_models]]
            logging.warning(f"Skipping unknown models: {', '.join(skipped)}")
        
        if not available_models:
            logging.error("No available models to test")
            return
        
        all_results = {}
        
        for model_name in available_models:
            try:
                logging.info(f"\n{'='*60}")
                logging.info(f"Starting evaluation for {model_name.upper()}")
                logging.info(f"{'='*60}")
                
                results = self.run_model_benchmark(model_name, max_questions, resume)
                all_results[model_name] = results
                
                # Clear model from memory after each model to free up resources
                self.clear_model_cache()
                
            except Exception as e:
                logging.error(f"Error running benchmark for {model_name}: {e}")
                # Clear cache even on error
                self.clear_model_cache()
                continue
        
        # Save combined summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': list(all_results.keys()),
            'total_questions': max_questions or len(self.questions_data),
            'config_used': {
                'device': self.get_device(),
                'torch_dtype': str(self.get_torch_dtype()),
                'quantization': self.config.get('USE_QUANTIZATION', False),
                'models': {k: v for k, v in self.config['MODELS'].items() if k in [m.lower() for m in available_models]}
            },
            'summary': {}
        }
        
        for model_name, results in all_results.items():
            if results:
                successful = len([r for r in results if r.get('llm_response')])
                
                # Calculate aggregate metrics
                total_exact_matches = sum(r['similarity_metrics']['exact_matches'] for r in results)
                total_predictions = sum(r['similarity_metrics']['total_predictions'] for r in results)
                total_valid_predictions = sum(r['similarity_metrics']['valid_predictions'] for r in results)
                total_ground_truth = sum(r['similarity_metrics']['ground_truth_count'] for r in results)
                
                # Overall metrics
                overall_precision = total_exact_matches / total_valid_predictions if total_valid_predictions > 0 else 0
                overall_recall = total_exact_matches / total_ground_truth if total_ground_truth > 0 else 0
                overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
                
                # Average metrics
                avg_f1 = sum(r['similarity_metrics']['f1_score'] for r in results) / successful if successful > 0 else 0
                avg_precision = sum(r['similarity_metrics']['precision'] for r in results) / successful if successful > 0 else 0
                avg_recall = sum(r['similarity_metrics']['recall'] for r in results) / successful if successful > 0 else 0
                avg_time = sum(r['response_time'] for r in results if r.get('response_time', 0)) / len(results) if results else 0
                
                summary['summary'][model_name] = {
                    'model_id': self.config['MODELS'].get(model_name.lower(), ''),
                    'total_questions': len(results),
                    'successful_responses': successful,
                    'overall_metrics': {
                        'precision': overall_precision,
                        'recall': overall_recall,
                        'f1_score': overall_f1,
                        'total_exact_matches': total_exact_matches,
                        'total_valid_predictions': total_valid_predictions,
                        'total_ground_truth_verses': total_ground_truth
                    },
                    'average_metrics': {
                        'avg_precision': avg_precision,
                        'avg_recall': avg_recall,
                        'avg_f1_score': avg_f1,
                        'avg_response_time': avg_time
                    }
                }
        
        summary_file = os.path.join(self.config['OUTPUT_DIR'], f'benchmark_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.save_json(summary, summary_file)
        
        # Print final summary
        print("\n" + "="*80)
        print("🏆 BENCHMARK SUMMARY (LOCAL MODELS - SIMILAR VERSE IDENTIFICATION)")
        print("="*80)
        print(f"Device: {self.get_device()}")
        print(f"Quantization: {'Enabled' if self.config.get('USE_QUANTIZATION') else 'Disabled'}")
        print("="*80)
        
        for model_name, stats in summary['summary'].items():
            print(f"\n{model_name.upper()} ({stats['model_id']}):")
            print(f"  Questions: {stats['total_questions']}")
            print(f"  Successful: {stats['successful_responses']}")
            print(f"  Overall Metrics:")
            print(f"    F1 Score: {stats['overall_metrics']['f1_score']:.3f}")
            print(f"    Precision: {stats['overall_metrics']['precision']:.3f}")
            print(f"    Recall: {stats['overall_metrics']['recall']:.3f}")
            print(f"    Exact Matches: {stats['overall_metrics']['total_exact_matches']}")
            print(f"  Average Metrics:")
            print(f"    Avg F1 Score: {stats['average_metrics']['avg_f1_score']:.3f}")
            print(f"    Avg Precision: {stats['average_metrics']['avg_precision']:.3f}")
            print(f"    Avg Recall: {stats['average_metrics']['avg_recall']:.3f}")
            print(f"    Avg Response Time: {stats['average_metrics']['avg_response_time']:.2f}s")

    def extract_detailed_verses_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract detailed verse information including relevance scores from LLM response"""
        detailed_verses = []
        
        try:
            # Try to parse as JSON first
            response_clean = response.strip()
            
            # If response starts with { and ends with }, it's likely already clean JSON
            if response_clean.startswith('{') and response_clean.endswith('}'):
                json_str = response_clean
            else:
                # Find JSON object in the response
                start_idx = response_clean.find('{')
                end_idx = response_clean.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_clean[start_idx:end_idx]
                else:
                    raise ValueError("No JSON object found")
            
            parsed_response = json.loads(json_str)
            
            # Extract verses from JSON structure
            verses = parsed_response.get('verses', [])
            
            # Sort verses by relevance score (highest first)
            verses_sorted = sorted(verses, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            for i, verse in enumerate(verses_sorted[:10]):  # Limit to 10
                if isinstance(verse, dict):
                    name_format = verse.get('surah_name_format', '')
                    number_format = verse.get('surah_number_format', '')
                    relevance_score = verse.get('relevance_score', 0)
                    
                    # Ensure proper format with brackets
                    if name_format and not name_format.startswith('['):
                        name_format = f"[{name_format}]"
                    if number_format and not number_format.startswith('['):
                        number_format = f"[{number_format}]"
                    
                    verse_info = {
                        'rank': i + 1,
                        'surah_name_format': name_format,
                        'surah_number_format': number_format,
                        'relevance_score': relevance_score,
                        'is_valid_name': self.validate_verse_reference(name_format) if name_format else False,
                        'is_valid_number': self.validate_verse_reference(number_format) if number_format else False,
                        'source': 'json_parsed'
                    }
                    detailed_verses.append(verse_info)
            
            logging.info(f"Successfully parsed {len(detailed_verses)} verses from JSON with relevance scores")
            return detailed_verses
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logging.warning(f"Failed to parse JSON response: {e}, falling back to regex parsing")
        
        # Fallback: Use existing parsing but create detailed structure
        name_format_verses, number_format_verses = self.extract_verses_from_response(response)
        all_verses = name_format_verses + number_format_verses
        
        for i, verse in enumerate(all_verses[:10]):
            verse_info = {
                'rank': i + 1,
                'surah_name_format': verse if verse in name_format_verses else '',
                'surah_number_format': verse if verse in number_format_verses else '',
                'relevance_score': 10 - i,  # Assign decreasing relevance scores
                'is_valid_name': self.validate_verse_reference(verse) if verse in name_format_verses else False,
                'is_valid_number': self.validate_verse_reference(verse) if verse in number_format_verses else False,
                'source': 'regex_parsed'
            }
            detailed_verses.append(verse_info)
        
        logging.info(f"Regex fallback created {len(detailed_verses)} detailed verses")
        return detailed_verses


def main():
    """Main function to run the benchmark"""
    import argparse
    
    # Hardcoded available models
    available_models = ['deepseek']
    default_model = 'deepseek'
    
    parser = argparse.ArgumentParser(description='LLM Benchmark for Quranic Similar Verse Identification (Local Models)')
    parser.add_argument('--models', nargs='+', default=[default_model],
                       choices=available_models,
                       help=f'Models to benchmark. Available: {", ".join(available_models)}')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of questions (for testing)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from scratch instead of resuming from checkpoint')
    parser.add_argument('--config', default='config.py',
                       help='Path to configuration file (not used, kept for compatibility)')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = LLMBenchmark(args.config)
    
    # Run benchmark
    benchmark.run_all_benchmarks(
        models=args.models,
        max_questions=args.limit,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()