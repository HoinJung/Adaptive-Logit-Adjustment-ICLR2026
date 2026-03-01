import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen', type=str, choices=['llava', 'paligemma', 'qwen'], help='Model to use: llava, paligemma, or qwen')
parser.add_argument('--mode', default='logit', type=str)
parser.add_argument('--gpu_id', default='2', type=str)
parser.add_argument('--lam', default=5.0, type=float)
parser.add_argument('--lam_gender', default=None, type=float, help='Lambda for gender debiasing (only used in intersection mode). If not set, uses --lam')
parser.add_argument('--lam_race', default=None, type=float, help='Lambda for race debiasing (only used in intersection mode). If not set, uses --lam')
parser.add_argument('--decoder_prune_num', default=50, type=int)
parser.add_argument('--t', default=0.5, type=float)
parser.add_argument('--debiasing_target', default='gender', type=str, choices=['gender', 'race', 'intersection'], help='Target attribute for debiasing: gender, race, or intersection (both)')
parser.add_argument(
    '-n',
    '--neutral',
    action='store_true',
    help='Description of your feature.',
)
parser.add_argument(
    '--debug',
    action='store_true',
    help='Debug mode: use reference answers for non-Male/Female detected_gender, only run inference for Male/Female.',
)
parser.add_argument(
    '--reference_file',
    default='result/reference.csv',
    type=str,
    help='Path to reference CSV file for debug mode.',
)
parser.add_argument(
    '--filter_image',
    default=None,
    type=str,
    help='Filter to process only a specific image filename (e.g., sa_2669421.jpg).',
)
import os
import sys
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# Repo root and config for paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from config import DATA_DIR, NLP_CLASSIFICATION_DIR
_FACET_DIR = os.path.join(DATA_DIR, "facet")
_EMBEDDING_DIR = os.path.join(_REPO_ROOT, "tasks", "facet", "embedding")
import torch
from transformers import AutoProcessor, AutoTokenizer
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

from ala.utils import decide_gender, evaluate_facet_open, load_and_normalize_beta, gender_words
import nltk
from ala.model import SimpleTransformerClassifier
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None  # install qwen-vl-utils when using Qwen model
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import copy

# Import model-specific classes
if args.model == 'llava':
    from ala.llava_model import CustomLlavaForConditionalGeneration
elif args.model == 'paligemma':
    from ala.paligemma_model import create_custom_paligemma_model
    from transformers import PaliGemmaProcessor
elif args.model == 'qwen':
    from ala.qwen_model import CustomQwenForConditionalGeneration

# Load FACET data
new_annotations = pd.DataFrame(columns=['filename', 'class1', 'gender', 'race', 'age'])
annotations = pd.read_csv(os.path.join(_FACET_DIR, 'annotations', 'annotations.csv'))
device = f"cuda:0" if torch.cuda.is_available() else "cpu"

# Load FACET data (only if the file doesn't exist)
new_annotations_file = os.path.join(_FACET_DIR, 'new_annotations.csv')

if os.path.exists(new_annotations_file):
    print("Loading preprocessed annotations...")
    new_annotations = pd.read_csv(new_annotations_file)

# Determine sensitive attribute index based on debiasing_target
if args.debiasing_target == 'gender':
    sens_idx = 1
elif args.debiasing_target == 'race':
    sens_idx = 2
elif args.debiasing_target == 'intersection':
    sens_idx = [1, 2]  # Both gender and race
else:
    sens_idx = 1  # default to gender

# Define model_name_suffix for all modes (used in multiple places)
if args.model == 'llava':
    model_name_suffix = 'llava'
elif args.model == 'paligemma':
    model_name_suffix = 'paligemma'
elif args.model == 'qwen':
    model_name_suffix = 'qwen'

# Define race-related words based on Stereoset and common race/ethnicity terms
# Based on MultiClassRaceLogitProcessor and Stereoset dataset categories
race_words = [
    # East Asian
    'asian', 'east', 'eastern', 'chinese', 'japanese', 'korean', 'oriental', 
    'china', 'japan', 'korea', 'taiwanese', 'taiwan', 'hong', 'kong',
    # Indian / South Asian
    'indian', 'india', 'south', 'asian', 'pakistani', 'pakistan', 'bangladeshi', 
    'bangladesh', 'sri', 'lankan', 'nepali', 'nepal', 'bengali',
    # Black / African
    'black', 'african', 'afro', 'africa', 'african-american', 'afro-american',
    'nigeria', 'nigerian', 'ethiopia', 'ethiopian', 'kenya', 'kenyan',
    # White / Caucasian / European
    'white', 'caucasian', 'european', 'europe', 'british', 'britain', 'english',
    'german', 'germany', 'french', 'france', 'italian', 'italy', 'spanish', 'spain',
    'russian', 'russia', 'polish', 'poland', 'irish', 'ireland', 'scottish', 'scotland',
    # Middle Eastern
    'middle', 'eastern', 'arab', 'arabic', 'persian', 'iranian', 'iran', 'turkish', 
    'turkey', 'iraqi', 'iraq', 'syrian', 'syria', 'lebanese', 'lebanon', 'egyptian', 
    'egypt', 'saudi', 'arabia', 'palestinian', 'palestine', 'israeli', 'israel',
    # Latino / Hispanic
    'latino', 'hispanic', 'latin', 'mexican', 'mexico', 'spanish', 'brazilian', 
    'brazil', 'argentina', 'argentine', 'chilean', 'chile', 'colombian', 'colombia',
    'peruvian', 'peru', 'venezuelan', 'venezuela', 'cuban', 'cuba', 'puerto', 'rican',
    # Southeast Asian
    'southeast', 'vietnamese', 'vietnam', 'thai', 'thailand', 'filipino', 'philippines',
    'indonesian', 'indonesia', 'malaysian', 'malaysia', 'singaporean', 'singapore',
    'cambodian', 'cambodia', 'laotian', 'laos', 'myanmar', 'burmese',
    # Other common terms
    'ethnicity', 'ethnic', 'race', 'racial', 'nationality', 'national'
]

def hardcode_race_token_bias(tokenizer, race_words, model_name_suffix='qwen', cache_dir='cache'):
    """
    Hardcode race token bias: 1.0 for race-related tokens, 0.0 for others.
    Ultra-fast version: only check exact matches and simple substring matches.
    Uses caching to avoid recomputing on every run.
    
    Args:
        tokenizer: The tokenizer to get vocabulary from
        race_words: List of race-related words to match
        model_name_suffix: Model suffix for cache file naming
        cache_dir: Directory to store cache files
    
    Returns:
        dict: Token bias dictionary with hardcoded values (only contains 0.0 and 1.0)
    """
    if tokenizer is None:
        return {}
    
    # Check for cached version
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"hardcoded_race_token_bias_{model_name_suffix}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading cached race token bias from {cache_file}...")
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Convert string keys back to float values (JSON doesn't support float keys)
                hardcoded_bias = {k: float(v) for k, v in cached_data.items()}
                race_token_count = sum(1 for v in hardcoded_bias.values() if v == 1.0)
                print(f"Loaded cached race token bias: {race_token_count} race tokens (out of {len(hardcoded_bias)} total)")
                return hardcoded_bias
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}. Recomputing...")
    
    race_words_lower = [word.lower() for word in race_words if len(word) >= 2]  # Skip single chars
    race_words_set = set(race_words_lower)
    
    hardcoded_bias = {}
    matching_tokens = []
    
    # Iterate through all tokens in vocabulary
    vocab_size = len(tokenizer)
    print(f"Hardcoding race token bias for {vocab_size} tokens (this may take a while, but will be cached)...")
    
    # Pre-build a set of all race word substrings for fast lookup
    race_substrings = set(race_words_lower)
    for word in race_words_lower:
        # Add common variations
        race_substrings.add(word)
    
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            if token_str is None:
                continue
            
            # Normalize token string
            token_normalized = token_str.lower().strip('Ġ▁')
            
            # Skip empty or single character tokens
            if not token_normalized or len(token_normalized) == 1:
                hardcoded_bias[token_str] = 0.0
                continue
            
            # Fast matching: exact match or simple substring check
            is_race_token = False
            if token_normalized in race_words_set:
                is_race_token = True
            else:
                # Quick substring check - only check if any race word appears in token
                for word in race_words_lower:
                    if word in token_normalized:
                        is_race_token = True
                        break
            
            if is_race_token:
                hardcoded_bias[token_str] = 1.0
                matching_tokens.append(token_str)
            else:
                hardcoded_bias[token_str] = 0.0
        except (IndexError, ValueError, TypeError):
            continue
    
    race_token_count = len(matching_tokens)
    print(f"Hardcoded race token bias: {race_token_count} race tokens (out of {len(hardcoded_bias)} total)")
    if matching_tokens:
        print(f"Sample: {sorted(set(matching_tokens))[:10]}")
    
    # Save to cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(hardcoded_bias, f)
        print(f"Cached race token bias to {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return hardcoded_bias

def hardcode_gender_token_bias(tokenizer, gender_words, model_name_suffix='qwen', cache_dir='cache'):
    """
    Hardcode token bias: 1.0 for gender-related tokens, 0.0 for others.
    Ultra-fast version: only check exact matches and simple substring matches.
    Uses caching to avoid recomputing on every run.
    
    Args:
        tokenizer: The tokenizer to get vocabulary from
        gender_words: List of gender-related words to match
        model_name_suffix: Model suffix for cache file naming
        cache_dir: Directory to store cache files
    
    Returns:
        dict: Token bias dictionary with hardcoded values (only contains 0.0 and 1.0)
    """
    if tokenizer is None:
        return {}
    
    # Check for cached version
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"hardcoded_gender_token_bias_{model_name_suffix}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading cached gender token bias from {cache_file}...")
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Convert string keys back to float values (JSON doesn't support float keys)
                hardcoded_bias = {k: float(v) for k, v in cached_data.items()}
                gender_token_count = sum(1 for v in hardcoded_bias.values() if v == 1.0)
                print(f"Loaded cached gender token bias: {gender_token_count} gender tokens (out of {len(hardcoded_bias)} total)")
                return hardcoded_bias
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}. Recomputing...")
    
    gender_words_lower = [word.lower() for word in gender_words if len(word) >= 2]  # Skip single chars
    gender_words_set = set(gender_words_lower)
    
    hardcoded_bias = {}
    matching_tokens = []
    
    # Iterate through all tokens in vocabulary
    vocab_size = len(tokenizer)
    print(f"Hardcoding gender token bias for {vocab_size} tokens (this may take a while, but will be cached)...")
    
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            if token_str is None:
                continue
            
            # Normalize token string
            token_normalized = token_str.lower().strip('Ġ▁')
            
            # Skip empty or single character tokens
            if not token_normalized or len(token_normalized) == 1:
                hardcoded_bias[token_str] = 0.0
                continue
            
            # Fast matching: exact match or simple substring check
            is_gender_token = False
            if token_normalized in gender_words_set:
                is_gender_token = True
            else:
                # Quick substring check - only check if any gender word appears in token
                for word in gender_words_lower:
                    if word in token_normalized:
                        is_gender_token = True
                        break
            
            if is_gender_token:
                hardcoded_bias[token_str] = 1.0
                matching_tokens.append(token_str)
            else:
                hardcoded_bias[token_str] = 0.0
        except (IndexError, ValueError, TypeError):
            continue
    
    gender_token_count = len(matching_tokens)
    print(f"Hardcoded gender token bias: {gender_token_count} gender tokens (out of {len(hardcoded_bias)} total)")
    if matching_tokens:
        print(f"Sample: {sorted(set(matching_tokens))[:10]}")
    
    # Save to cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(hardcoded_bias, f)
        print(f"Cached gender token bias to {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return hardcoded_bias

tokenizer = None
alpha_tokenizer = None
if args.mode=='logit':
    # Load tokenizer based on model
    if args.model == 'llava':
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    elif args.model == 'paligemma':
        # Will be loaded with processor
        pass
    elif args.model == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    if tokenizer is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        alpha_tokenizer = copy.deepcopy(tokenizer)
        if alpha_tokenizer.pad_token is None:
            alpha_tokenizer.pad_token = alpha_tokenizer.eos_token
    
    # Get vocab size - must match the training script
    if args.model == 'llava':
        # Use tokenizer.vocab_size to match training script
        vocab_size = tokenizer.vocab_size
    elif args.model == 'paligemma':
        # Will be set after processor is loaded
        vocab_size = None
    elif args.model == 'qwen':
        # Use tokenizer.vocab_size to match training script
        vocab_size = tokenizer.vocab_size
    
    # Use max_length=128 to match training script (stereoset_classification_qwen.py uses default 128)
    max_length = 128
    text_classifier = SimpleTransformerClassifier(vocab_size=vocab_size, max_length=max_length)
    
    # Load gender or race classifier based on debiasing_target
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.debiasing_target == 'intersection':
        # Load both gender and race classifiers and token biases
        gender_model_path = os.path.join(NLP_CLASSIFICATION_DIR, f"gender_model_{model_name_suffix}_pytorch_generated", "pytorch_model.bin")
        gender_importance_path = os.path.join(NLP_CLASSIFICATION_DIR, f"importance_dict_{model_name_suffix}_pytorch_generated.json")
        race_model_path = os.path.join(NLP_CLASSIFICATION_DIR, f"race_model_{model_name_suffix}_pytorch_generated", "pytorch_model.bin")
        race_importance_path = os.path.join(NLP_CLASSIFICATION_DIR, f"importance_race_dict_{model_name_suffix}_pytorch_generated.json")
        
        # Load gender classifier - first inspect checkpoint to get correct parameters
        if os.path.exists(gender_model_path):
            checkpoint = torch.load(gender_model_path, map_location='cpu')
            # Infer vocab_size and max_length from checkpoint
            if 'token_embedding.weight' in checkpoint:
                gender_vocab_size = checkpoint['token_embedding.weight'].shape[0]
            else:
                gender_vocab_size = vocab_size
            if 'position_embedding.weight' in checkpoint:
                gender_max_length = checkpoint['position_embedding.weight'].shape[0]
            else:
                gender_max_length = max_length
            gender_classifier = SimpleTransformerClassifier(vocab_size=gender_vocab_size, max_length=gender_max_length)
            gender_classifier.load_state_dict(checkpoint)
            print(f"Loaded gender classifier with vocab_size={gender_vocab_size}, max_length={gender_max_length}")
        else:
            print(f"Warning: Gender classifier model not found at {gender_model_path}. Using untrained model.")
            gender_classifier = SimpleTransformerClassifier(vocab_size=vocab_size, max_length=max_length)
        gender_classifier.to(device)
        gender_classifier.eval()
        
        # Load race classifier - first inspect checkpoint to get correct parameters
        if os.path.exists(race_model_path):
            checkpoint = torch.load(race_model_path, map_location='cpu')
            # Infer vocab_size and max_length from checkpoint
            if 'token_embedding.weight' in checkpoint:
                race_vocab_size = checkpoint['token_embedding.weight'].shape[0]
            else:
                race_vocab_size = vocab_size
            if 'position_embedding.weight' in checkpoint:
                race_max_length = checkpoint['position_embedding.weight'].shape[0]
            else:
                race_max_length = max_length
            race_classifier = SimpleTransformerClassifier(vocab_size=race_vocab_size, max_length=race_max_length)
            race_classifier.load_state_dict(checkpoint)
            print(f"Loaded race classifier with vocab_size={race_vocab_size}, max_length={race_max_length}")
        else:
            print(f"Warning: Race classifier model not found at {race_model_path}. Using untrained model.")
            race_classifier = SimpleTransformerClassifier(vocab_size=vocab_size, max_length=max_length)
        race_classifier.to(device)
        race_classifier.eval()
        
        # Load gender token bias
        if os.path.exists(gender_importance_path):
            gender_token_bias = load_and_normalize_beta(gender_importance_path)
            if not gender_token_bias:
                print(f"Warning: Gender importance file {gender_importance_path} exists but is empty. Using empty token_bias.")
        else:
            print(f"Warning: Gender importance file {gender_importance_path} not found. Using empty token_bias.")
            gender_token_bias = {}
        
        # Hardcode gender token bias: 1.0 for gender tokens, 0.0 for others
        if tokenizer is not None:
            print("Hardcoding gender token bias...")
            gender_token_bias = hardcode_gender_token_bias(tokenizer, gender_words, model_name_suffix)
        
        # Load race token bias
        if os.path.exists(race_importance_path):
            race_token_bias = load_and_normalize_beta(race_importance_path)
            if not race_token_bias:
                print(f"Warning: Race importance file {race_importance_path} exists but is empty. Using empty token_bias.")
        else:
            print(f"Warning: Race importance file {race_importance_path} not found. Using empty token_bias.")
            race_token_bias = {}
        
        # Hardcode race token bias: 1.0 for race tokens, 0.0 for others
        if tokenizer is not None:
            print("Hardcoding race token bias for intersection mode...")
            race_token_bias = hardcode_race_token_bias(tokenizer, race_words, model_name_suffix)
        
        # For backward compatibility, set text_classifier and token_bias to gender (will be overridden in intersection mode)
        text_classifier = gender_classifier
        token_bias = gender_token_bias
    else:
        # Single target (gender or race)
        if args.debiasing_target == 'gender':
            model_path = os.path.join(NLP_CLASSIFICATION_DIR, f"gender_model_{model_name_suffix}_pytorch_generated", "pytorch_model.bin")
            importance_path = os.path.join(NLP_CLASSIFICATION_DIR, f"importance_dict_{model_name_suffix}_pytorch_generated.json")
        else:  # race
            # For race, we'll use a similar structure but with race-specific models
            model_path = os.path.join(NLP_CLASSIFICATION_DIR, f"race_model_{model_name_suffix}_pytorch_generated", "pytorch_model.bin")
            importance_path = os.path.join(NLP_CLASSIFICATION_DIR, f"importance_race_dict_{model_name_suffix}_pytorch_generated.json")
        
        if os.path.exists(model_path):
            # First inspect checkpoint to get correct parameters
            checkpoint = torch.load(model_path, map_location='cpu')
            # Infer vocab_size and max_length from checkpoint
            if 'token_embedding.weight' in checkpoint:
                checkpoint_vocab_size = checkpoint['token_embedding.weight'].shape[0]
            else:
                checkpoint_vocab_size = vocab_size
            if 'position_embedding.weight' in checkpoint:
                checkpoint_max_length = checkpoint['position_embedding.weight'].shape[0]
            else:
                checkpoint_max_length = max_length
            
            # Recreate model with correct parameters
            if checkpoint_vocab_size != vocab_size or checkpoint_max_length != max_length:
                print(f"Model parameters mismatch. Creating model with vocab_size={checkpoint_vocab_size}, max_length={checkpoint_max_length}")
                text_classifier = SimpleTransformerClassifier(vocab_size=checkpoint_vocab_size, max_length=checkpoint_max_length)
            
            text_classifier.load_state_dict(checkpoint)
        else:
            print(f"Warning: Classifier model not found at {model_path}. Using untrained model.")
        
        text_classifier.to(device)
        text_classifier.eval()
        
        if os.path.exists(importance_path):
            token_bias = load_and_normalize_beta(importance_path)
            if not token_bias:
                print(f"Warning: Importance file {importance_path} exists but is empty or has no valid data. Using empty token_bias.")
        else:
            print(f"Warning: Importance file {importance_path} not found. Using empty token_bias.")
            token_bias = {}
        
        # Hardcode token bias: 1.0 for target tokens, 0.0 for others
        if args.debiasing_target == 'gender' and tokenizer is not None:
            print("Hardcoding gender token bias...")
            token_bias = hardcode_gender_token_bias(tokenizer, gender_words, model_name_suffix)
        elif args.debiasing_target == 'race' and tokenizer is not None:
            print("Hardcoding race token bias...")
            token_bias = hardcode_race_token_bias(tokenizer, race_words, model_name_suffix)
    
    # Only load/train image classifier if needed (not needed for race debiasing in logit mode)
    # But we still need it for prompt mode and gender debiasing
    if args.debiasing_target == 'race' and args.mode == 'logit':
        # For race debiasing in logit mode, we use s_scale = -1 directly, so no classifier needed
        predict_attribute = None
        classifier = None
        predict_attribute_gender = None
        predict_attribute_race = None
        gender_classifier_img = None
        race_classifier_img = None
    elif args.debiasing_target == 'intersection':
        # For intersection mode, we need both gender and race image classifiers
        try :
            embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train.pt'))
        except:
            embedding = torch.load(f'../../vqa_bias/facet_open/embedding/fairface_{model_name_suffix}_train.pt')
        X_train = embedding['image_embeddings']
        
        # Load gender classifier
        gender_classifier_img = LogisticRegression(max_iter=1000, solver='lbfgs')
        y_train_gender = embedding['sensitive_attributes'][:, 1]  # gender index
        if isinstance(X_train, torch.Tensor):
            X_train_np = X_train.float().detach().cpu().numpy()
        else:
            X_train_np = X_train
        if isinstance(y_train_gender, torch.Tensor):
            y_train_gender_np = y_train_gender.float().detach().cpu().numpy()
        else:
            y_train_gender_np = y_train_gender
        
        gender_classifier_path = f"image_gender_classifier_{model_name_suffix}.pkl"
        try:
            gender_classifier_img = load(gender_classifier_path)
        except:
            gender_classifier_img.fit(X_train_np, y_train_gender_np)
            dump(gender_classifier_img, gender_classifier_path)
            gender_classifier_img = load(gender_classifier_path)
        
        # Load race classifier
        race_classifier_img = LogisticRegression(max_iter=1000, solver='lbfgs')
        y_train_race = embedding['sensitive_attributes'][:, 2]  # race index
        if isinstance(y_train_race, torch.Tensor):
            y_train_race_np = y_train_race.float().detach().cpu().numpy()
        else:
            y_train_race_np = y_train_race
        
        race_classifier_path = f"image_race_classifier_{model_name_suffix}.pkl"
        try:
            race_classifier_img = load(race_classifier_path)
        except:
            race_classifier_img.fit(X_train_np, y_train_race_np)
            dump(race_classifier_img, race_classifier_path)
            race_classifier_img = load(race_classifier_path)
        
        def predict_attribute_gender(image_embedding):
            """ Predict gender from image embedding using the trained classifier. """
            image_embedding = image_embedding.reshape(1, -1)
            prob = gender_classifier_img.predict_proba(image_embedding)[:, 1]
            return 2 * prob[0] - 1  # Normalize to range [-1, 1]
        
        def predict_attribute_race(image_embedding):
            """ Predict race from image embedding using the trained classifier. """
            image_embedding = image_embedding.reshape(1, -1)
            prob = race_classifier_img.predict_proba(image_embedding)[:, 1]
            return 2 * prob[0] - 1  # Normalize to range [-1, 1]
        
        # For backward compatibility
        predict_attribute = predict_attribute_gender
        classifier = gender_classifier_img
    else:
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs')
        try:
            embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train.pt'))
        except Exception:
            embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train.pt'))

        X_train = embedding['image_embeddings']
        y_train = embedding['sensitive_attributes'][:,sens_idx]
        if isinstance(X_train, torch.Tensor):
            X_train_np = X_train.float().detach().cpu().numpy()
        else:
            X_train_np = X_train
        if isinstance(y_train, torch.Tensor):
            y_train_np = y_train.float().detach().cpu().numpy()
        else:
            y_train_np = y_train
        
        classifier_path = f"image_{args.debiasing_target}_classifier_{model_name_suffix}.pkl"
        try:
            classifier = load(classifier_path)
        except:
            classifier.fit(X_train_np, y_train_np)
            dump(classifier, classifier_path)
            classifier = load(classifier_path)
        
        def predict_attribute(image_embedding):
            """ Predict attribute (gender or race) from image embedding using the trained classifier. """
            image_embedding = image_embedding.reshape(1, -1)
            prob = classifier.predict_proba(image_embedding)[:, 1]
            return 2 * prob[0] - 1  # Normalize to range [-1, 1]
        
        predict_attribute_gender = None
        predict_attribute_race = None
        gender_classifier_img = None
        race_classifier_img = None

if args.mode=='prompt':
    classifier = LogisticRegression(max_iter=1000, solver='lbfgs')
    embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train.pt'))

    X_train = embedding['image_embeddings']
    y_train = embedding['sensitive_attributes'][:,sens_idx]
    if isinstance(X_train, torch.Tensor):
        X_train_np = X_train.float().detach().cpu().numpy()
    else:
        X_train_np = X_train
    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.float().detach().cpu().numpy()
    else:
        y_train_np = y_train
    
    classifier_path = f"image_{args.debiasing_target}_classifier_{model_name_suffix}.pkl"
    try:
        classifier = load(classifier_path)
    except:
        classifier.fit(X_train_np, y_train_np)
        dump(classifier, classifier_path)
        classifier = load(classifier_path)
    
    def predict_attribute_for_prompt(image_embedding):
        """ Predict attribute from image embedding for prompt mode. """
        image_embedding = image_embedding.reshape(1, -1)
        prob = classifier.predict_proba(image_embedding)[:, 1]
        if args.debiasing_target == 'gender':
            return "Male" if prob[0] > 0.5 else "Female"
        else:  # race
            # Map race indices to labels (adjust based on your data)
            race_labels = ['White', 'Black', 'Indian', 'Asian', 'Middle Eastern', 'Latino']
            return race_labels[int(prob[0] * len(race_labels))] if prob[0] < 1.0 else race_labels[-1]

if args.mode=='steer':
    alpha_tokenizer = None
    if args.model == 'llava':
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    elif args.model == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

if args.mode=='sfid':
    print("Debias Text Decoder")
    tokenizer = None
    alpha_tokenizer = None
    sfid_seed = 0
    import numpy as np
    import random
    from sklearn.ensemble import RandomForestClassifier
    random.seed(sfid_seed)
    torch.manual_seed(sfid_seed)
    torch.cuda.manual_seed(sfid_seed)
    np.random.seed(sfid_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    os.environ['PYTHONHASHSEED'] = str(sfid_seed)
    embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train_decoder.pt'))
    embedding_val = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_val_decoder.pt'))

    X_train = embedding['decode_embeddings']
    y_train = embedding['sensitive_attributes'][:,sens_idx]
    X_test = embedding_val['decode_embeddings']
    y_test = embedding_val['sensitive_attributes'][:,sens_idx]
    
    text_model_path = f'checkpoint/{model_name_suffix}_text_decoder_random_forest_model_{args.debiasing_target}.joblib'
    if os.path.exists(text_model_path):
        dec_clf = load(text_model_path)
        print("Load pretrained Random Forest.")
    else : 
        dec_clf = RandomForestClassifier(n_estimators=100)
        dec_clf.fit(X_train.float().detach().cpu().numpy(), y_train.float().detach().cpu().numpy())
        dump(dec_clf,text_model_path)
    probabilities = dec_clf.predict_proba(X_test.float().detach().cpu().numpy())
    max_probabilities = probabilities.max(axis=1)
    low_confidence_samples = X_test[max_probabilities < args.t]    
    decoder_mean_features_lowconfidence = torch.mean(torch.tensor(low_confidence_samples).float(),axis=0)
    decoder_importances = dec_clf.feature_importances_
    embedding_dim = X_test.shape[1]

if args.mode=='clipclip':
    from sklearn.feature_selection import mutual_info_classif
    print("Debias Text Decoder")
    tokenizer = None
    alpha_tokenizer = None
    sfid_seed = 0
    import numpy as np
    import random
    from sklearn.ensemble import RandomForestClassifier
    random.seed(sfid_seed)
    torch.manual_seed(sfid_seed)
    torch.cuda.manual_seed(sfid_seed)
    np.random.seed(sfid_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    os.environ['PYTHONHASHSEED'] = str(sfid_seed)
    embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train_decoder.pt'))
    embedding_val = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_val_decoder.pt'))

    X_train = embedding['decode_embeddings']
    y_train = embedding['sensitive_attributes'][:,sens_idx]
    X_test = embedding_val['decode_embeddings']
    y_test = embedding_val['sensitive_attributes'][:,sens_idx]
    prune_indices_path = os.path.join(_EMBEDDING_DIR, f'pruned_feature_indices_{model_name_suffix}_{args.debiasing_target}.npy')

    try:
        decoder_importances = np.load(prune_indices_path)
        print(f"Loaded precomputed pruned feature indices from {prune_indices_path}")
    except FileNotFoundError:
        print("Pruned feature indices not found. Computing mutual information...")
        mutual_info = mutual_info_classif(X_train.float().detach().cpu().numpy(), y_train.float().detach().cpu().numpy(), discrete_features=False)
        K = 200
        decoder_importances = np.argsort(mutual_info)[:K]
        np.save(prune_indices_path, decoder_importances)
        print(f"Computed and saved pruned feature indices at {prune_indices_path}")

    X_train[:, decoder_importances] = 0
    X_test[:, decoder_importances] = 0
    decoder_mean_features_lowconfidence = torch.zeros(X_train.shape[1])
    embedding_dim = X_test.shape[1]

results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}.csv"
if args.mode=='logit':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}_{args.neutral}_{args.lam}.csv"
elif args.mode=='sfid':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}_{args.decoder_prune_num}_{args.t}.csv"
elif args.mode=='clipclip':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}_{args.decoder_prune_num}.csv"
elif args.mode=='steer':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}.csv"
elif args.mode=='prompt':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}.csv"
elif args.mode=='vdd':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}.csv"
    tokenizer=None
    alpha_tokenizer=None
elif args.mode=='dear':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}.csv"
    import torch.nn as nn

    tokenizer=None
    alpha_tokenizer=None
    class Adaptor(nn.Module):
        def __init__(self, input_dim):
            super(Adaptor, self).__init__()
            self.fc1 = nn.Linear(input_dim, input_dim)
            self.fc2 = nn.Linear(input_dim, input_dim)
            self.activation = nn.ReLU()

        def forward(self, x):
            residual = self.fc1(x)
            residual = self.activation(residual)
            residual = self.fc2(residual)
            return x + residual
    
    # Determine input_dim based on model
    if args.model == 'llava':
        input_dim = 1024
    elif args.model == 'paligemma':
        input_dim = 1152
    elif args.model == 'qwen':
        input_dim = 1024  # Adjust based on Qwen2-VL architecture
    
    adaptor = Adaptor(input_dim).to(device)
    adaptor_path = f"adaptor_model_l_transform_{model_name_suffix}_{args.debiasing_target}.pth"
    if os.path.exists(adaptor_path):
        adaptor.load_state_dict(torch.load(adaptor_path))
    adaptor.eval()

os.makedirs("result", exist_ok=True)

if os.path.exists(results_file):
    print("Result Exist")
else:
    # Load model and processor based on model type
    if args.model=='llava':
        model = CustomLlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True
        )
        # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
        # # Load processor - try without revision first, then with revision if needed
        # try:
        #     processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        # except:
        #     processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
        
        # Ensure processor has patch_size set (for LLaVA 1.5, patch_size should be 14)
        # The patch_size is typically in the image_processor
        if hasattr(processor, 'image_processor'):
            if not hasattr(processor.image_processor, 'patch_size') or processor.image_processor.patch_size is None:
                # Set patch_size to 14 for LLaVA 1.5 (CLIP ViT-L/14 uses patch_size 14)
                processor.image_processor.patch_size = 14
        # Also check if processor itself has patch_size
        if hasattr(processor, 'patch_size') and processor.patch_size is None:
            processor.patch_size = 14
        model = model.to(device)
    elif args.model=='paligemma':
        model_id = "google/paligemma-3b-mix-224"
        model = create_custom_paligemma_model(model_id, device)
        processor = PaliGemmaProcessor.from_pretrained(model_id)
        torch.cuda.empty_cache()
        model.eval()
        if args.mode == 'logit':
            tokenizer = processor.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            alpha_tokenizer = copy.deepcopy(tokenizer)
            if alpha_tokenizer.pad_token is None:
                alpha_tokenizer.pad_token = alpha_tokenizer.eos_token
            vocab_size = tokenizer.vocab_size
    elif args.model=='qwen':
        # Load base model first, then create custom model and load state dict
        # Try to load Qwen2.5-VL-7B-Instruct using Qwen2VLForConditionalGeneration (they share architecture)
        from qwen_model import Qwen2_5_VLForConditionalGeneration  # Use alias from qwen_model
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        try:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
            # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

        except Exception as e:
            print(f"Warning: Failed to load {model_name}: {e}")
            print("Falling back to Qwen2-VL-3B-Instruct")
            from transformers import Qwen2VLForConditionalGeneration
            model_name = "Qwen/Qwen2-VL-3B-Instruct"
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
        # Create custom model with same config
        model = CustomQwenForConditionalGeneration(base_model.config)
        model.load_state_dict(base_model.state_dict(), strict=False)
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        
    # Function to get VQA answer
    def get_vqa_answer(image_path,lam=None,mode=None,tokenizer=None, alpha_tokenizer=None):
        image = Image.open(image_path).convert("RGB")

        if args.model=='llava':
            prompt = f"USER: <image>\nDescribe the photo in detail. ASSISTANT:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)  # Move inputs to device
            with torch.no_grad():
                if args.mode=='steer':
                    dummy_prompt = f"USER: <image>\nDescribe the {args.debiasing_target} of the person in the photo. ASSISTANT:"
                    dummy_input = processor(images=image, text=dummy_prompt, return_tensors="pt").to(device)
                    a_vector = model.steering(**dummy_input, max_new_tokens=50,mode=args.mode,vqa_tokenizer=tokenizer)
                if args.mode=='logit':
                    if args.debiasing_target == 'intersection':
                        # For intersection mode, compute both gender and race s_scales
                        prefix = model.vision_tower(inputs['pixel_values'].squeeze(1), output_hidden_states = True)
                        prefix = prefix.pooler_output
                        s_scale_gender = predict_attribute_gender(prefix.detach().cpu().float().numpy())
                        s_scale_race = -1
                        generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='logit',
                                                gender_text_classifier=gender_classifier,
                                                race_text_classifier=race_classifier,
                                                gender_token_bias=gender_token_bias,
                                                race_token_bias=race_token_bias,
                                                s_scale_gender=s_scale_gender,
                                                s_scale_race=s_scale_race,
                                                lam=lam, lam_gender=args.lam_gender, lam_race=args.lam_race,
                                                neutral=args.neutral, 
                                                vqa_tokenizer=tokenizer, alpha_tokenizer=alpha_tokenizer, 
                                                device=device, threshold=args.t)
                    elif args.debiasing_target == 'race':
                        # For race debiasing, set s_scale to -1 as specified
                        s_scale = -1
                        generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='logit',
                                                s_scale=s_scale, text_classifier=text_classifier,
                                                token_bias=token_bias, lam=lam, neutral=args.neutral, 
                                                vqa_tokenizer=tokenizer, alpha_tokenizer=alpha_tokenizer, 
                                                device=device, threshold=args.t)
                    else:
                        # For gender debiasing, use predicted attribute from image
                        prefix = model.vision_tower(inputs['pixel_values'].squeeze(1), output_hidden_states = True)
                        prefix = prefix.pooler_output
                        s_scale = predict_attribute(prefix.detach().cpu().float().numpy())
                        generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='logit',
                                                s_scale=s_scale, text_classifier=text_classifier,
                                                token_bias=token_bias, lam=lam, neutral=args.neutral, 
                                                vqa_tokenizer=tokenizer, alpha_tokenizer=alpha_tokenizer, 
                                                device=device, threshold=args.t)
                elif args.mode in ['sfid','clipclip']:
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode=mode,
                                            decoder_mean_features_lowconfidence=decoder_mean_features_lowconfidence,
                                            decoder_important_indices=decoder_importances, device=device,
                                            vqa_name='llava')
                elif args.mode=='dear':
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='dear', 
                                            dear_adaptor=adaptor)    
                elif args.mode=='steer':
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='steer',
                                            dummy_input=dummy_input, vqa_tokenizer=tokenizer)    
                elif args.mode=='vdd':
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='vdd', lam=args.lam)
                elif args.mode=='prompt':
                    # Get image embedding and predict gender
                    prefix = model.vision_tower(inputs['pixel_values'].squeeze(1), output_hidden_states = True)
                    prefix = prefix.pooler_output
                    predicted_attr = predict_attribute_for_prompt(prefix.detach().cpu().numpy())
                    
                    # Modify the prompt to include gender information
                    prompt = f"USER: <image>\nThe {args.debiasing_target} of the person in the image is {predicted_attr}. Describe the photo in detail. ASSISTANT:"
                    print(prompt)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='naive')
                else:
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='naive')
                    
            answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer = answer.strip()
            answer = answer.split("ASSISTANT:")[-1]
            
        elif args.model=='paligemma':
            prompt = "Describe the photo in detail."
            if args.mode == 'prompt':
                # Will be modified below
                pass
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(device)
            input_len = model_inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                if args.mode=='logit':
                    
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50,do_sample=False,output_hidden_states=False,
                                                s_scale=s_scale,text_classifier=text_classifier,
                                                token_bias=token_bias,lam=args.lam,neutral=args.neutral,
                                                vqa_tokenizer=tokenizer,alpha_tokenizer=alpha_tokenizer, device=device,threshold=args.t,
                                                mode=args.mode,vqa_name = 'paligemma')
                elif args.mode in ['sfid','clipclip']:
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50,do_sample=False,output_hidden_states=False,
                                                decoder_mean_features_lowconfidence=decoder_mean_features_lowconfidence,
                                                decoder_important_indices=decoder_importances, device=device
                                                ,mode=args.mode,vqa_name = 'paligemma')
                elif args.mode=='dear':
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False,output_hidden_states=False,
                                                mode=args.mode,dear_adaptor = adaptor)
                elif args.mode=='vdd':
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50,mode=args.mode,vqa_name='paligemma')
                elif args.mode=='prompt':
                    # Get vision features and predict attribute
                    vision_outputs = model.vision_tower(model_inputs['pixel_values'].squeeze(1), output_hidden_states=True)
                    vision_features = vision_outputs.last_hidden_state.mean(dim=1)
                    predicted_attr = predict_attribute_for_prompt(vision_features.detach().cpu().numpy())
                    prompt = f"The {args.debiasing_target} of the person in the image is {predicted_attr}. Describe the photo in detail."
                    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(device)
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False,output_hidden_states=False)
                else:
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False,output_hidden_states=False)
            
            generation = generate_ids[0][input_len:]
            answer = processor.decode(generation, skip_special_tokens=True)
            answer = answer.strip()
            
        elif args.model=='qwen':
            prompt_text = "Describe the photo in detail."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs, video_inputs = processor.process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                if args.mode=='logit':
                    if args.debiasing_target == 'intersection':
                        # For intersection mode, compute both gender and race s_scales
                        if hasattr(model, 'visual') and model.visual is not None:
                            # Handle Qwen vision model which may require grid_thw
                            if 'grid_thw' in inputs:
                                vision_outputs = model.visual(
                                    pixel_values=inputs['pixel_values'],
                                    grid_thw=inputs['grid_thw']
                                )
                            else:
                                # Try without grid_thw (for older QWEN versions or if not needed)
                                try:
                                    vision_outputs = model.visual(inputs['pixel_values'])
                                except TypeError:
                                    # If that fails, use model forward to extract features
                                    outputs = model(**inputs, output_hidden_states=True)
                                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                        vision_outputs = outputs.hidden_states[0]
                                    else:
                                        vision_outputs = None
                            
                            if vision_outputs is not None:
                                if hasattr(vision_outputs, 'last_hidden_state'):
                                    prefix = vision_outputs.last_hidden_state.mean(dim=1)
                                else:
                                    prefix = vision_outputs.mean(dim=1)
                            else:
                                prefix = torch.zeros(1, 1024).to(device)
                        else:
                            prefix = torch.zeros(1, 1024).to(device)
                        # Convert to float32 before numpy conversion (BFloat16 not supported by numpy)
                        s_scale_gender = predict_attribute_gender(prefix.detach().cpu().float().numpy())
                        s_scale_race = -1
                        generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='logit',
                                                gender_text_classifier=gender_classifier,
                                                race_text_classifier=race_classifier,
                                                gender_token_bias=gender_token_bias,
                                                race_token_bias=race_token_bias,
                                                s_scale_gender=s_scale_gender,
                                                s_scale_race=s_scale_race,
                                                lam=lam, lam_gender=args.lam_gender, lam_race=args.lam_race,
                                                neutral=args.neutral, 
                                                vqa_tokenizer=tokenizer, alpha_tokenizer=alpha_tokenizer, 
                                                device=device, threshold=args.t)
                    elif args.debiasing_target == 'race':
                        # For race debiasing, set s_scale to -1 as specified
                        s_scale = -1
                        generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='logit',
                                            s_scale=s_scale, text_classifier=text_classifier,
                                            token_bias=token_bias, lam=lam, neutral=args.neutral, 
                                            vqa_tokenizer=tokenizer, alpha_tokenizer=alpha_tokenizer, 
                                            device=device, threshold=args.t)
                    else:
                        # For gender debiasing, use predicted attribute from image
                        if hasattr(model, 'visual') and model.visual is not None:
                            # Handle Qwen vision model which may require grid_thw
                            if 'grid_thw' in inputs:
                                vision_outputs = model.visual(
                                    pixel_values=inputs['pixel_values'],
                                    grid_thw=inputs['grid_thw']
                                )
                            else:
                                # Try without grid_thw (for older QWEN versions or if not needed)
                                try:
                                    vision_outputs = model.visual(inputs['pixel_values'])
                                except TypeError:
                                    # If that fails, use model forward to extract features
                                    outputs = model(**inputs, output_hidden_states=True)
                                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                        vision_outputs = outputs.hidden_states[0]
                                    else:
                                        vision_outputs = None
                            
                            if vision_outputs is not None:
                                if hasattr(vision_outputs, 'last_hidden_state'):
                                    prefix = vision_outputs.last_hidden_state.mean(dim=1)
                                else:
                                    prefix = vision_outputs.mean(dim=1)
                            else:
                                prefix = torch.zeros(1, 1024).to(device)
                        else:
                            prefix = torch.zeros(1, 1024).to(device)
                        # Convert to float32 before numpy conversion (BFloat16 not supported by numpy)
                        s_scale = predict_attribute(prefix.detach().cpu().float().numpy())
                        generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='logit',
                                            s_scale=s_scale, text_classifier=text_classifier,
                                            token_bias=token_bias, lam=lam, neutral=args.neutral, 
                                            vqa_tokenizer=tokenizer, alpha_tokenizer=alpha_tokenizer, 
                                            device=device, threshold=args.t)
                elif args.mode in ['sfid','clipclip']:
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode=mode,
                                            decoder_mean_features_lowconfidence=decoder_mean_features_lowconfidence,
                                            decoder_important_indices=decoder_importances, device=device,
                                            vqa_name='qwen')
                elif args.mode=='dear':
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='dear', 
                                            dear_adaptor=adaptor)    
                elif args.mode=='vdd':
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='vdd', lam=args.lam)
                elif args.mode=='prompt':
                    if hasattr(model, 'visual') and model.visual is not None:
                        # Handle Qwen vision model which may require grid_thw
                        if 'grid_thw' in inputs:
                            vision_outputs = model.visual(
                                pixel_values=inputs['pixel_values'],
                                grid_thw=inputs['grid_thw']
                            )
                        else:
                            # Try without grid_thw (for older QWEN versions or if not needed)
                            try:
                                vision_outputs = model.visual(inputs['pixel_values'])
                            except TypeError:
                                # If that fails, use model forward to extract features
                                outputs = model(**inputs, output_hidden_states=True)
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                    vision_outputs = outputs.hidden_states[0]
                                else:
                                    vision_outputs = None
                        
                        if vision_outputs is not None:
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                prefix = vision_outputs.last_hidden_state.mean(dim=1)
                            else:
                                prefix = vision_outputs.mean(dim=1)
                        else:
                            prefix = torch.zeros(1, 1024).to(device)
                    else:
                        prefix = torch.zeros(1, 1024).to(device)
                    # Convert to float32 before numpy conversion (BFloat16 not supported by numpy)
                    predicted_attr = predict_attribute_for_prompt(prefix.detach().cpu().float().numpy())
                    prompt_text = f"The {args.debiasing_target} of the person in the image is {predicted_attr}. Describe the photo in detail."
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": prompt_text}
                            ]
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if process_vision_info is not None:
                        image_inputs, video_inputs = process_vision_info(messages)
                    else:
                        image_inputs, video_inputs = processor.process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(device)
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='naive')
                else:
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='naive')
                    
            # Extract only the generated tokens (not the input prompt)
            # For QWEN, input_ids length should match the tokenized prompt length
            input_len = inputs['input_ids'].shape[-1]
            
            # Debug: print token info to understand what's being generated
            # print(f"Input length: {input_len}, Generated length: {len(generate_ids[0])}")
            # print(f"Generated tokens: {generate_ids[0][:input_len+10] if len(generate_ids[0]) > input_len else generate_ids[0]}")
            
            generated_ids = generate_ids[0][input_len:] if len(generate_ids[0]) > input_len else generate_ids[0]
            
            # Decode using tokenizer (processor.tokenizer) for more reliable decoding
            if hasattr(processor, 'tokenizer'):
                answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                answer = processor.decode(generated_ids, skip_special_tokens=True)
            
            answer = answer.strip()
            
            # Clean up the answer - remove any remaining prompt artifacts
            if "assistant" in answer.lower():
                answer = answer.split("assistant")[-1].strip()
            if ":" in answer:
                answer = answer.split(":")[-1].strip()
        
        return answer
        
    # Load reference data for debug mode
    reference_data = {}
    if args.debug:
        if os.path.exists(args.reference_file):
            print(f"Loading reference data from {args.reference_file}...")
            reference_df = pd.read_csv(args.reference_file)
            # Create mapping from image_path to reference row
            for _, ref_row in reference_df.iterrows():
                ref_image_path = ref_row['image_path']
                reference_data[ref_image_path] = {
                    'answer': ref_row['answer'],
                    'detected_gender': ref_row['detected_gender']
                }
            print(f"Loaded {len(reference_data)} reference entries")
        else:
            print(f"Warning: Reference file {args.reference_file} not found. Debug mode disabled.")
            args.debug = False
    
    # Analyze VQA results by demographic group
    results = []
    skipped_count = 0
    processed_count = 0
    
    # Filter annotations if --filter_image is specified
    if args.filter_image:
        new_annotations = new_annotations[new_annotations['filename'] == args.filter_image]
        if len(new_annotations) == 0:
            print(f"Error: Image '{args.filter_image}' not found in annotations.")
            exit(1)
        print(f"Filtered to process only: {args.filter_image}")
    
    for index, row in tqdm(new_annotations.iterrows(), total=len(new_annotations)):
        image_path = os.path.join(_FACET_DIR, 'image', row['filename'])
        
        occupation = row['class1']
        gender = row['gender']
        if int(gender)==0:
            gender='Male'
        elif int(gender)==1:
            gender='Female'
        race = row['race']
        
        # Debug mode: check reference detected_gender
        if args.debug and image_path in reference_data:
            ref_detected_gender = reference_data[image_path]['detected_gender']
            # Check if detected_gender is Male or Female
            if ref_detected_gender not in ['Male', 'Female']:
                # Use reference answer, skip inference
                answer = reference_data[image_path]['answer']
                detected_gender = ref_detected_gender
                skipped_count += 1
                if skipped_count % 100 == 0:  # Print every 100 skipped to reduce output
                    print(f"[DEBUG] Skipped {skipped_count} images (using reference answers)")
            else:
                # Run inference for Male/Female cases
                answer = get_vqa_answer(image_path,args.lam,args.mode,tokenizer, alpha_tokenizer)
                detected_gender = decide_gender(nltk.word_tokenize(answer))
                processed_count += 1
        elif args.debug:
            # Image not in reference file, run inference normally
            print(f"[DEBUG] Warning: {row['filename']} not found in reference file, running inference")
            answer = get_vqa_answer(image_path,args.lam,args.mode,tokenizer, alpha_tokenizer)
            detected_gender = decide_gender(nltk.word_tokenize(answer))
            processed_count += 1
        else:
            # Normal mode: always run inference
            answer = get_vqa_answer(image_path,args.lam,args.mode,tokenizer, alpha_tokenizer)
            detected_gender = decide_gender(nltk.word_tokenize(answer))
            processed_count += 1
        
        print(answer)
        results.append({'image_path':image_path,'ground_truth_gender': gender, 'race': race, 'answer': answer, 'occupation': occupation,'detected_gender': detected_gender})
    
    if args.debug:
        print(f"\n[DEBUG SUMMARY] Processed: {processed_count}, Skipped (used reference): {skipped_count}, Total: {len(results)}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
# evaluate_facet_open(results_file)
