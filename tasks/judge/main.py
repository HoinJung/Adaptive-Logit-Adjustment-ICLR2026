import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen', type=str, choices=['llava', 'paligemma', 'qwen'], help='Model to use: llava, paligemma, or qwen')
parser.add_argument('--mode', default='logit', type=str)
parser.add_argument('--gpu_id', default='3', type=str)
parser.add_argument('--lam', default=0.9, type=float)
parser.add_argument('--decoder_prune_num', default=50, type=int)
parser.add_argument('--t', default=0.1, type=float)
parser.add_argument('--debiasing_target', default='gender', type=str, choices=['gender', 'race'], help='Target attribute for debiasing: gender or race')
parser.add_argument(
    '-n',
    '--neutral',
    action='store_true',
    help='Description of your feature.',
)
import os
import sys
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from config import DATA_DIR, NLP_CLASSIFICATION_DIR
_FACET_DIR = os.path.join(DATA_DIR, "facet")
_EMBEDDING_DIR = os.path.join(_REPO_ROOT, "tasks", "judge", "embedding")
import torch
from transformers import AutoProcessor, AutoTokenizer
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

from ala.utils import decide_gender, evaluate_facet_open, load_and_normalize_beta
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # Fallback: if qwen_vl_utils is not available, use processor method
    process_vision_info = None
import nltk
from ala.model import SimpleTransformerClassifier
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
else:
    sens_idx = 1  # default to gender

# Determine model name suffix
if args.model == 'llava':
    model_name_suffix = 'llava'
elif args.model == 'paligemma':
    model_name_suffix = 'paligemma'
elif args.model == 'qwen':
    model_name_suffix = 'qwen'

if args.mode=='logit':
    # Load tokenizer based on model
    if args.model == 'llava':
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    elif args.model == 'paligemma':
        # Will be loaded with processor
        tokenizer = None
    elif args.model == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    if tokenizer is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        alpha_tokenizer = copy.deepcopy(tokenizer)
        if alpha_tokenizer.pad_token is None:
            alpha_tokenizer.pad_token = alpha_tokenizer.eos_token
    
    # Get vocab size
    if args.model == 'llava':
        vocab = alpha_tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1
    elif args.model == 'paligemma':
        vocab_size = None  # Will be set after processor is loaded
    elif args.model == 'qwen':
        vocab = alpha_tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1
    
    max_length = 50
    
    # For race, we use hard-coded token mapping instead of text_classifier and token_bias
    if args.debiasing_target == 'race':
        # Don't load text_classifier and token_bias for race (use hard-coding instead)
        text_classifier = None
        token_bias = None
        
        # Use multi-class classifier for race
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
        try:
            embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train.pt'))
        except:
            embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train.pt'))


        X_train = embedding['image_embeddings']
        y_train = embedding['sensitive_attributes'][:,sens_idx]
        
        classifier_path = f"image_{args.debiasing_target}_classifier_{model_name_suffix}.pkl"
        try:
            classifier = load(classifier_path)
        except:
            classifier.fit(X_train, y_train)
            dump(classifier, classifier_path)
            classifier = load(classifier_path)
        
        # Race mapping: {'East Asian': 0, 'Indian': 1, 'Black': 2, 'White': 3, 'Middle Eastern': 4, 'Latino_Hispanic': 5, 'Southeast Asian': 6}
        race_map = {'East Asian': 0, 'Indian': 1, 'Black': 2, 'White': 3, 'Middle Eastern': 4, 'Latino_Hispanic': 5, 'Southeast Asian': 6}
        race_map_reverse = {v: k for k, v in race_map.items()}
        
        def predict_attribute(image_embedding):
            """ Predict race probabilities from image embedding using multi-class classifier. """
            image_embedding = image_embedding.reshape(1, -1)
            # Get probabilities for all 7 classes
            probs = classifier.predict_proba(image_embedding)[0]  # Shape: (7,)
            return probs
    else:
        # For gender, use binary classifier with text_classifier and token_bias
        text_classifier = SimpleTransformerClassifier(vocab_size=vocab_size, max_length=max_length)
        
        # Load gender classifier
        model_path = os.path.join(NLP_CLASSIFICATION_DIR, f"gender_model_{model_name_suffix}_pytorch_generated", "pytorch_model.bin")
        importance_path = os.path.join(NLP_CLASSIFICATION_DIR, f"importance_dict_{model_name_suffix}_pytorch_generated.json")
        
        if os.path.exists(model_path):
            text_classifier.load_state_dict(torch.load(model_path, weights_only=False))
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        text_classifier.to(device)
        text_classifier.eval()
        
        token_bias = load_and_normalize_beta(importance_path) if os.path.exists(importance_path) else {}
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs')
        embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_{model_name_suffix}_train.pt'))

        X_train = embedding['image_embeddings']
        y_train = embedding['sensitive_attributes'][:,sens_idx]
        
        classifier_path = f"image_{args.debiasing_target}_classifier_{model_name_suffix}.pkl"
        try:
            classifier = load(classifier_path)
        except:
            classifier.fit(X_train, y_train)
            dump(classifier, classifier_path)
            classifier = load(classifier_path)
        
        def predict_attribute(image_embedding):
            """ Predict attribute (gender) from image embedding using the trained classifier. """
            image_embedding = image_embedding.reshape(1, -1)
            prob = classifier.predict_proba(image_embedding)[:, 1]
            return 2 * prob[0] - 1

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

if args.mode=='sfidba':
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

    high_confidence_mask = max_probabilities >= args.t
    high_confidence_samples = X_test[high_confidence_mask]
    high_confidence_labels = y_test[high_confidence_mask]

    unique_groups = torch.unique(y_test)
    decoder_mean_features_highconfidence = {}

    for group in unique_groups:
        group_samples = high_confidence_samples[high_confidence_labels == group]
        if len(group_samples) > 0:
            decoder_mean_features_highconfidence[int(group)] = torch.mean(group_samples.float(), axis=0)

    decoder_importances = dec_clf.feature_importances_
    embedding_dim = X_test.shape[1]
    classifier = LogisticRegression(max_iter=1000, solver='lbfgs')
    embedding = torch.load(os.path.join(_EMBEDDING_DIR, f'fairface_qwen_train.pt'))

    X_train = embedding['image_embeddings']
    y_train = embedding['sensitive_attributes'][:,1]
    
    try:
        classifier = load("image_gender_classifier_qwen.pkl")
    except:
        classifier.fit(X_train.float().detach().cpu().numpy(), y_train.float().detach().cpu().numpy())
        dump(classifier, "image_gender_classifier_qwen.pkl")
        classifier = load("image_gender_classifier_qwen.pkl")
    
    def predict_gender(image_embedding):
        image_embedding = image_embedding.reshape(1, -1)
        prob = classifier.predict_proba(image_embedding)[:, 1]
        return 2 * prob[0] - 1

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
elif args.mode=='sfidba':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}_{args.decoder_prune_num}_{args.t}.csv"
elif args.mode=='clipclip':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}_{args.decoder_prune_num}.csv"
elif args.mode=='naive':
    results_file = f"result/vqa_results_{args.model}_{args.mode}_{args.debiasing_target}.csv"
    tokenizer=None
    alpha_tokenizer=None
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
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
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
        # Try to load Qwen2.5-VL-3B-Instruct using Qwen2VLForConditionalGeneration (they share architecture)
        from qwen_model import Qwen2_5_VLForConditionalGeneration  # Use alias from qwen_model
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        try:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Warning: Failed to load {model_name}: {e}")
            print("Falling back to Qwen2-VL-3B-Instruct")
            from transformers import Qwen2VLForConditionalGeneration
            model_name = "Qwen/Qwen2-VL-3B-Instruct"
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True
            )
        # Create custom model with same config
        model = CustomQwenForConditionalGeneration(base_model.config)
        model.load_state_dict(base_model.state_dict(), strict=False)
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)
    
    def get_vqa_answer(image_path,lam=None,mode=None,tokenizer=None, alpha_tokenizer=None, race_map=None):
        image = Image.open(image_path).convert("RGB")

        if args.model=='llava':
            if args.debiasing_target == 'gender':
                prompt = f"USER: <image>\nWhat is the gender of the person in this image? Choose either Male or Female as your response. ASSISTANT:"
            else:  # race
                prompt = f"USER: <image>\nWhat is the race of the person in this image? Choose from: White, Black, Indian, Asian, Middle Eastern, or Latino. ASSISTANT:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                if args.mode=='logit':
                    prefix = model.vision_tower(inputs['pixel_values'].squeeze(1), output_hidden_states = True)
                    prefix = prefix.pooler_output
                    if args.debiasing_target == 'race':
                        # For race: get multi-class probabilities
                        race_probs = predict_attribute(prefix.detach().cpu().numpy())
                        generate_ids = model.generate(**inputs, max_new_tokens=50, race_probs=race_probs,
                                                lam=lam, vqa_tokenizer=tokenizer, device=device,
                                                mode=mode, race_map=race_map)
                    else:
                        # For gender: use binary classifier
                        s_scale = predict_attribute(prefix.detach().cpu().numpy())
                        generate_ids = model.generate(**inputs, max_new_tokens=50,s_scale=s_scale,text_classifier=text_classifier,
                                                token_bias=token_bias,lam=lam,neutral=args.neutral,vqa_tokenizer=tokenizer,alpha_tokenizer=alpha_tokenizer, device=device,threshold=args.t
                                                ,mode=mode)
                elif args.mode in ['sfid','clipclip']:
                    generate_ids = model.generate(**inputs, max_new_tokens=50,
                                            decoder_mean_features_lowconfidence=decoder_mean_features_lowconfidence,
                                            decoder_important_indices=decoder_importances, device=device
                                            ,mode=mode,vqa_name = 'llava')
                elif args.mode=='dear':
                    generate_ids = model.generate(**inputs, max_new_tokens=50,mode=args.mode, dear_adaptor=adaptor)    
                elif args.mode=='sfidba':
                    prefix = model.vision_tower(inputs['pixel_values'].squeeze(1), output_hidden_states = True)
                    prefix = prefix.pooler_output
                    s_scale = predict_attribute(prefix.detach().cpu().numpy())
                    generate_ids = model.generate(**inputs, max_new_tokens=50,
                                            decoder_mean_features_lowconfidence=decoder_mean_features_highconfidence,
                                            decoder_important_indices=decoder_importances, device=device
                                            ,mode=mode,vqa_name = 'llava',s_scale=s_scale)
                elif args.mode=='vdd':
                    generate_ids = model.generate(**inputs, max_new_tokens=50,mode='vdd')    
                else:
                    generate_ids = model.generate(**inputs, max_new_tokens=50,mode='naive')
                    
            answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer = answer.strip()
            answer = answer.split("ASSISTANT:")[-1]
            
        elif args.model=='paligemma':
            if args.debiasing_target == 'gender':
                prompt = "What is the gender of the person in this image? Choose either Male or Female as your response."
            else:  # race
                prompt = "What is the race of the person in this image? Choose from: White, Black, Indian, Asian, Middle Eastern, or Latino."
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(device)
            input_len = model_inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                if args.mode=='logit':
                    if args.debiasing_target == 'race':
                        # For race: get multi-class probabilities
                        vision_outputs = model.vision_tower(model_inputs['pixel_values'].squeeze(1), output_hidden_states=True)
                        vision_features = vision_outputs.last_hidden_state.mean(dim=1)
                        race_probs = predict_attribute(vision_features.detach().cpu().numpy())
                        generate_ids = model.generate(**model_inputs, max_new_tokens=50,do_sample=False,output_hidden_states=False,
                                                race_probs=race_probs, lam=args.lam, vqa_tokenizer=tokenizer, device=device,
                                                mode=args.mode, vqa_name='paligemma', race_map=race_map)
                    else:
                        # For gender: use binary classifier
                        s_scale = -1
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
                elif args.mode=='sfidba':
                    vision_outputs = model.vision_tower(model_inputs['pixel_values'].squeeze(1), output_hidden_states=True)
                    vision_features = vision_outputs.last_hidden_state.mean(dim=1)
                    s_scale = predict_attribute(vision_features.detach().cpu().numpy())
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50,do_sample=False,output_hidden_states=False,
                                                decoder_mean_features_lowconfidence=decoder_mean_features_highconfidence,
                                                decoder_important_indices=decoder_importances, device=device
                                                ,mode=args.mode,vqa_name = 'paligemma',s_scale=s_scale)
                elif args.mode=='vdd':
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50,mode=args.mode,vqa_name='paligemma')
                else:
                    generate_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False,output_hidden_states=False)
            
            generation = generate_ids[0][input_len:]
            answer = processor.decode(generation, skip_special_tokens=True)
            answer = answer.strip()
            
        elif args.model=='qwen':
            if args.debiasing_target == 'gender':
                prompt_text = "What is the gender of the person in this image? Choose either Male or Female as your response."
            else:  # race
                prompt_text = "What is the race of the person in this image? Choose from: White, Black, Indian, Asian, Middle Eastern, or Latino."
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
                    if hasattr(model, 'visual') and model.visual is not None:
                        vision_outputs = model.visual(inputs['pixel_values'])
                        if hasattr(vision_outputs, 'last_hidden_state'):
                            prefix = vision_outputs.last_hidden_state.mean(dim=1)
                        else:
                            prefix = vision_outputs.mean(dim=1)
                    else:
                        prefix = torch.zeros(1, 1024).to(device)
                    if args.debiasing_target == 'race':
                        # For race: get multi-class probabilities
                        race_probs = predict_attribute(prefix.detach().cpu().numpy())
                        generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='logit',
                                                race_probs=race_probs, lam=lam, vqa_tokenizer=tokenizer,
                                                device=device, race_map=race_map)
                    else:
                        # For gender: use binary classifier
                        s_scale = predict_attribute(prefix.detach().cpu().numpy())
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
                elif args.mode=='sfidba':
                    if hasattr(model, 'visual') and model.visual is not None:
                        vision_outputs = model.visual(inputs['pixel_values'])
                        if hasattr(vision_outputs, 'last_hidden_state'):
                            prefix = vision_outputs.last_hidden_state.mean(dim=1)
                        else:
                            prefix = vision_outputs.mean(dim=1)
                    else:
                        prefix = torch.zeros(1, 1024).to(device)
                    s_scale = predict_attribute(prefix.detach().cpu().numpy())
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50,
                                            decoder_mean_features_lowconfidence=decoder_mean_features_highconfidence,
                                            decoder_important_indices=decoder_importances, device=device
                                            ,mode=mode,vqa_name = 'qwen',s_scale=s_scale)
                elif args.mode=='vdd':
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='vdd')    
                else:
                    generate_ids = model.generate_with_debiasing(**inputs, max_new_tokens=50, mode='naive')
                    
            answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer = answer.strip()
            if "assistant" in answer.lower():
                answer = answer.split("assistant")[-1].strip()
            if ":" in answer:
                answer = answer.split(":")[-1].strip()
        
        return answer
        
    results = []
    
    # For race detection, use SocialCounterfactuals dataset images
    if args.debiasing_target == 'race':
        import glob
        # Path to saved images from SocialCounterfactuals
        saved_images_dir = os.path.join(_REPO_ROOT, "tasks", "counterfactual", "saved_images")
        
        # Get all image files that contain race information in filename
        # Patterns: 
        # - a_photo_of_physical_race_{occupation}_{id}_{physical_attr}_{race}.png
        # - a_photo_of_race_gender_{occupation}_{id}_{race}_{gender}.png
        race_image_files = glob.glob(os.path.join(saved_images_dir, "*physical_race*.png"))
        race_image_files += glob.glob(os.path.join(saved_images_dir, "*race_gender*.png"))
        
        print(f"Found {len(race_image_files)} images with race information")
        
        # Race label mapping from filename to standard format
        race_mapping = {
            'White': 'White',
            'Black': 'Black',
            'Indian': 'Indian',
            'Asian': 'Asian',
            'Middle_Eastern': 'Middle Eastern',
            'Latino': 'Latino'
        }
        
        for image_path in tqdm(race_image_files, total=len(race_image_files)):
            # Extract race from filename
            # Filename formats:
            # - a_photo_of_physical_race_{occupation}_{id}_{physical_attr}_{race}.png
            # - a_photo_of_race_gender_{occupation}_{id}_{race}_{gender}.png
            filename = os.path.basename(image_path)
            filename_no_ext = filename.replace('.png', '')
            
            # Split by underscore and find the race label
            parts = filename_no_ext.split('_')
            race = None
            occupation = None
            
            # Check which pattern we have
            has_physical_race = 'physical' in parts and 'race' in parts and parts.index('physical') < parts.index('race')
            has_race_gender = 'race' in parts and 'gender' in parts and parts.index('race') < parts.index('gender')
            
            if has_physical_race:
                # Pattern: a_photo_of_physical_race_{occupation}_{id}_{physical_attr}_{race}
                # Physical attributes that appear before race
                physical_attrs = ['obese', 'old', 'skinny', 'tattooed', 'young']
                
                # Find physical attribute index
                physical_attr_idx = None
                for attr in physical_attrs:
                    if attr in parts:
                        physical_attr_idx = parts.index(attr)
                        break
                
                # Find race - it comes after physical attribute
                # Handle multi-word races like "Middle_Eastern"
                if physical_attr_idx is not None:
                    remaining_parts = parts[physical_attr_idx + 1:]
                    
                    # Check for "Middle_Eastern" (two words)
                    if len(remaining_parts) >= 2 and remaining_parts[0] == 'Middle' and remaining_parts[1] == 'Eastern':
                        race = 'Middle Eastern'
                    # Check for single-word races
                    elif len(remaining_parts) >= 1:
                        race_candidate = remaining_parts[0]
                        if race_candidate in race_mapping:
                            race = race_mapping[race_candidate]
                
                # Extract occupation - it's between "race" (index 4) and the first number (ID)
                if 'race' in parts:
                    race_keyword_idx = parts.index('race')  # Should be index 4
                    # Find first numeric part (the ID)
                    id_start_idx = None
                    for i in range(race_keyword_idx + 1, len(parts)):
                        if parts[i].isdigit():
                            id_start_idx = i
                            break
                    
                    if id_start_idx is not None and id_start_idx > race_keyword_idx + 1:
                        # Occupation is between "race" and the ID
                        occupation = '_'.join(parts[race_keyword_idx + 1:id_start_idx])
                    elif physical_attr_idx is not None and physical_attr_idx > race_keyword_idx + 1:
                        # Fallback: occupation is between "race" and physical attribute
                        occupation = '_'.join(parts[race_keyword_idx + 1:physical_attr_idx])
            
            elif has_race_gender:
                # Pattern: a_photo_of_race_gender_{occupation}_{id_parts}_{race}_{gender}
                # Race comes after ID, before gender values
                if 'race' in parts and 'gender' in parts:
                    race_keyword_idx = parts.index('race')
                    gender_keyword_idx = parts.index('gender')
                    
                    # Find all numeric parts (the ID can be multiple parts)
                    id_start_idx = None
                    id_end_idx = None
                    for i in range(gender_keyword_idx + 1, len(parts)):
                        if parts[i].isdigit():
                            if id_start_idx is None:
                                id_start_idx = i
                            id_end_idx = i
                        elif id_start_idx is not None:
                            # Found end of ID sequence
                            break
                    
                    # Race should be right after the ID sequence
                    # Pattern: a, photo, of, race, gender, occupation, id_parts..., race_value, gender_value
                    if id_end_idx is not None:
                        race_idx = id_end_idx + 1
                        if race_idx < len(parts):
                            race_candidate = parts[race_idx]
                            # Check for "Middle_Eastern" (two words)
                            if race_candidate == 'Middle' and race_idx + 1 < len(parts) and parts[race_idx + 1] == 'Eastern':
                                race = 'Middle Eastern'
                            elif race_candidate in race_mapping:
                                race = race_mapping[race_candidate]
                    
                    # Extract occupation - it's between "gender" and the ID
                    if id_start_idx is not None and id_start_idx > gender_keyword_idx + 1:
                        occupation = '_'.join(parts[gender_keyword_idx + 1:id_start_idx])
            
            # Skip if no race information found
            if race is None:
                continue
            
            # Get VQA answer
            answer = get_vqa_answer(image_path, args.lam, args.mode, tokenizer, alpha_tokenizer, race_map if args.debiasing_target == 'race' else None)
            results.append({
                'image_path': image_path,
                'ground_truth_race': race,
                'answer': answer,
                'occupation': occupation if occupation else 'unknown'
            })
    else:
        # For gender detection, use FACET dataset (original code)
        for index, row in tqdm(new_annotations.iterrows(), total=len(new_annotations)):
            image_path = os.path.join(_FACET_DIR, 'image', row['filename'])
            occupation = row['class1']
            gender = row['gender']
            if int(gender)==0:
                gender='Male'
            elif int(gender)==1:
                gender='Female'

            race = row['race']
            answer = get_vqa_answer(image_path,args.lam,args.mode,tokenizer, alpha_tokenizer, race_map if args.debiasing_target == 'race' else None)
            results.append({'image_path':image_path,'ground_truth_gender': gender, 'race': race, 'answer': answer, 'occupation': occupation})

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

