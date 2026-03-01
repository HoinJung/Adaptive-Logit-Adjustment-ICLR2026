from sklearn.utils import resample
import numpy as np
from tqdm import tqdm
import ast
from pprint import pprint
import pandas as pd
import json
from collections import defaultdict

masculine = ['man','men','male','father','gentleman','gentlemen','boy','boys','uncle','husband','actor',
            'prince','waiter','son','he','his','him','himself','brother','brothers', 'guy', 'guys',
            'emperor','emperors','dude','dudes','cowboy','boyfriend','chairman','policeman','policemen','mr']
feminine = ['woman','women','female','lady','ladies','mother','girl', 'girls','aunt','wife','actress',
            'princess','waitress','daughter','she','her','hers','herself','sister','sisters', 'queen',
            'queens','pregnant','girlfriend','chairwoman','policewoman','policewomen','ms']
gender_words = masculine + feminine

neutral = ['person','people','parent','parents','child','children','spouse','server','they','their','them',
           'theirs','baby','babies','partner','partners','friend','friends','spouse','spouses','sibling',
           'siblings', 'chairperson','officer', 'surfer', 'kid', 'kids']


def standardize_scores(scores):
    if not scores:
        return scores
    values = np.array(list(scores.values()))
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val > 0:
        return {token: (score - mean_val) / std_val for token, score in scores.items()}
    return {token: 0 for token in scores.keys()}


def normalize_scores(scores):
    if not scores:
        return scores
    min_val, max_val = min(scores.values()), max(scores.values())
    if max_val > min_val:
        return {token: (score - min_val) / (max_val - min_val) for token, score in scores.items()}
    return scores


def load_and_normalize_beta(importance_dict_path):
    """
    Load token importance from JSON. Handles gender format {male, female} or direct {token: score}.
    Returns dict mapping tokens to normalized bias scores (β_i).
    """
    with open(importance_dict_path, "r") as f:
        importance_dict = json.load(f)
    if "male" in importance_dict or "female" in importance_dict:
        male_scores = defaultdict(float, importance_dict.get("male", {}))
        female_scores = defaultdict(float, importance_dict.get("female", {}))
        all_tokens = set(male_scores.keys()).union(set(female_scores.keys()))
        token_bias = {token: male_scores.get(token, 0) - female_scores.get(token, 0) for token in all_tokens}
    else:
        token_bias = dict(importance_dict)
    if not token_bias:
        return {}
    min_beta = min(token_bias.values())
    max_beta = max(token_bias.values())
    if max_beta > 0 and min_beta < 0:
        for token, value in token_bias.items():
            if value >= 0:
                token_bias[token] = value / max_beta
            else:
                token_bias[token] = value / abs(min_beta)
    elif max_beta > min_beta:
        for token, value in token_bias.items():
            token_bias[token] = (value - min_beta) / (max_beta - min_beta)
    return token_bias


def decide_gender(sent_tokens):
    gender_list = []
    for token in sent_tokens:
        token = token.lower()
        if token in masculine:
            gender_list.append('Male')
        if token in feminine:
            gender_list.append('Female')
        if token in neutral:
            gender_list.append('Neut')
    if 'Male' in gender_list and 'Female' not in gender_list:
        gender = 'Male'
    elif 'Male' not in gender_list and 'Female' in gender_list:
        gender = 'Female'
    elif 'Male' in gender_list and 'Female' in gender_list:
        gender = 'Both'
    elif 'Neut' in gender_list:
        gender = 'Neut'
    else:
        gender = 'None'
    return gender


def bootstrap(df, num_samples=1000, sample_size=10000):
    bootstrap_results = []
    for _ in tqdm(range(num_samples)):
        sample_df = resample(df, n_samples=min(sample_size, len(df)), replace=True)
        rates = report_df(sample_df)
        male_mr = rates['Male Misclassification Rate']
        female_mr = rates['Female Misclassification Rate']
        rates['Absolute Difference'] = abs(male_mr - female_mr)
        bootstrap_results.append(rates)
    return bootstrap_results


def calculate_confidence_intervals(bootstrap_results, confidence_level=0.95):
    metrics = [
        'Male Misclassification Rate', 'Female Misclassification Rate',
        'Overall Misclassification Rate', 'Composite Misclassification Rate', 'Absolute Difference'
    ]
    ci_lower = {}
    ci_upper = {}
    for metric in metrics:
        values = [result[metric] for result in bootstrap_results]
        ci_lower[metric] = np.percentile(values, (1 - confidence_level) / 2 * 100)
        ci_upper[metric] = np.percentile(values, (1 + confidence_level) / 2 * 100)
    return ci_lower, ci_upper


def misclassification_rate(df):
    total_males = df[df['ground_truth_gender'] == 'Male'].shape[0]
    total_females = df[df['ground_truth_gender'] == 'Female'].shape[0]
    male_lowconfidence = df[(df['ground_truth_gender'] == 'Male') & (df['detected_gender'] == 'Female')].shape[0]
    female_lowconfidence = df[(df['ground_truth_gender'] == 'Female') & (df['detected_gender'] == 'Male')].shape[0]
    male_mr = male_lowconfidence / total_males if total_males > 0 else 0
    female_mr = female_lowconfidence / total_females if total_females > 0 else 0
    overall_mr = (male_lowconfidence + female_lowconfidence) / (total_males + total_females) if (total_males + total_females) > 0 else 0
    composite_mr = np.sqrt(overall_mr**2 + (female_mr - male_mr)**2)
    abs_diff = abs(male_mr - female_mr)
    return {
        'Male Misclassification Rate': round(male_mr * 100, 2),
        'Female Misclassification Rate': round(female_mr * 100, 2),
        'Overall Misclassification Rate': round(overall_mr * 100, 2),
        'Composite Misclassification Rate': round(composite_mr * 100, 2),
        'Absolute Difference': round(abs_diff * 100, 2),
    }


def convert_str_to_list(str_list):
    try:
        return ast.literal_eval(str_list)
    except ValueError:
        return []


def report_df(df):
    return misclassification_rate(df)


def evaluate_facet_open(file_path):
    print(f'Evaluating Image Captioning for {file_path}')
    df = pd.read_csv(file_path)
    bootstrap_results = bootstrap(df)
    ci_lower, ci_upper = calculate_confidence_intervals(bootstrap_results)

    def mean_margin(lower, upper):
        return (lower + upper) / 2, (upper - lower) / 2

    abs_diff_mean, abs_diff_margin = mean_margin(ci_lower['Absolute Difference'], ci_upper['Absolute Difference'])
    male_mis_mean, male_mis_margin = mean_margin(ci_lower['Male Misclassification Rate'], ci_upper['Male Misclassification Rate'])
    female_mis_mean, female_mis_margin = mean_margin(ci_lower['Female Misclassification Rate'], ci_upper['Female Misclassification Rate'])
    overall_mis_mean, overall_mis_margin = mean_margin(ci_lower['Overall Misclassification Rate'], ci_upper['Overall Misclassification Rate'])
    composite_mis_mean, composite_mis_margin = mean_margin(ci_lower['Composite Misclassification Rate'], ci_upper['Composite Misclassification Rate'])

    new_row = {
        'file_path': file_path,
        'Male Misclassification Rate': f"{male_mis_mean:.2f} ± {male_mis_margin:.2f}",
        'Female Misclassification Rate': f"{female_mis_mean:.2f} ± {female_mis_margin:.2f}",
        'Overall Misclassification Rate': f"{overall_mis_mean:.2f} ± {overall_mis_margin:.2f}",
        'Composite Misclassification Rate': f"{composite_mis_mean:.2f} ± {composite_mis_margin:.2f}",
        '|Male-Female|': f"{abs_diff_mean:.2f} ± {abs_diff_margin:.2f}",
    }
    output_file = file_path.replace(".csv", "_eval.csv")
    try:
        results_df = pd.read_csv(output_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=new_row.keys())
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    pprint(new_row)
