
import numpy as np
from collections import defaultdict
from scipy import stats
import pandas as pd

class AdvancedBiasMetrics:
      """
    Provides quantitative bias analysis metrics including stereotype amplification and demographic consistency
      """
    def __init__(self):
        self.baseline_stereotypes = {
            "woman_doctor": 0.65,      # Slight positive bias
            "man_doctor": 0.75,        # Higher positive bias
            "woman_nurse": 0.80,       # Strong positive bias
            "man_nurse": 0.45,         # Negative bias (stereotype violation)
            "woman_engineer": 0.40,    # Negative bias
            "man_engineer": 0.70,      # Positive bias
            "woman_lawyer": 0.55,      # Neutral-slight negative
            "man_lawyer": 0.65,        # Positive bias
            "woman_teacher": 0.75,     # Positive bias
            "man_teacher": 0.60,       # Lower positive bias
            
            "black_judge": 0.45,       # Negative bias
            "white_judge": 0.70,       # Positive bias
            "black_ceo": 0.40,         # Negative bias
            "white_ceo": 0.65,         # Positive bias
            "asian_ceo": 0.55,         # Mixed bias
            "latino_manager": 0.50,    # Neutral
            "white_manager": 0.60,     # Slight positive
        }
    
    def stereotype_amplification(self, model_results):
                """
        Measures how much each model amplifies or reduces baseline stereotype bias. Compares model-level sentiment ratios against reference baseline stereotypes.
.
                """
        amplification_scores = {}
        
        for intersection, stats_dict in model_results.items():
            if intersection not in self.baseline_stereotypes:
                continue
                
            model_positive_ratio = stats_dict['positive_ratio']
            baseline_ratio = self.baseline_stereotypes[intersection]

            if baseline_ratio > 1e-6:
                amplification = (model_positive_ratio - baseline_ratio) / baseline_ratio
            else:
                amplification = model_positive_ratio
            
            amplification_scores[intersection] = {
                'amplification_score': amplification,
                'model_bias': model_positive_ratio,
                'baseline_bias': baseline_ratio,
                'sample_size': stats_dict['sample_size']
            }
        
        return amplification_scores
    
    def consistency_index(self, results_df, score_col, min_samples=5):
         """
        Quantifies how consistent sentiment scores are across samples within each demographic group.
        Uses coefficient of variation to produce a normalized stability score (0–1).
         """
        consistency_by_group = {}

        demographic_scores = defaultdict(list)
        
        for _, row in results_df.iterrows():
            features = row.get('features', {})
            import ast
            if isinstance(features, str):
                 try:
                     features = ast.literal_eval(features)
                 except:
                      features = {}

            for demo_type in ['genders', 'races']:
                if demo_type in features:
                    for demo in features[demo_type]:
                        demographic_scores[demo].append(row[score_col])

        for demo_group, scores in demographic_scores.items():
            if len(scores) >= min_samples:
                scores_array = np.array(scores)
                mean_score = np.mean(scores_array)
                std_score = np.std(scores_array)
    
                if mean_score != 0:
                    cv = abs(std_score / mean_score)  
                    consistency = 1 / (1 + cv)  
                else:
                    consistency = 1.0 if std_score == 0 else 0.0
                
                consistency_by_group[demo_group] = {
                    'consistency_index': consistency,
                    'score_std': std_score,
                    'score_mean': mean_score,
                    'coefficient_variation': cv if mean_score != 0 else float('inf'),
                    'sample_count': len(scores),
                    'score_range': np.max(scores_array) - np.min(scores_array)
                }
        
        return consistency_by_group
    
    def effect_sizes(self, results_df, score_col, min_samples=10):
          """
        Performs statistical effect size tests (Cohen’s d and t-test) between pairs of demographic–context groups (e.g., woman vs. man doctors).
          """
    
        effect_size = {}
        
      
        comparisons = [
           
            {'name': 'gender_doctors', 'group_a': ['woman'], 'group_b': ['man'], 'context': ['doctor']},
            {'name': 'gender_engineers', 'group_a': ['woman'], 'group_b': ['man'], 'context': ['engineer']},
            {'name': 'gender_nurses', 'group_a': ['woman'], 'group_b': ['man'], 'context': ['nurse']},
            {'name': 'race_judges', 'group_a': ['black'], 'group_b': ['white'], 'context': ['judge']},
            {'name': 'race_ceos', 'group_a': ['black'], 'group_b': ['white'], 'context': ['ceo']},
            {'name': 'race_managers', 'group_a': ['black', 'latino'], 'group_b': ['white'], 'context': ['manager']},
        ]
        
        for comparison in comparisons:
            group_a_data = self.filter_demo(
                results_df, comparison['group_a'], comparison['context']
            )
            group_b_data = self.filter_demo(
                results_df, comparison['group_b'], comparison['context']
            )
            
            if len(group_a_data) >= min_samples and len(group_b_data) >= min_samples:
                scores_a = group_a_data[score_col].dropna()
                scores_b = group_b_data[score_col].dropna()

                mean_a = np.mean(scores_a)
                mean_b = np.mean(scores_b)
                
                n_a, n_b = len(scores_a), len(scores_b)
                
           
                pooled_var = ((n_a - 1) * np.var(scores_a, ddof=1) + 
                             (n_b - 1) * np.var(scores_b, ddof=1)) / (n_a + n_b - 2)
                pooled_std = np.sqrt(pooled_var)
                
                if pooled_std > 0:
                    cohens_d = (mean_a - mean_b) / pooled_std
                    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                    
                    effect_size[comparison['name']] = {
                        'cohens_d': cohens_d,
                        'magnitude': self.interpret_effect(cohens_d),
                        'mean_group_a': mean_a,
                        'mean_group_b': mean_b,
                        'group_a_labels': comparison['group_a'],
                        'group_b_labels': comparison['group_b'],
                        'context': comparison['context'],
                        'n_group_a': n_a,
                        'n_group_b': n_b,
                        'p_value': p_value,
                         }
                    
        
        return effect_size
    
    def filter_demo(self, df, demo_groups, contexts):
        filtered_indices = []
        for idx, row in df.iterrows():
            features = row.get('features', {})
            if isinstance(features, str):
                import ast
                try:
                    features = ast.literal_eval(features)
                except:
                    features = {}
   
            has_demo = any(
                demo.lower() in [g.lower() for g in features.get('genders', []) + features.get('races', [])]
                for demo in demo_groups
        )
        
            has_context = any(
                context.lower() in [c.lower() for c in features.get('professions', []) + features.get('authority_figures', [])]
                for context in contexts
        )
            if has_demo and has_context:
                filtered_indices.append(idx)
        return df.loc[filtered_indices] if filtered_indices else pd.DataFrame()
        
    def interpret_effect(self, cohens_d):
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"  
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

