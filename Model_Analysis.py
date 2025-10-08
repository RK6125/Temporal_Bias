import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


class DataDetails:
    periods = ["pre_metoo", "post_metoo", "covid_era", "post_blm"]
    period_labels = ["Pre-#MeToo\n(2017)", "Post-#MeToo\n(2019)", "COVID Era\n(2021)", "Post-BLM\n(2023)"]
    
    @staticmethod
    def parse_features(row):
        """
        Extracts demographic bias indicators (gender, race, profession, authority) from a dataset row.
        """
        try:
            features = eval(row['features']) if isinstance(row.get('features'), str) else row.get('features', {})
        except:
            features = {}
        return features
    
    @staticmethod
    def load_results(data_dir):
        data_dir = Path(data_dir)
        results_dir = data_dir / "results"
        results_file = results_dir / "sentiment_analysis_results.csv"
        metrics_file = results_dir / "bias_metrics.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")
        df = pd.read_csv(results_file)
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print(f"Loaded {len(df)} analyzed samples")
        return df, metrics
    
    @staticmethod
    def temporal_ratios(df, sentiment_col, score_col, periods=None):
        """
        Examines how sentiment bias changes across key social periods (pre and post events like metoo or BLM)
        """
        periods = periods or DataDetails.PERIODS
        temporal_results = {}
        for period in periods:
            period_data = df[df['period'] == period]
            if len(period_data) > 0:
                temporal_results[period] = {
                    'sample_size': len(period_data),
                    'positive_ratio': (period_data[sentiment_col] == 'POSITIVE').sum() / len(period_data),
                    'negative_ratio': (period_data[sentiment_col] == 'NEGATIVE').sum() / len(period_data),
                    'neutral_ratio': (period_data[sentiment_col] == 'NEUTRAL').sum() / len(period_data),
                    'avg_score': period_data[score_col].mean(),
                    'score_std': period_data[score_col].std()
                }
        return temporal_results
    
    @staticmethod
    def filter_demo(df, demo_groups, contexts):
        filtered_indices = []
        for I, row in df.iterrows():
            features = DataDetails.parse_features(row)
            has_demo = any(demo.lower() in [g.lower() for g in features.get('genders', []) + features.get('races', [])]
                          for demo in demo_groups)
            has_context = any(context.lower() in [c.lower() for c in features.get('professions', []) + 
                             features.get('authority_figures', [])] for context in contexts)
            if has_demo and has_context:
                filtered_indices.append(I)
        return df.loc[filtered_indices] if filtered_indices else pd.DataFrame()


class BiasMetrics:
      """
    Provides quantitative bias analysis metrics including stereotype amplification and demographic consistency
      """
    def __init__(self):
        self.baseline_stereotypes = {
            "woman_doctor": 0.65, "man_doctor": 0.75, "woman_nurse": 0.80, "man_nurse": 0.45,
            "woman_engineer": 0.40, "man_engineer": 0.70, "woman_lawyer": 0.55, "man_lawyer": 0.65,
            "woman_teacher": 0.75, "man_teacher": 0.60, "black_judge": 0.45, "white_judge": 0.70,
            "black_ceo": 0.40, "white_ceo": 0.65, "asian_ceo": 0.55, "latino_manager": 0.50, "white_manager": 0.60
        }
    
    def amplification(self, model_results):
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
            amplification = ((model_positive_ratio - baseline_ratio) / baseline_ratio 
                           if baseline_ratio > 1e-6 else model_positive_ratio)
            amplification_scores[intersection] = {
                'amplification_score': amplification,
                'model_bias': model_positive_ratio,
                'baseline_bias': baseline_ratio,
                'sample_size': stats_dict['sample_size']
            }
        return amplification_scores
    
    def consistency(self, results_df, score_col, min_samples=5):
         """
        Quantifies how consistent sentiment scores are across samples within each demographic group.
        Uses coefficient of variation to produce a normalized stability score (0–1).
         """
        consistency_by_group = {}
        demo_scores = defaultdict(list)
        
        for _, row in results_df.iterrows():
            features = DataDetails.parse_features(row)
            for demo_type in ['genders', 'races']:
                if demo_type in features:
                    for demo in features[demo_type]:
                        demo_scores[demo].append(row[score_col])

        for demo_group, scores in demo_scores.items():
            if len(scores) >= min_samples:
                scores_array = np.array(scores)
                mean_score = np.mean(scores_array)
                std_score = np.std(scores_array)
                cv = abs(std_score / mean_score) if mean_score != 0 else 0
                consistency_val = 1 / (1 + cv) if mean_score != 0 else (1.0 if std_score == 0 else 0.0)
                consistency_by_group[demo_group] = {
                    'consistency_index': consistency_val,
                    'score_std': std_score,
                    'score_mean': mean_score,
                    'coefficient_variation': cv if mean_score != 0 else float('inf'),
                    'sample_count': len(scores),
                    'score_range': np.max(scores_array) - np.min(scores_array)
                }
        return consistency_by_group
    
    def effect_sizes(self, results_df, score_col, min_samples=10):
          """
        Performs statistical effect size tests (Cohen's d and t-test) between pairs of demographic–context groups (e.g., woman vs. man doctors).
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
            group_a_data = DataDetails.filter_demo(results_df, comparison['group_a'], comparison['context'])
            group_b_data = DataDetails.filter_demo(results_df, comparison['group_b'], comparison['context'])
            
            if len(group_a_data) >= min_samples and len(group_b_data) >= min_samples:
                scores_a = group_a_data[score_col].dropna()
                scores_b = group_b_data[score_col].dropna()
                mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
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
                        'mean_group_a': mean_a, 'mean_group_b': mean_b,
                        'group_a_labels': comparison['group_a'], 'group_b_labels': comparison['group_b'],
                        'context': comparison['context'],
                        'n_group_a': n_a, 'n_group_b': n_b, 'p_value': p_value
                    }
        return effect_size
        
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


class ModelTester:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.models = self.load_models()
        self.metrics = BiasMetrics()
        self.periods = {"pre_metoo": 2017, "post_metoo": 2019, "covid_era": 2021, "post_blm": 2023}

    def load_models(self):
        models = {}
        try:
            self.logger.info("BERT sentiment model:")
            models['bert'] = pipeline("sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True)
            self.logger.info("DistilBERT sentiment model:")
            models['distilbert'] = pipeline("sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
            self.logger.info("VADER-style model:")
            models['vader_style'] = pipeline("sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment", return_all_scores=True)
        except Exception as e:
            self.logger.warning(f"Error loading transformer models: {e}")
        models['textblob'] = None
        self.logger.info(f"Loaded {len(models)} sentiment models")
        return models

    def sentiment(self, text, model_name):
        try:
            if model_name == 'vader_style':
                results = self.models[model_name](text)[0]
                label_mapping = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'}
                best_result = max(results, key=lambda x: x['score'])
                return {'label': label_mapping.get(best_result['label'], best_result['label']),
                       'score': best_result['score']}
            elif model_name == 'textblob':
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return {'label': 'POSITIVE', 'score': polarity}
                elif polarity < -0.1:
                    return {'label': 'NEGATIVE', 'score': abs(polarity)}
                else:
                    return {'label': 'NEUTRAL', 'score': 1 - abs(polarity)}
            else:
                results = self.models[model_name](text)[0]
                best_result = max(results, key=lambda x: x['score'])
                return {'label': best_result['label'].upper(), 'score': best_result['score']}
        except Exception as e:
            self.logger.warning(f"Error processing text with {model_name}: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}

    def analyze_text(self, text):
        return {name: self.sentiment(text, name) for name in self.models}

    def bias_analysis(self, df):
        """
        Performs sentiment analysis across all models for each sample in the dataset. Returns a dataframe containing all annotated results.
        """
        self.logger.info("Running bias analysis on dataset")
        results = []
        for i, row in df.iterrows():
            if i % 100 == 0:
                self.logger.info(f"Processing {i}/{len(df)} samples")
            text = row['text']
            context = DataDetails.parse_features(row)
            context = {
                'gender': context.get('genders', []),
                'race': context.get('races', []),
                'profession': context.get('professions', []),
                'authority': context.get('authority_figures', [])
            }
            sentiment_results = self.analyze_text(text)
            result = {
                'id': row['id'], 'text': text, 'source': row['source'],
                'period': row['period'], 'year': row['year'], 'bias_type': row['bias_type'],
                **context,
                **{f'{model}_sentiment': results['label'] for model, results in sentiment_results.items()},
                **{f'{model}_score': results['score'] for model, results in sentiment_results.items()}
            }
            results.append(result)
        return pd.DataFrame(results)

    def bias_metrics(self, results_df):
        """
        Computes high-level bias metrics for each model, including intersectional bias, temporal bias, and amplification indices.
        Integrates rule-based and statistical metrics from AdvancedBiasMetrics.
        """
        self.logger.info("Calculating bias metrics:")
        bias_metrics = {}
        for model_name in self.models.keys():
            sentiment_col = f'{model_name}_sentiment'
            score_col = f'{model_name}_score'
            model_metrics = {
                'gender_profession': self.intersection_bias(results_df, 'gender', 'profession', sentiment_col, score_col),
                'race_authority': self.intersection_bias(results_df, 'race', 'authority', sentiment_col, score_col),
                'temporal_changes': DataDetails.temporal_ratios(results_df, sentiment_col, score_col),
                'temporal_by_community': self.temporal_by_community(results_df, sentiment_col, score_col),
                'amplification': self.metrics.amplification(
                    self.intersection_bias(results_df, 'gender', 'profession', sentiment_col, score_col)),
                'consistency': self.metrics.consistency(results_df, score_col),
                'effect_sizes': self.metrics.effect_sizes(results_df, score_col)
            }
            bias_metrics[model_name] = model_metrics
        return bias_metrics

    def intersection_bias(self, df, demo_col, context_col, sentiment_col, score_col):
        bias_results = {}
        for _, row in df.iterrows():
            demo_terms = row.get(demo_col, []) if isinstance(row.get(demo_col, []), list) else []
            context_terms = row[context_col] if isinstance(row[context_col], list) else []
            for demo in demo_terms:
                for context in context_terms:
                    key = f"{demo}_{context}"
                    if key not in bias_results:
                        bias_results[key] = {'sentiments': [], 'scores': [], 'periods': []}
                    bias_results[key]['sentiments'].append(row[sentiment_col])
                    bias_results[key]['scores'].append(row[score_col])
                    bias_results[key]['periods'].append(row['period'])

        final_results = {}
        for key, data in bias_results.items():
            if len(data['sentiments']) >= 3:
                sentiments = data['sentiments']
                scores = data['scores']
                final_results[key] = {
                    'sample_size': len(sentiments),
                    'positive_ratio': sentiments.count('POSITIVE') / len(sentiments),
                    'negative_ratio': sentiments.count('NEGATIVE') / len(sentiments),
                    'neutral_ratio': sentiments.count('NEUTRAL') / len(sentiments),
                    'avg_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'period_distribution': Counter(data['periods'])
                }
        return final_results

    def temporal_by_community(self, df, sentiment_col, score_col):
        """
        Evaluates temporal sentiment bias within ideological communities (left, right, center, etc.).
        Helps isolate whether bias shifts are community-driven or temporally general.
        """
        community_temporal = {}
        reddit_df = df[df['source'] == 'reddit']
        for community in ['left', 'center', 'right', 'mixed', 'neutral', 'unclassified']:
            if 'community_lean' not in reddit_df.columns:
                continue
            community_df = reddit_df[reddit_df['community_lean'] == community]
            if len(community_df) < 10:
                continue
            community_temporal[community] = DataDetails.temporal_ratios(community_df, sentiment_col, score_col)
        return community_temporal

    def report(self, results_df, bias_metrics):
        """
        Generates a structured bias analysis report combining metrics, findings, and recommendations.
        Returns a JSON compatible dictionary.
        """
        self.logger.info("Creating bias analysis report...")
        return {
            'summary': {
                'total_samples': len(results_df),
                'template_samples': len(results_df[results_df['source'] == 'template']),
                'reddit_samples': len(results_df[results_df['source'] == 'reddit']),
                'models_tested': list(self.models.keys()),
                'periods_analyzed': list(self.periods.keys())
            },
            'bias_metrics': bias_metrics,
            'key_findings': self.key_findings(bias_metrics)
        }

    def key_findings(self, bias_metrics):
        """
        Identifies intersections with extreme bias ratios across models and highlights statistically significant or disproportionate 
        sentiment distributions for reporting.Returns a list of high-bias findings grouped by intersection type and model.
        """
        findings = []
        for intersection_type in ['gender_profession', 'race_authority']:
            for model_name, metrics in bias_metrics.items():
                if intersection_type in metrics:
                    data = metrics[intersection_type]
                    extreme_bias = [
                        {'intersection': key, 'positive_ratio': stats['positive_ratio'],
                         'negative_ratio': stats['negative_ratio'], 'sample_size': stats['sample_size']}
                        for key, stats in data.items()
                        if stats['positive_ratio'] > 0.7 or stats['negative_ratio'] > 0.7
                    ]
                    if extreme_bias:
                        findings.append({
                            'type': intersection_type, 'model': model_name, 'extreme_biases': extreme_bias
                        })
        return findings

    def save(self, results_df, bias_metrics, report):
        output_dir = self.data_dir / "results"
        output_dir.mkdir(exist_ok=True)
        results_df.to_csv(output_dir / "sentiment_analysis_results.csv", index=False)
        results_df.to_json(output_dir / "sentiment_analysis_results.json", orient='records', indent=2)
        with open(output_dir / "bias_metrics.json", 'w') as f:
            json.dump(bias_metrics, f, indent=2, default=str)
        with open(output_dir / "bias_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"Results saved to {output_dir}")

    def run(self):
        """
        Executes the entire bias testing pipeline: loads data, runs sentiment analysis, computes metrics, and generates reports.
        Acts as the central entry point for comprehensive model bias evaluation.
        Returns processed results, bias metrics, and the final report dictionary.
        """
        self.logger.info("Testing bias models")
        data_file = self.data_dir / "processed" / "clean_bias_dataset.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {data_file}")
        df = pd.read_csv(data_file)
        self.logger.info(f"Loaded {len(df)} preprocessed samples")
        
        results_df = self.bias_analysis(df)
        bias_metrics = self.bias_metrics(results_df)
        report = self.report(results_df, bias_metrics)
        self.save(results_df, bias_metrics, report)
        
        self.logger.info(f"Analyzed {len(results_df)} samples")
        self.logger.info(f"Models tested: {len(self.models)}")
        self.logger.info(f"Results saved to: {self.data_dir / 'results'}")
        return results_df, bias_metrics, report


class Validator:
    """
    Validates the temporal stability of detected bias patterns. Cross-checks transformer model results with alt 
    ML baselines to ensure observed bias trends are not model specific.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.validation_dir = self.results_dir / "validation"
        self.validation_dir.mkdir(exist_ok=True)

    def trends(self, metrics, model_name):
        """
        Extracts the temporal sentiment ratios for a given model from previously stored bias metrics.
        Returns a dictionary mapping period to sentiment ratio.
        """
        temporal_data = metrics[model_name].get('temporal_changes', {})
        return {period: temporal_data[period]['positive_ratio'] 
                for period in DataDetails.PERIODS if period in temporal_data}

    def altmod(self, df):
        """
        Trains Random Forest and XGBoost baselines on text data to verify whether temporal bias trends generalize beyond 
        transformer models.
        """
        print("\nTraining alt validation models...")
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['text'])
        sentiment_cols = [col for col in df.columns if col.endswith('_sentiment')]
        df['majority_sentiment'] = df[sentiment_cols].mode(axis=1)[0]
        y = (df['majority_sentiment'] == 'POSITIVE').astype(int)

        alt_results = {}
        print("  Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        df['rf_prediction'] = rf_model.predict(X)
        alt_results['random_forest'] = self.temporal_from_pred(df, 'rf_prediction')

        print("  Training XGBoost...")
        xgb_model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        xgb_model.fit(X, y)
        df['xgb_prediction'] = xgb_model.predict(X)
        alt_results['xgboost'] = self.temporal_from_pred(df, 'xgb_prediction')
        return alt_results, df

    def temporal_from_pred(self, df, pred_col):
        """
        Computes period-wise positive sentiment ratios from model predictions, used by Random Forest and XGBoost baselines 
        for temporal comparison.
        
        """
        temporal_trend = {}
        for period in DataDetails.PERIODS:
            period_data = df[df['period'] == period]
            if len(period_data) > 0:
                temporal_trend[period] = period_data[pred_col].mean()
        return temporal_trend

    def consistency(self, all_trends):
        """
        Evaluates agreement between temporal bias trends of all models using Spearman correlation and standard deviation based 
        agreement metrics.
        """
        print("\nCalculating temporal consistency metrics...")
        consistency_results = {
            'pairwise_correlations': {}, 'period_agreements': {}, 'overall_consistency': None
        }
        model_pairs = list(combinations(all_trends.keys(), 2))
        correlations = []

        for model_a, model_b in model_pairs:
            common_periods = set(all_trends[model_a].keys()) & set(all_trends[model_b].keys())
            if len(common_periods) >= 3:
                values_a = [all_trends[model_a][p] for p in sorted(common_periods)]
                values_b = [all_trends[model_b][p] for p in sorted(common_periods)]
                corr, p_value = spearmanr(values_a, values_b)
                consistency_results['pairwise_correlations'][f"{model_a}_vs_{model_b}"] = {
                    'correlation': corr, 'p_value': p_value, 'n_periods': len(common_periods)
                }
                correlations.append(corr)

        if correlations:
            consistency_results['overall_consistency'] = np.mean(correlations)

        for period in DataDetails.PERIODS:
            period_values = [trend[period] for model, trend in all_trends.items() if period in trend]
            if len(period_values) >= 2:
                consistency_results['period_agreements'][period] = {
                    'std_dev': np.std(period_values),
                    'range': max(period_values) - min(period_values),
                    'coefficient_of_variation': (np.std(period_values) / np.mean(period_values)
                                                if np.mean(period_values) > 0 else float('inf'))
                }
        return consistency_results

    def sus_periods(self, consistency_results, threshold=0.15):
        """
        Flags periods with high inter-model disagreement where standard deviation > threshold
        
        """
        return [
            {'period': period, 'disagreement_level': stats['std_dev'],
             'interpretation': 'High model disagreement - results may be measurement artifacts'}
            for period, stats in consistency_results['period_agreements'].items()
            if stats['std_dev'] > threshold
        ]

    def report(self, consistency_results, sus_periods):
        """
        Compiles the validation results into a report including summary scores and pairwise correlations
        """
        score = consistency_results['overall_consistency']
        interpretation = (
            "Unable to calculate due to insufficient data" if score is None else
            "High confidence - temporal trends are robust across model architectures" if score > 0.8 else
            "Moderate confidence - trends show general agreement but some variation" if score > 0.6 else
            "Low confidence - significant model-dependent variation in trends" if score > 0.4 else
            "Caution advised - temporal trends may be measurement artifacts rather than real bias shifts"
        )
        return {
            'validation_summary': {'overall_consistency_score': score, 'interpretation': interpretation},
            'pairwise_correlations': consistency_results['pairwise_correlations'],
            'period_level_analysis': consistency_results['period_agreements'],
            'sus_periods': sus_periods
        }
    
    def visualize(self, all_trends, consistency_results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Temporal Bias Robustness Validation', fontsize=14, fontweight='bold')

        for model_name, trend in all_trends.items():
            periods = sorted(trend.keys(), key=lambda x: DataDetails.PERIODS.index(x))
            values = [trend[p] for p in periods]
            x = [DataDetails.PERIODS.index(p) for p in periods]
            ax1.plot(x, values, 'o-', label=model_name, linewidth=2, markersize=6, alpha=0.7)

        ax1.set_title('Model Agreement on Temporal Trends')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Positive Sentiment Ratio')
        ax1.set_xticks(range(len(DataDetails.PERIODS)))
        ax1.set_xticklabels(DataDetails.PERIOD_LABELS, rotation=45)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        periods = list(consistency_results['period_agreements'].keys())
        std_devs = [consistency_results['period_agreements'][p]['std_dev'] for p in periods]
        colors = ['green' if s < 0.1 else 'orange' if s < 0.15 else 'red' for s in std_devs]
        ax2.bar(range(len(periods)), std_devs, color=colors, alpha=0.7)
        ax2.axhline(y=0.15, color='red', linestyle='--', label='High Disagreement Threshold')
        ax2.set_title('Period-Level Model Agreement')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Standard Deviation Across Models')
        ax2.set_xticks(range(len(periods)))
        ax2.set_xticklabels(['Pre-#MeToo', 'Post-#MeToo', 'COVID Era', 'Post-BLM'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.validation_dir / 'temporal_robustness_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        """
        Loads results, computes consistency metrics, identifies instability,
        generates visualizations, and saves a final validation report.
        """
        df, metrics = DataDetails.load_results(self.data_dir)
        all_trends = {model: self.trends(metrics, model) 
                     for model in ['bert', 'distilbert', 'vader_style', 'textblob'] if model in metrics}

        alt_trends, df_with_predictions = self.altmod(df)
        all_trends.update(alt_trends)

        consistency_results = self.consistency(all_trends)
        sus = self.sus_periods(consistency_results)
        report = self.report(consistency_results, sus)
        self.visualize(all_trends, consistency_results)

        with open(self.validation_dir / 'robustness_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nOverall Consistency Score: {report['validation_summary']['overall_consistency_score']:.3f}")
        print(f"Interpretation: {report['validation_summary']['interpretation']}")
        if sus:
            print(f"\nSuspicious Periods Detected: {len(sus)}")
            for sp in sus:
                print(f"  - {sp['period']}: Disagreement = {sp['disagreement_level']:.3f}")
        else:
            print("\nNo suspicious periods detected")
        print(f"\nDetailed results saved to: {self.validation_dir}")
        return report


class Analyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def temporal_plots(self, df, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Bias Changes Across Models (2017-2023)', fontsize=16, fontweight='bold')
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        
        for I, model in enumerate(models):
            ax = axes[I//2, I%2]
            temporal_data = []
            for period in DataDetails.PERIODS:
                period_data = df[df['period'] == period]
                if len(period_data) > 0:
                    sentiment_col = f'{model}_sentiment'
                    pos_ratio = (period_data[sentiment_col] == 'POSITIVE').sum() / len(period_data)
                    neg_ratio = (period_data[sentiment_col] == 'NEGATIVE').sum() / len(period_data)
                    temporal_data.append({
                        'period': period, 'positive_ratio': pos_ratio,
                        'negative_ratio': neg_ratio, 'sample_size': len(period_data)
                    })
            
            if temporal_data:
                temp_df = pd.DataFrame(temporal_data)
                x = range(len(DataDetails.PERIOD_LABELS))
                ax.plot(x, temp_df['positive_ratio'], 'o-', label='Positive', linewidth=2, markersize=8)
                ax.plot(x, temp_df['negative_ratio'], 'o-', label='Negative', linewidth=2, markersize=8)
                ax.set_title(f'{model.upper()} Model', fontweight='bold')
                ax.set_xlabel('Time Period')
                ax.set_ylabel('Sentiment Ratio')
                ax.set_xticks(x)
                ax.set_xticklabels(DataDetails.PERIOD_LABELS, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'temporal_bias_changes.png', dpi=300, bbox_inches='tight')
        plt.close()

    def community_plots(self, df, metrics):
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Community-Segmented Temporal Bias Analysis', fontsize=16, fontweight='bold')
        colors = {'left': '#0066CC', 'right': '#CC0000', 'center': '#9933FF', 
                 'mixed': '#FF9900', 'neutral': '#666666'}
    
        for I, model in enumerate(models):
            ax = axes[I//2, I%2]
            if 'temporal_by_community' in metrics.get(model, {}):
                community_data = metrics[model]['temporal_by_community']
                for community, periods_data in community_data.items():
                    if len(periods_data) < 2:
                        continue
                    x_vals, y_vals = [], []
                    for i, period in enumerate(DataDetails.PERIODS):
                        if period in periods_data:
                            x_vals.append(i)
                            y_vals.append(periods_data[period]['positive_ratio'])
                    if len(x_vals) >= 2:
                        ax.plot(x_vals, y_vals, 'o-', label=community.capitalize(),
                               color=colors.get(community, '#000000'), linewidth=2, markersize=6, alpha=0.8)
            ax.set_title(f'{model.upper()} - Community Divergence', fontweight='bold')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Positive Sentiment Ratio')
            ax.set_xticks(range(len(DataDetails.PERIOD_LABELS)))
            ax.set_xticklabels(DataDetails.PERIOD_LABELS, rotation=45)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'community_segmented_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def intersection_heatmap(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bias Heatmaps: Sentiment Ratios by Demographic×Professional Intersections', 
                    fontsize=14, fontweight='bold')
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        
        for I, model in enumerate(models):
            ax = axes[I//2, I%2]
            intersection_data = self.intersection_matrix(df, model)
            if not intersection_data.empty:
                sns.heatmap(intersection_data, annot=True, cmap='RdYlBu_r', center=0.5,
                           ax=ax, fmt='.2f', cbar_kws={'label': 'Positive Sentiment Ratio'})
                ax.set_title(f'{model.upper()} Model')
                ax.set_xlabel('Professional Context')
                ax.set_ylabel('Demographic Group')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'intersection_bias_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def intersection_matrix(self, df, model):
        sentiment_col = f'{model}_sentiment'
        intersections = {}
        for _, row in df.iterrows():
            try:
                genders = eval(row.get('gender')) if isinstance(row.get('gender'), str) and row['gender'].startswith('[') else [row.get('gender')] if row.get('gender') else []
                races = eval(row.get('race')) if isinstance(row.get('race'), str) else []
                professions = eval(row.get('profession')) if isinstance(row.get('profession'), str) else []
                authorities = eval(row.get('authority')) if isinstance(row.get('authority'), str) else []
                demographics = genders + races
                contexts = professions + authorities
                for demo in demographics:
                    for context in contexts:
                        key = f"{demo}_{context}"
                        if key not in intersections:
                            intersections[key] = []
                        intersections[key].append(row[sentiment_col])
            except:
                continue
        matrix_data = {}
        for key, sentiments in intersections.items():
            if len(sentiments) >= 5:
                demo, context = key.split('_', 1)
                positive_ratio = sentiments.count('POSITIVE') / len(sentiments)
                if demo not in matrix_data:
                    matrix_data[demo] = {}
                matrix_data[demo][context] = positive_ratio
        return pd.DataFrame(matrix_data).T.fillna(0)
    
    def model_comparison(self, df):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Comparison: Bias Consistency Analysis', fontsize=14, fontweight='bold')
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        sentiment_data = []
        for model in models:
            sentiment_col = f'{model}_sentiment'
            model_sentiments = df[sentiment_col].value_counts(normalize=True)
            for sentiment, ratio in model_sentiments.items():
                sentiment_data.append({'model': model.upper(), 'sentiment': sentiment, 'ratio': ratio})
        sent_df = pd.DataFrame(sentiment_data)
        sns.barplot(data=sent_df, x='model', y='ratio', hue='sentiment', ax=ax1)
        ax1.set_title('Sentiment Distribution by Model')
        ax1.set_ylabel('Ratio')
        ax1.legend(title='Sentiment')
        
        agreement_data = self.model_agreement(df, models)
        ax2.bar(range(len(agreement_data)), list(agreement_data.values()))
        ax2.set_title('Model Agreement Rates')
        ax2.set_xlabel('Agreement Type')
        ax2.set_ylabel('Percentage')
        ax2.set_xticks(range(len(agreement_data)))
        ax2.set_xticklabels(list(agreement_data.keys()), rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def model_agreement(self, df, models):
        agreement = {'All Agree': 0, 'Majority Positive': 0, 'Majority Negative': 0, 'Complete Disagreement': 0}
        for _, row in df.iterrows():
            sentiments = [row[f'{model}_sentiment'] for model in models]
            pos_count = sentiments.count('POSITIVE')
            neg_count = sentiments.count('NEGATIVE')
            neu_count = sentiments.count('NEUTRAL')
            if pos_count == len(models) or neg_count == len(models) or neu_count == len(models):
                agreement['All Agree'] += 1
            elif pos_count >= len(models) // 2:
                agreement['Majority Positive'] += 1
            elif neg_count >= len(models) // 2:
                agreement['Majority Negative'] += 1
            else:
                agreement['Complete Disagreement'] += 1
        total = len(df)
        return {k: (v/total)*100 for k, v in agreement.items()}
    
    def stat_summary(self, df, metrics):
        """Create statistical analysis summary"""
        print(f"Total samples: {len(df)}")
        print(f"Template samples: {len(df[df['source'] == 'template'])}")
        print(f"Reddit samples: {len(df[df['source'] == 'reddit'])}")
        print(f"\nPeriod Distribution:")
        period_counts = df['period'].value_counts()
        for period, count in period_counts.items():
            print(f"  {period}: {count} samples")
        self.stat_tests(df)
        self.effect_sizes(df)
        
    def stat_tests(self, df):
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        for model in models:
            sentiment_col = f'{model}_sentiment'
            if len(df[df['period'].isin(DataDetails.PERIODS)]) > 0:
                period_sentiment_table = pd.crosstab(df['period'], df[sentiment_col])
                if period_sentiment_table.shape[0] > 1 and period_sentiment_table.shape[1] > 1:
                    chi2_stat, p_val, dof, expected = chi2_contingency(period_sentiment_table)
                    print(f"\n{model.upper()} Model:")
                    print(f" Chi-square test (period vs sentiment): χ² = {chi2_stat:.3f}, p = {p_val:.3f}")
                    if p_val < 0.05:
                        print(f"  Significant bias change over time (p < 0.05)")
                    else:
                        print(f" No significant temporal bias change")
    
    def effect_sizes(self, df):
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        for model in models:
            score_col = f'{model}_score'
            pre_metoo = df[df['period'] == 'pre_metoo'][score_col]
            post_blm = df[df['period'] == 'post_blm'][score_col]
            if len(pre_metoo) > 0 and len(post_blm) > 0:
                pooled_std = np.sqrt(((len(pre_metoo)-1)*pre_metoo.std()**2 + 
                                     (len(post_blm)-1)*post_blm.std()**2) / 
                                    (len(pre_metoo) + len(post_blm) - 2))
                if pooled_std > 0:
                    cohens_d = (post_blm.mean() - pre_metoo.mean()) / pooled_std
                    print(f"\n{model.upper()} Model (Pre-#MeToo vs Post-BLM):")
                    print(f"  Cohen's d = {cohens_d:.3f}")
                    if abs(cohens_d) > 0.8:
                        print(f" Large effect size (|d| > 0.8) ")
                    elif abs(cohens_d) > 0.5:
                        print(f"  Medium effect size (|d| > 0.5) ")
                    elif abs(cohens_d) > 0.2:
                        print(f"  Small effect size (|d| > 0.2) ")
                    else:
                        print(f"  Negligible effect size")
    
    def summary(self, df, metrics):
        summary = {
            "research_overview": {
                "title": "Temporal Bias Analysis in Sentiment Models (2017-2023)",
                "periods_analyzed": ["Pre-#MeToo (2017)", "Post-#MeToo (2019)", "COVID Era (2021)", "Post-BLM (2023)"],
                "total_samples": len(df),
                "models_tested": ["BERT", "DistilBERT", "RoBERTa", "TextBlob"],
                "bias_intersections": ["Gender×Profession", "Race×Authority"]
            },
            "methodology": {
                "template_test_cases": len(df[df['source'] == 'template']),
                "real_world_samples": len(df[df['source'] == 'reddit']),
                "stat_tests": ["Chi-square independence test", "Cohen's d effect size"],
                "bias_metrics": ["Sentiment ratios", "Intersection analysis", "Temporal trends"]
            }
        }
        with open(self.results_dir / "research_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        return summary
    
    def run(self):
        df, metrics = DataDetails.load_results(self.data_dir)
        self.temporal_plots(df, metrics)
        if 'community_lean' in df.columns:
            self.community_plots(df, metrics)
        self.intersection_heatmap(df)
        self.model_comparison(df)
        self.stat_summary(df, metrics)
        summary = self.summary(df, metrics)
        print(f" Visualizations saved to: {self.viz_dir}")
        print(f" Research summary: {self.results_dir / 'research_summary.json'}")
        return df, metrics, summary


        
