import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import torch
from bias_list import BiasTerms


class BiasModelTester:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.setup_logging()

        self.models = self._load_models()

        from bias_metrics import AdvancedBiasMetrics
        self.advanced_metrics = AdvancedBiasMetrics()

        self.periods = {
            "pre_metoo": 2017,
            "post_metoo": 2019,
            "covid_era": 2021,
            "post_blm": 2023
        }
        bias_terms = BiasTerms.get_dict()
        self.demographic_groups = {
            "gender": bias_terms["genders"],
            "race": bias_terms["races"]
        }
        self.professional_contexts = {
            "professions": bias_terms["professions"],
            "authority": bias_terms["authority_figures"]
        }

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_models(self):
        models = {}
        try:
            self.logger.info(" BERT sentiment model:")
            models['bert'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            self.logger.info(" DistilBERT sentiment model:")
            models['distilbert'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            self.logger.info("VADER-style model:")
            models['vader_style'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                return_all_scores=True
            )
        except Exception as e:
            self.logger.warning(f"Error loading transformer models: {e}")
            self.logger.info("Defaulting to TextBlob")

        models['textblob'] = None

        self.logger.info(f"Loaded {len(models)} sentiment models")
        return models

    def sentiment_scores(self, text, model_name):
        try:
            if model_name == 'vader_style':
                results = self.models[model_name](text)[0]
                label_mapping = {
                    'LABEL_0': 'NEGATIVE',
                    'LABEL_1': 'NEUTRAL',
                    'LABEL_2': 'POSITIVE'
                }
                best_result = max(results, key=lambda x: x['score'])
                return {
                    'label': label_mapping.get(best_result['label'], best_result['label']),
                    'score': best_result['score']
                }

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
                return {
                    'label': best_result['label'].upper(),
                    'score': best_result['score']
                }

        except Exception as e:
            self.logger.warning(f"Error processing text with {model_name}: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}

    def analyze_single_text(self, text):
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.sentiment_scores(text, model_name)
        return results

    def load_preprocessed_data(self):
        data_file = self.data_dir / "processed" / "clean_bias_dataset.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {data_file}")

        df = pd.read_csv(data_file)
        self.logger.info(f"Loaded {len(df)} preprocessed samples")
        return df

    def demographic_context(self, row):
        """
        Extracts demographic bias indicators (gender, race, profession, authority) from a dataset row.
        """
        try:
            if isinstance(row['features'], str):
                import ast
                features = ast.literal_eval(row['features'])
            else:
                features = row['features']
        except:
            features = {}

        context = {
            'gender': features.get('genders', []),
            'race': features.get('races', []),
            'profession': features.get('professions', []),
            'authority': features.get('authority_figures', [])
        }

        return context

    def run_bias_analysis(self, df):
        """
        Performs sentiment analysis across all models for each sample in the dataset. Returns a dataframe containing all annotated results.
        """
        self.logger.info("Running bias analysis on dataset")

        results = []
        for i, row in df.iterrows():
            if i % 100 == 0:
                self.logger.info(f"Processing {i}/{len(df)} samples...")

            text = row['text']
            context = self.demographic_context(row)
            sentiment_results = self.analyze_single_text(text)

            result = {
                'id': row['id'],
                'text': text,
                'source': row['source'],
                'period': row['period'],
                'year': row['year'],
                'bias_type': row['bias_type'],
                **context,
                **{f'{model}_sentiment': results['label'] for model, results in sentiment_results.items()},
                **{f'{model}_score': results['score'] for model, results in sentiment_results.items()}
            }

            results.append(result)

        return pd.DataFrame(results)

    def calculate_bias_metrics(self, results_df):
        """
        Computes high-level bias metrics for each model, including intersectional bias, temporal bias, and amplification indices.
        Integrates rule-based and statistical metrics from AdvancedBiasMetrics.
        """
        self.logger.info("Calculating bias metrics:")

        bias_metrics = {}

        for model_name in self.models.keys():
            sentiment_col = f'{model_name}_sentiment'
            score_col = f'{model_name}_score'
            model_metrics = {}

            gender_prof_bias = self.intersection_bias(results_df, 'gender', 'profession', sentiment_col, score_col)
            model_metrics['gender_profession'] = gender_prof_bias

            race_auth_bias = self.intersection_bias(results_df, 'race', 'authority', sentiment_col, score_col)
            model_metrics['race_authority'] = race_auth_bias

            temporal_bias = self.temporal_bias(results_df, sentiment_col, score_col)
            model_metrics['temporal_changes'] = temporal_bias

            temporal_by_community = self.temporal_bias_by_community(results_df, sentiment_col, score_col)
            model_metrics['temporal_by_community'] = temporal_by_community

            amplification = self.advanced_metrics.calculate_stereotype_amplification(gender_prof_bias)
            consistency = self.advanced_metrics.calculate_consistency_index(results_df, score_col)
            effect_sizes = self.advanced_metrics.calculate_effect_sizes(results_df, score_col)

            model_metrics['amplification'] = amplification
            model_metrics['consistency'] = consistency
            model_metrics['effect_sizes'] = effect_sizes

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
                        bias_results[key] = {
                            'sentiments': [],
                            'scores': [],
                            'periods': []
                        }

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

    def temporal_bias(self, df, sentiment_col, score_col):
        """
        Examines how sentiment bias changes across key social periods (pre and post events like metoo or BLM)
        """
        temporal_results = {}

        for period in self.periods.keys():
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

    def temporal_bias_by_community(self, df, sentiment_col, score_col):
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

            community_temporal[community] = {}

            for period in self.periods.keys():
                period_data = community_df[community_df['period'] == period]

                if len(period_data) > 0:
                    community_temporal[community][period] = {
                        'sample_size': len(period_data),
                        'positive_ratio': (period_data[sentiment_col] == 'POSITIVE').sum() / len(period_data),
                        'negative_ratio': (period_data[sentiment_col] == 'NEGATIVE').sum() / len(period_data),
                        'neutral_ratio': (period_data[sentiment_col] == 'NEUTRAL').sum() / len(period_data),
                        'avg_score': period_data[score_col].mean(),
                        'score_std': period_data[score_col].std()
                    }

        return community_temporal

    def bias_report(self, results_df, bias_metrics):
        """
        Generates a structured bias analysis report combining metrics, findings, and recommendations.
        Returns a JSON-compatible dictionary.
        """
        self.logger.info("Creating bias analysis report...")

        report = {
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

        return report

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
                    extreme_bias = []
                    for key, stats in data.items():
                        pos_ratio = stats['positive_ratio']
                        neg_ratio = stats['negative_ratio']
                        if pos_ratio > 0.7 or neg_ratio > 0.7:
                            extreme_bias.append({
                                'intersection': key,
                                'positive_ratio': pos_ratio,
                                'negative_ratio': neg_ratio,
                                'sample_size': stats['sample_size']
                            })

                    if extreme_bias:
                        findings.append({
                            'type': intersection_type,
                            'model': model_name,
                            'extreme_biases': extreme_bias
                        })

        return findings

    def save_results(self, results_df, bias_metrics, report):
        output_dir = self.data_dir / "results"
        output_dir.mkdir(exist_ok=True)

        results_df.to_csv(output_dir / "sentiment_analysis_results.csv", index=False)
        results_df.to_json(output_dir / "sentiment_analysis_results.json", orient='records', indent=2)

        with open(output_dir / "bias_metrics.json", 'w') as f:
            json.dump(bias_metrics, f, indent=2, default=str)

        with open(output_dir / "bias_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Results saved to {output_dir}")

    def complete_analysis(self):
        """
        Executes the entire bias testing pipeline: loads data, runs sentiment analysis, computes metrics, and generates reports.
        Acts as the central entry point for comprehensive model bias evaluation.
        Returns processed results, bias metrics, and the final report dictionary.
        """
        self.logger.info("Testing bias models")

        df = self.load_preprocessed_data()
        results_df = self.run_bias_analysis(df)

        bias_metrics = self.calculate_bias_metrics(results_df)

        report = self.bias_report(results_df, bias_metrics)

        self.save_results(results_df, bias_metrics, report)

        self.logger.info("Analysis over:")
        self.logger.info(f"Analyzed {len(results_df)} samples")
        self.logger.info(f"Models tested: {len(self.models)}")
        self.logger.info(f"Results saved to: {self.data_dir / 'results'}")

        return results_df, bias_metrics, report


def main():
    tester = BiasModelTester()
    results_df, bias_metrics, report = tester.complete_analysis()
    print(f" Results: data/results/sentiment_analysis_results.csv")
    print(f" Bias metrics: data/results/bias_metrics.json")
    print(f" Full report: data/results/bias_analysis_report.json")
    print(f"\n Key findings: {len(report['key_findings'])} bias patterns detected")


if __name__ == "__main__":
    main()
