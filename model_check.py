import pandas as pd
import numpy as np
import json
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class Validator:
    """
    Validates the temporal stability of detected bias patterns. Cross-checks transformer model results with alternative 
    ML baselines to ensure observed bias trends are not model specific.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.validation_dir = self.results_dir / "validation"
        self.validation_dir.mkdir(exist_ok=True)

        self.periods = ["pre_metoo", "post_metoo", "covid_era", "post_blm"]

    def load_existing_results(self):
        metrics_file = self.results_dir / "bias_metrics.json"
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        print(f"Loaded bias metrics from existing models")
        return metrics

    def extract_trends(self, metrics, model_name):
        """
        Extracts the temporal sentiment ratios for a given model from previously stored bias metrics.
        Returns a dictionary mapping period to sentiment ratio.
        """
        temporal_data = metrics[model_name].get('temporal_changes', {})
        trend = {}
        for period in self.periods:
            if period in temporal_data:
                trend[period] = temporal_data[period]['positive_ratio']
        return trend

    def alternative_models(self, df):
        """
        Trains Random Forest and XGBoost baselines on text data to verify whether temporal bias trends generalize beyond 
        transformer models.
        """
        print("\nTraining alternative validation models...")

        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['text'])

        sentiment_cols = [col for col in df.columns if col.endswith('_sentiment')]
        df['majority_sentiment'] = df[sentiment_cols].mode(axis=1)[0]
        y = (df['majority_sentiment'] == 'POSITIVE').astype(int)

        alternative_results = {}

        print("  Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        df['rf_prediction'] = rf_model.predict(X)
        alternative_results['random_forest'] = self.temporal_from_predictions(df, 'rf_prediction')

        print("  Training XGBoost...")
        xgb_model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        xgb_model.fit(X, y)
        df['xgb_prediction'] = xgb_model.predict(X)
        alternative_results['xgboost'] = self.temporal_from_predictions(df, 'xgb_prediction')

        return alternative_results, df

    def temporal_from_predictions(self, df, pred_col):
        """
        Computes period-wise positive sentiment ratios from model predictions, used by Random Forest and XGBoost baselines 
        for temporal comparison.
        
        """
        temporal_trend = {}
        for period in self.periods:
            period_data = df[df['period'] == period]
            if len(period_data) > 0:
                positive_ratio = period_data[pred_col].mean()
                temporal_trend[period] = positive_ratio
        return temporal_trend

    def consistency(self, all_trends):
        """
        Evaluates agreement between temporal bias trends of all models using Spearman correlation and standard deviation based 
        agreement metrics.
        """
        print("\nCalculating temporal consistency metrics...")

        consistency_results = {
            'pairwise_correlations': {},
            'period_agreements': {},
            'overall_consistency': None
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
                    'correlation': corr,
                    'p_value': p_value,
                    'n_periods': len(common_periods)
                }
                correlations.append(corr)

        if correlations:
            consistency_results['overall_consistency'] = np.mean(correlations)

        for period in self.periods:
            period_values = []
            for model, trend in all_trends.items():
                if period in trend:
                    period_values.append(trend[period])

            if len(period_values) >= 2:
                consistency_results['period_agreements'][period] = {
                    'std_dev': np.std(period_values),
                    'range': max(period_values) - min(period_values),
                    'coefficient_of_variation': np.std(period_values) / np.mean(period_values)
                    if np.mean(period_values) > 0 else float('inf')
                }

        return consistency_results

    def sus_periods(self, consistency_results, threshold=0.15):
        """
        Flags periods with high inter-model disagreement where standard deviation > threshold
        
        """
        sus = []
        for period, stats in consistency_results['period_agreements'].items():
            if stats['std_dev'] > threshold:
                sus.append({
                    'period': period,
                    'disagreement_level': stats['std_dev'],
                    'interpretation': 'High model disagreement - results may be measurement artifacts'
                })
        return sus

    def validation_report(self, consistency_results, sus_periods):
        """
        Compiles the validation results into a report including summary scores and pairwise correlations
        """
        report = {
            'validation_summary': {
                'overall_consistency_score': consistency_results['overall_consistency'],
                'interpretation': self.interpret_consistency(consistency_results['overall_consistency'])
            },
            'pairwise_correlations': consistency_results['pairwise_correlations'],
            'period_level_analysis': consistency_results['period_agreements'],
            'sus_periods': sus_periods
        }
        return report

    def interpret_consistency(self, score):
    
        if score is None:
            return "Unable to calculate due to insufficient data"
        elif score > 0.8:
            return "High confidence - temporal trends are robust across model architectures"
        elif score > 0.6:
            return "Moderate confidence - trends show general agreement but some variation"
        elif score > 0.4:
            return "Low confidence - significant model-dependent variation in trends"
        else:
            return "Caution advised - temporal trends may be measurement artifacts rather than real bias shifts"

    
    def visualizations(self, all_trends, consistency_results):
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Temporal Bias Robustness Validation', fontsize=14, fontweight='bold')

        for model_name, trend in all_trends.items():
            periods = sorted(trend.keys(), key=lambda x: self.periods.index(x))
            values = [trend[p] for p in periods]
            x = [self.periods.index(p) for p in periods]
            ax1.plot(x, values, 'o-', label=model_name, linewidth=2, markersize=6, alpha=0.7)

        ax1.set_title('Model Agreement on Temporal Trends')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Positive Sentiment Ratio')
        ax1.set_xticks(range(len(self.periods)))
        ax1.set_xticklabels(['Pre-#MeToo\n(2017)', 'Post-#MeToo\n(2019)',
                             'COVID Era\n(2021)', 'Post-BLM\n(2023)'], rotation=45)
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
        plt.savefig(self.validation_dir / 'temporal_robustness_validation.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def run_validation(self):
        """
        Loads results, computes consistency metrics, identifies instability,
        generates visualizations, and saves a final validation report.
        """

        metrics = self.load_existing_results()

        all_trends = {}
        for model in ['bert', 'distilbert', 'vader_style', 'textblob']:
            if model in metrics:
                all_trends[model] = self.extract_trends(metrics, model)

        df = pd.read_csv(self.data_dir / "results" / "sentiment_analysis_results.csv")

        alternative_trends, df_with_predictions = self.alternative_models(df)
        all_trends.update(alternative_trends)

        consistency_results = self.consistency(all_trends)
        sus_periods = self.sus_periods(consistency_results)
        report = self.validation_report(consistency_results, sus_periods)
        self.visualizations(all_trends, consistency_results)

        with open(self.validation_dir / 'robustness_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("Validation done")
        print(f"\nOverall Consistency Score: {report['validation_summary']['overall_consistency_score']:.3f}")
        print(f"Interpretation: {report['validation_summary']['interpretation']}")

        if sus_periods:
            print(f"\n Suspicious Periods Detected: {len(sus_periods)}")
            for sp in sus_periods:
                print(f"  - {sp['period']}: Disagreement = {sp['disagreement_level']:.3f}")
        else:
            print("\nNo suspicious periods detected")

        print(f"\nDetailed results saved to: {self.validation_dir}")
        return report


def main():
    """
    Entry point for standalone execution of temporal robustness validation.
    Loads prior model outputs and produces consistency reports and plots.
    """
    validator = Validator()
    report = validator.run_validation()
    print("\n Validation complete")


if __name__ == "__main__":
    main()
