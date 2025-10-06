
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
import warnings
warnings.filterwarnings('ignore')

class BiasResultsAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        self.periods = ["pre_metoo", "post_metoo", "covid_era", "post_blm"]
        self.period_labels = ["Pre-#MeToo\n(2017)", "Post-#MeToo\n(2019)", "COVID Era\n(2021)", "Post-BLM\n(2023)"]
        
    def load_results(self):
  
        results_file = self.results_dir / "sentiment_analysis_results.csv"
        metrics_file = self.results_dir / "bias_metrics.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")
        
        df = pd.read_csv(results_file)
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"Loaded {len(df)} analyzed samples")
        return df, metrics
    
    def create_temporal_bias_plots(self, df, metrics):
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Bias Changes Across Models (2017-2023)', fontsize=16, fontweight='bold')
        
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        
        for idx, model in enumerate(models):
            ax = axes[idx//2, idx%2]
            
            # Get temporal data for this model
            temporal_data = []
            
            for period in self.periods:
                period_data = df[df['period'] == period]
                if len(period_data) > 0:
                    sentiment_col = f'{model}_sentiment'
                    pos_ratio = (period_data[sentiment_col] == 'POSITIVE').sum() / len(period_data)
                    neg_ratio = (period_data[sentiment_col] == 'NEGATIVE').sum() / len(period_data)
                    
                    temporal_data.append({
                        'period': period,
                        'positive_ratio': pos_ratio,
                        'negative_ratio': neg_ratio,
                        'sample_size': len(period_data)
                    })
            
            if temporal_data:
                temp_df = pd.DataFrame(temporal_data)
   
                x = range(len(self.period_labels))
                ax.plot(x, temp_df['positive_ratio'], 'o-', label='Positive', linewidth=2, markersize=8)
                ax.plot(x, temp_df['negative_ratio'], 'o-', label='Negative', linewidth=2, markersize=8)
                
                ax.set_title(f'{model.upper()} Model', fontweight='bold')
                ax.set_xlabel('Time Period')
                ax.set_ylabel('Sentiment Ratio')
                ax.set_xticks(x)
                ax.set_xticklabels(self.period_labels, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'temporal_bias_changes.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_community_segmented_plots(self, df, metrics):

        models = ['bert', 'distilbert', 'vader_style', 'textblob']
    
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Community-Segmented Temporal Bias Analysis', 
                 fontsize=16, fontweight='bold')
    
        colors = {
        'left': '#0066CC',
        'right': '#CC0000',
        'center': '#9933FF',
        'mixed': '#FF9900',
        'neutral': '#666666'
    }
    
        for idx, model in enumerate(models):
            ax = axes[idx//2, idx%2]

            if 'temporal_by_community' in metrics.get(model, {}):
                community_data = metrics[model]['temporal_by_community']
            
                for community, periods_data in community_data.items():
                    if len(periods_data) < 2:  
                        continue

                    x_vals = []
                    y_vals = []
                
                    for i, period in enumerate(self.periods):
                        if period in periods_data:
                            x_vals.append(i)
                            y_vals.append(periods_data[period]['positive_ratio'])
                
                    if len(x_vals) >= 2:
                        ax.plot(x_vals, y_vals, 'o-', 
                           label=community.capitalize(),
                           color=colors.get(community, '#000000'),
                           linewidth=2, markersize=6, alpha=0.8)
        
            ax.set_title(f'{model.upper()} - Community Divergence', fontweight='bold')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Positive Sentiment Ratio')
            ax.set_xticks(range(len(self.period_labels)))
            ax.set_xticklabels(self.period_labels, rotation=45)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
    
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'community_segmented_temporal.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_intersection_heatmap(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bias Heatmaps: Sentiment Ratios by Demographic×Professional Intersections', 
                    fontsize=14, fontweight='bold')
        
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        
        for idx, model in enumerate(models):
            ax = axes[idx//2, idx%2]
    
            intersection_data = self._calculate_intersection_matrix(df, model)
            
            if not intersection_data.empty:
                sns.heatmap(intersection_data, annot=True, cmap='RdYlBu_r', center=0.5,
                           ax=ax, fmt='.2f', cbar_kws={'label': 'Positive Sentiment Ratio'})
                ax.set_title(f'{model.upper()} Model')
                ax.set_xlabel('Professional Context')
                ax.set_ylabel('Demographic Group')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'intersection_bias_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_intersection_matrix(self, df, model):
        sentiment_col = f'{model}_sentiment'
        
        intersections = {}
        
        for _, row in df.iterrows():
            try:
               
                if isinstance(row.get('gender'), str) and row['gender'].startswith('['):
                    genders = eval(row['gender'])
                    races = eval(row['race']) if isinstance(row.get('race'), str) else []
                    professions = eval(row['profession']) if isinstance(row.get('profession'), str) else []
                    authorities = eval(row['authority']) if isinstance(row.get('authority'), str) else []
                else:
                    genders = [row.get('gender')] if row.get('gender') else []
                    races = [row.get('race')] if row.get('race') else []
                    professions = [row.get('profession')] if row.get('profession') else []
                    authorities = [row.get('authority')] if row.get('authority') else []
         
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
    
    def create_model_comparison_plot(self, df):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Comparison: Bias Consistency Analysis', fontsize=14, fontweight='bold')
        
        models = ['bert', 'distilbert', 'vader_style', 'textblob']

        sentiment_data = []
        for model in models:
            sentiment_col = f'{model}_sentiment'
            model_sentiments = df[sentiment_col].value_counts(normalize=True)
            
            for sentiment, ratio in model_sentiments.items():
                sentiment_data.append({
                    'model': model.upper(),
                    'sentiment': sentiment,
                    'ratio': ratio
                })
        
        sent_df = pd.DataFrame(sentiment_data)
        sns.barplot(data=sent_df, x='model', y='ratio', hue='sentiment', ax=ax1)
        ax1.set_title('Sentiment Distribution by Model')
        ax1.set_ylabel('Ratio')
        ax1.legend(title='Sentiment')
        agreement_data = self._calculate_model_agreement(df, models)
        ax2.bar(range(len(agreement_data)), list(agreement_data.values()))
        ax2.set_title('Model Agreement Rates')
        ax2.set_xlabel('Agreement Type')
        ax2.set_ylabel('Percentage')
        ax2.set_xticks(range(len(agreement_data)))
        ax2.set_xticklabels(list(agreement_data.keys()), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_model_agreement(self, df, models):

        agreement = {
            'All Agree': 0,
            'Majority Positive': 0,
            'Majority Negative': 0,
            'Complete Disagreement': 0
        }
        
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
    
    def create_statistical_summary(self, df, metrics):
        """Create statistical analysis summary"""
        print("\n" + "="*60)
        print("STATISTICAL BIAS ANALYSIS SUMMARY")
        print("="*60)

        print(f"\nSample Sizes:")
        print(f"Total samples: {len(df)}")
        print(f"Template samples: {len(df[df['source'] == 'template'])}")
        print(f"Reddit samples: {len(df[df['source'] == 'reddit'])}")

        print(f"\nPeriod Distribution:")
        period_counts = df['period'].value_counts()
        for period, count in period_counts.items():
            print(f"  {period}: {count} samples")

        self._run_statistical_tests(df)

        self._calculate_effect_sizes(df)
        
    def _run_statistical_tests(self, df):
       
        print(f"\nStatistical Significance Tests:")
        print(f"-" * 40)
        
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        
        for model in models:
            sentiment_col = f'{model}_sentiment'
           
            if len(df[df['period'].isin(self.periods)]) > 0:
                period_sentiment_table = pd.crosstab(df['period'], df[sentiment_col])
                
                if period_sentiment_table.shape[0] > 1 and period_sentiment_table.shape[1] > 1:
                    chi2_stat, p_val, degrees_of_freedom, expected_freq = chi2_contingency(period_sentiment_table)  # type: ignore
                    p_val : float = p_val
                    print(f"\n{model.upper()} Model:")
                    print(f" Chi-square test (period vs sentiment): χ² = {chi2_stat:.3f}, p = {p_val:.3f}")
            if p_val < 0.05:
                print(f"  Significant bias change over time (p < 0.05)")
            else:
                print(f" No significant temporal bias change")
    
    def _calculate_effect_sizes(self, df):
        print(f"\nEffect Size Analysis:")
        print(f"-" * 40)
        
        models = ['bert', 'distilbert', 'vader_style', 'textblob']
        
        for model in models:
            score_col = f'{model}_score'
            
            # Compare pre-MeToo vs post-BLM periods
            pre_metoo = df[df['period'] == 'pre_metoo'][score_col]
            post_blm = df[df['period'] == 'post_blm'][score_col]
            
            if len(pre_metoo) > 0 and len(post_blm) > 0:
                # Cohen's d effect size
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
    
    def generate_research_summary(self, df, metrics):
        summary = {
            "research_overview": {
                "title": "Temporal Bias Analysis in Sentiment Models (2017-2023)",
                "periods_analyzed": ["Pre-#MeToo (2017)", "Post-#MeToo (2019)", "COVID Era (2021)", "Post-BLM (2023)"],
                "total_samples": len(df),
                "models_tested": ["BERT", "DistilBERT", "RoBERTa", "TextBlob"],
                "bias_intersections": ["Gender×Profession", "Race×Authority"]
            },
            "key_findings": self._extract_key_research_findings(df),
            "methodology": {
                "template_test_cases": len(df[df['source'] == 'template']),
                "real_world_samples": len(df[df['source'] == 'reddit']),
                "statistical_tests": ["Chi-square independence test", "Cohen's d effect size"],
                "bias_metrics": ["Sentiment ratios", "Intersection analysis", "Temporal trends"]
            }
        }
        
        # Save summary
        with open(self.results_dir / "research_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _extract_key_research_findings(self, df):
        findings = []  
        return findings
    
    def run_complete_analysis(self):
        df, metrics = self.load_results()
     
        print("\nTemporal bias plots")
        self.create_temporal_bias_plots(df, metrics)
        
        if 'community_lean' in df.columns:
            print("Cmmunity-segmented plots")
            self.create_community_segmented_plots(df, metrics)

        print("Intersection heatmaps")
        self.create_intersection_heatmap(df)
        
        print("Model comparison plots")
        self.create_model_comparison_plot(df)

        self.create_statistical_summary(df, metrics)
    
        summary = self.generate_research_summary(df, metrics)
        print(f" Visualizations saved to: {self.viz_dir}")
        print(f" Research summary: {self.results_dir / 'research_summary.json'}")
        
        return df, metrics, summary

def main():
    analyzer = BiasResultsAnalyzer()
    df, metrics, summary = analyzer.run_complete_analysis()
    
    print("\n Complete")

if __name__ == "__main__":
    main()