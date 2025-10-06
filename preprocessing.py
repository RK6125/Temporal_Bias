
import json
import pandas as pd
import re
import logging
from pathlib import Path
from collections import Counter
import numpy as np
from bias_list import BiasTerms

class BiasDataPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.setup_logging()
 
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.username_pattern = re.compile(r'@\w+|u/\w+|r/\w+')
        self.extra_spaces = re.compile(r'\s+')
        self.special_chars = re.compile(r'[^\w\s.,!?-]')
        self.subreddit_ideology = {
            'politics': 'left',
            'liberal': 'left',
            'progressive': 'left',
            'democrats': 'left',
            'conservative': 'right',
            'republican': 'right',
            'libertarian': 'right',    
            'moderatepolitics': 'center',
            'neutralpolitics': 'center',
            'news': 'mixed',
            'worldnews': 'mixed', 
            'jobs': 'neutral',
            'careerguidance': 'neutral',
            'AskReddit': 'neutral',
            'unpopularopinion': 'mixed'
        }

        self.bias_terms = BiasTerms.get_dict()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = self.url_pattern.sub('', text)
        text = self.username_pattern.sub('', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      
        text = re.sub(r'&gt;.*', '', text)              
        text = re.sub(r'Edit:|UPDATE:|EDIT:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        text = self.extra_spaces.sub(' ', text)
        text = text.strip()
        if len(text) < 10 or len(text) > 2000: 
            return ""  
        return text   
    
    
    def is_bias_relevant(self, text):
    
        text_lower = text.lower()
        has_demo = any(term.lower() in text_lower for term_list in 
                      [self.bias_terms["genders"], self.bias_terms["races"]] 
                      for term in term_list)
        
        has_context = any(term.lower() in text_lower for term_list in 
                         [self.bias_terms["professions"], self.bias_terms["authority_figures"],
                          self.bias_terms["traits"], self.bias_terms["adjectives"], 
                          self.bias_terms["emotions"]] 
                         for term in term_list)
        
        if has_demo and has_context:
          return True
        return False
    
    def extract_bias_features(self, text):
        text_lower = text.lower()
        features = {}        
        features['genders'] = [term for term in self.bias_terms["genders"] if term.lower() in text_lower]
        features['races'] = [term for term in self.bias_terms["races"] if term.lower() in text_lower]
        
        features['professions'] = [term for term in self.bias_terms["professions"] if term.lower() in text_lower]
        features['authority_figures'] = [term for term in self.bias_terms["authority_figures"] if term.lower() in text_lower]
        
        features['traits'] = [term for term in self.bias_terms["traits"] if term.lower() in text_lower]
        features['adjectives'] = [term for term in self.bias_terms["adjectives"] if term.lower() in text_lower]
        features['emotions'] = [term for term in self.bias_terms["emotions"] if term.lower() in text_lower]
        return features  
    def process_reddit_data(self):
        self.logger.info("Processing Reddit posts")        
        reddit_file = self.data_dir / "reddit_posts.csv"
        if not reddit_file.exists():
            self.logger.error(f"Reddit data not found: {reddit_file}")
            return None        
        df = pd.read_csv(reddit_file)
        self.logger.info(f"Loaded {len(df)} Reddit posts")
        processed = []        
        for idx, row in df.iterrows():
            text = row.get('text', '') if isinstance(row, dict) else row['text'] if 'text' in row else ''
            clean_text = self.clean_text(text)
            if clean_text and self.is_bias_relevant(clean_text):
                features = self.extract_bias_features(clean_text)
                
                processed.append({
                    'id': f"reddit_{idx}",
                    'text': clean_text,
                    'original_text': row['text'],
                    'period': row.get('period', 'unknown'),
                    'year': row.get('year', 0),
                    'bias_type': row.get('bias_type', 'unknown'),
                    'subreddit': row.get('subreddit', 'unknown'),
                    'score': row.get('score', 0),
                    'features': features,
                    'source': 'reddit'
                })
        total = len(df) or 1
        self.logger.info(f"Processed {len(processed)} Reddit posts ({len(processed)/total*100:.1f}% kept)")
        return processed    
    def process_template_data(self):
        self.logger.info("Processing template test cases...")  # CORRECT MESSAGE    
        template_file = self.data_dir / "template_test_cases.csv"  # CORRECT FILE
        if not template_file.exists():
            self.logger.error(f"Template data not found: {template_file}")
            return None    
        df = pd.read_csv(template_file)
        self.logger.info(f"Loaded {len(df)} template test cases")    
        processed = []
    
        for idx, row in df.iterrows():
            text = row.get('text', '')
        
            if text and self.is_bias_relevant(text):
                features = self.extract_bias_features(text)
            
                processed.append({
                    'id': f"template_{idx}",
                    'text': text,
                    'original_text': text,
                    'period': row.get('period', 'synthetic'),  # Templates might not have periods
                    'year': row.get('year', 0),
                    'bias_type': row.get('bias_type', 'unknown'),
                    'subreddit': 'template',  # No subreddit for templates
                    'community_lean': 'synthetic',  # Mark as synthetic
                    'score': 0,
                    'features': features,
                    'source': 'template'
            })
    
        self.logger.info(f"Processed {len(processed)} template cases")
        return processed
           
    def create_analysis_dataset(self, reddit_data, template_data):

        self.logger.info("Creating final analysis dataset:")
        all_data = reddit_data + template_data
        df = pd.DataFrame(all_data)
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['has_gender'] = df['features'].apply(lambda x: len(x.get('genders', [])) > 0)
        df['has_race'] = df['features'].apply(lambda x: len(x.get('races', [])) > 0)
        df['has_profession'] = df['features'].apply(lambda x: len(x.get('professions', [])) > 0)
        df['has_authority'] = df['features'].apply(lambda x: len(x.get('authority_figures', [])) > 0)

        quality_stats = {
            'total_samples': len(df),
            'reddit_samples': len(df[df['source'] == 'reddit']),
            'template_samples': len(df[df['source'] == 'template']),
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'period_distribution': df['period'].value_counts().to_dict(),
            'bias_type_distribution': df['bias_type'].value_counts().to_dict()
        }
        
        return df, quality_stats
    
    def diagnose_coverage(self, reddit_data):
   
        if not reddit_data:
            print("No data to diagnose")
            return
        import ast
        df_temp = pd.DataFrame(reddit_data)
        for col in ["has_gender", "has_race", "has_profession", "has_authority"]:
            if col not in df_temp.columns:
                df_temp[col] = False
    
  
        print(f"Raw posts collected: {len(reddit_data)}")
        print(f"\nFeature distribution:")
        print(f"  Has gender: {df_temp['has_gender'].sum()}")
        print(f"  Has race: {df_temp['has_race'].sum()}")
        print(f"  Has profession: {df_temp['has_profession'].sum()}")
        print(f"  Has authority: {df_temp['has_authority'].sum()}")
    
    
        intersections = set()
        for _, row in df_temp.iterrows():
            features = ast.literal_eval(row['features']) if isinstance(row['features'], str) else row['features']
            demos = features.get('genders', []) + features.get('races', [])
            contexts = features.get('professions', []) + features.get('authority_figures', [])
            for d in demos:
                for c in contexts:
                    intersections.add(f"{d}_{c}")
    
        print(f"\nIntersection coverage:")
        print(f"  Total unique intersections: {len(intersections)}")
        print(f"  Sample intersections: {list(intersections)[:10]}")
        print("=" * 40)


    def run_preprocessing(self, diagnose=False):

        self.logger.info("Beginning Data Preprocessing")
        reddit_data = self.process_reddit_data()
        template_data = self.process_template_data()

        if diagnose and reddit_data:
            self.diagnose_coverage(reddit_data)
    
        if not reddit_data and not template_data:
            self.logger.error("No data found after preprocessing filters")
            return None
    
        reddit_data = reddit_data or []
        template_data = template_data or []

        df, stats = self.create_analysis_dataset(reddit_data, template_data)
    
        output_dir = self.data_dir / "processed"
        output_dir.mkdir(exist_ok=True)
    
        df.to_csv(output_dir / "clean_bias_dataset.csv", index=False)
        df.to_json(output_dir / "clean_bias_dataset.json", orient='records', indent=2)
    
        if reddit_data:
            reddit_df = df[df['source'] == 'reddit']
            reddit_df.to_csv(output_dir / "clean_reddit_data.csv", index=False)
    
        if template_data:
            template_df = df[df['source'] == 'template']
            template_df.to_csv(output_dir / "clean_template_data.csv", index=False)
    
        with open(output_dir / "preprocessing_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info("Preprocessing complete")
        self.logger.info(f"Total samples: {stats['total_samples']}")
        self.logger.info(f"Reddit: {stats['reddit_samples']}, Templates: {stats['template_samples']}")
        self.logger.info(f"Avg length: {stats['avg_text_length']:.0f} chars, {stats['avg_word_count']:.0f} words")
        self.logger.info(f"Saved to: {output_dir}")
    
        return df, stats
def main():
    preprocessor = BiasDataPreprocessor()
    df, stats = preprocessor.run_preprocessing() # type: ignore
    if df is not None:
        print("\nPreprocessing complete!")
        print(f"Clean dataset: data/processed/clean_bias_dataset.csv")
        print(f"Ready for sentiment analysis!")
    else:
        print(" Preprocessing failed!")
if __name__ == "__main__":
    main()