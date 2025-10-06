
import os
import json
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import logging

from collect_reddit import RedditBiasCollector
from reference import TestCaseGenerator

class BiasResearchDataCollector:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()
        
        self.periods = {
            "pre_metoo": 2017,
            "post_metoo": 2019, 
            "covid_era": 2021,
            "post_blm": 2023
        }
        
        self.search_terms = {
            "gender_profession": [
                ["woman doctor", "female doctor"],
                ["man nurse", "male nurse"], 
                ["woman engineer", "female engineer"],
                ["woman lawyer", "female lawyer"],
                ["man teacher", "male teacher"]
            ],
            "race_authority": [
                ["black manager", "african american manager"],
                ["asian ceo", "asian director"],
                ["latino supervisor", "hispanic supervisor"],
                ["white judge", "caucasian judge"]
            ]
        }
        
        self.reddit_collector = RedditBiasCollector()
        self.test_generator = TestCaseGenerator()
    
    def setup_directories(self):

        dirs = [
            "raw/reddit",
            "processed", 
            "templates",
            "results",
            "logs"
        ]
        for dir_path in dirs:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        log_file = self.base_dir / "logs" / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_template_test_cases(self):
        self.logger.info("Generating template-based test cases...")
        
        from reference import auto_generate_test_cases
        test_cases = auto_generate_test_cases(target_count=200)
        
        output_file = self.base_dir / "templates" / "systematic_test_cases.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(test_cases)} test cases saved to {output_file}")
        return test_cases
    
    def collect_reddit_data_for_period(self, period_name, year, posts_per_term=50):
        self.logger.info(f"Collecting Reddit data for {period_name} ({year})")
        
        period_data = []
        
        for bias_type, term_groups in self.search_terms.items():
            self.logger.info(f"  Collecting {bias_type} terms...")
            
            for terms in term_groups:
                try:
                    filename = self.reddit_collector.collect_data(terms, year, limit=posts_per_term)
                    
                    if filename:
                        with open(filename, 'r', encoding='utf-8') as f:
                            posts = json.load(f)
                        
                        for post in posts:
                            post['period'] = period_name
                            post['year'] = year
                            post['bias_type'] = bias_type
                            post['search_terms'] = terms
                        
                        period_data.extend(posts)
                        self.logger.info(f"    Collected {len(posts)} posts for {terms}")
                
                    time.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"    Failed to collect {terms}: {e}")
                    continue

        if period_data:
            output_file = self.base_dir / "processed" / f"reddit_{period_name}_{year}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(period_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"  Period total: {len(period_data)} posts saved to {output_file}")
        
        return period_data
    
    def collect_all_periods(self, posts_per_term=50):
        self.logger.info("Starting complete data collection...")
        self.logger.info(f"Target: {posts_per_term} posts per search term, per period")
        
        all_data = {}
        
        for period_name, year in self.periods.items():
            period_data = self.collect_reddit_data_for_period(period_name, year, posts_per_term)
            all_data[period_name] = period_data
            
            total_collected = sum(len(data) for data in all_data.values())
            self.logger.info(f"Progress: {len(all_data)}/{len(self.periods)} periods complete, {total_collected} total posts")
        
        return all_data
    
    def create_analysis_ready_dataset(self, all_data, test_cases):
        self.logger.info("Creating analysis-ready dataset...")
        
        reddit_posts = []
        for period_name, posts in all_data.items():
            reddit_posts.extend(posts)
        
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_reddit_posts": len(reddit_posts),
                "total_test_cases": len(test_cases),
                "periods": list(self.periods.keys()),
                "bias_types": list(self.search_terms.keys())
            },
            "reddit_data": reddit_posts,
            "template_test_cases": test_cases,
            "periods": self.periods,
            "search_terms": self.search_terms
        }
        
        output_file = self.base_dir / "bias_research_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        if reddit_posts:
            df_reddit = pd.DataFrame(reddit_posts)
            df_reddit.to_csv(self.base_dir / "reddit_posts.csv", index=False)
        
        if test_cases:
            df_tests = pd.DataFrame(test_cases)
            df_tests.to_csv(self.base_dir / "template_test_cases.csv", index=False)
        
        self.logger.info(f"Analysis-ready dataset saved to {output_file}")
        return dataset
    
    def run_complete_collection(self, posts_per_term=50):

        start_time = datetime.now()
        self.logger.info("="*60)
        self.logger.info("STARTING BIAS RESEARCH DATA COLLECTION")
        self.logger.info("="*60)
        
        try:
        
            test_cases = self.generate_template_test_cases()
            

            all_data = self.collect_all_periods(posts_per_term)
            
            
            dataset = self.create_analysis_ready_dataset(all_data, test_cases)
            
        
            end_time = datetime.now()
            duration = end_time - start_time
            
            total_posts = sum(len(posts) for posts in all_data.values())
            
            self.logger.info("="*60)
            self.logger.info("COLLECTION COMPLETE")
            self.logger.info("="*60)
            self.logger.info(f"Duration: {duration}")
            self.logger.info(f"Template test cases: {len(test_cases)}")
            self.logger.info(f"Reddit posts collected: {total_posts}")
            self.logger.info(f"Periods covered: {len(self.periods)}")
            self.logger.info(f"Data saved to: {self.base_dir}")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Collection failed: {e}")
            raise


def main():
    
    collector = BiasResearchDataCollector()
    dataset = collector.run_complete_collection(posts_per_term=50)
    print(f"Main dataset: data/bias_research_dataset.json")
    print(f"CSV files: data/reddit_posts.csv, data/template_test_cases.csv")

if __name__ == "__main__":
    main()