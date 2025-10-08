import datetime
import json
import os
import time
import praw
import re
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")

class RedditCollector:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD")

        )
        from bias_list import BiasTerms

        self.bias_terms = BiasTerms.get_dict()
        
        # List of 20 subreddits likely to contain posts about professional or demographic bias. Can be edited later
        self.subreddits = ['jobs', 'AskReddit', 'TwoXChromosomes', 'medicine', 'nursing', 
                          'engineering', 'teachers', 'WorkReform', 'careerguidance',  'cscareerquestions','Accounting',         
                            'consulting', 'LawSchool', 'lawyers', 'blackladies', 'blackfellas', 'asianamerican',  'Hispanic', 'MensRights', 'feminism']
    
    def bias_relevance(self, text):
        
        text_lower = text.lower()
        
        has_gender = any(term.lower() in text_lower for term in self.bias_terms["genders"])
        has_race = any(term.lower() in text_lower for term in self.bias_terms["races"])
        has_profession = any(term.lower() in text_lower for term in self.bias_terms["professions"])
        has_authority = any(term.lower() in text_lower for term in self.bias_terms["authority_figures"])
        has_leadership = any(term.lower() in text_lower for term in self.bias_terms["leader_roles"])
        demographic = has_gender or has_race
        professional = has_profession or has_authority or has_leadership
       
        return demographic and professional and len(text) > 50
    
    def collect_data(self, keywords, year, limit=1000):
        
        start_time = int(datetime.datetime(year, 1, 1).timestamp())
        end_time = int(datetime.datetime(year, 12, 31).timestamp())
        
        posts = []
        keywords = [keywords] if isinstance(keywords, str) else keywords
        
        for subreddit_name in self.subreddits:
            if len(posts) >= limit:
                break
                
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for keyword in keywords:
                    search_results = subreddit.search(keyword, sort='top', time_filter='all', limit=100)
                    
                    for post in search_results:
                        if len(posts) >= limit:
                            break
                        if not (start_time <= post.created_utc <= end_time):
                            continue
                            
                        full_text = f"{post.title} {post.selftext}".strip()
                        
                        if self.bias_relevance(full_text):
                            posts.append({
                                "text": full_text,
                                "title": post.title,
                                "subreddit": post.subreddit.display_name,
                                "timestamp": int(post.created_utc),
                                "score": post.score,
                                "keyword": keyword
                            })
                
                time.sleep(1)  
            except Exception as e:
                print(f"Error with r/{subreddit_name}: {e}")
                continue
        if posts:
            unique_posts = self.deduplicate(posts)
            
            os.makedirs("data/raw/reddit", exist_ok=True)
            filename = f"data/raw/reddit/{'_'.join(keywords)}_{year}.json"
            
            with open(filename, "w", encoding='utf-8') as f:
                json.dump(unique_posts[:limit], f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(unique_posts)} posts to {filename}")
            return filename
        
        print(" No relevant posts found")
        return None
    
    def deduplicate(self, posts):
    # Removes seemingly duplicate posts based on the first 100 characters of text.
        seen = set()
        unique = []
        
        for post in posts:
            first100 = post['text'][:100].lower().strip()
            if first100 not in seen and len(first100) > 50:
                seen.add(first100)
                unique.append(post)
        
        return unique

def collect_reddit_data(keyword, year, limit=1000):
    collector = RedditCollector()
    return collector.collect_data(keyword, year, limit)



        
