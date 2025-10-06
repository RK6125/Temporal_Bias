
#Temporal Bias Analysis In Reddit Discourse (2017–2023)

This project analyzes shifts in social bias across major social movements from Reddit posts using multiple sentiment models (BERT, DistilBERT, VADER, TextBlob).

---

##  Project Overview
- **Objective:** Measure how bias language changes before and after major social events (#MeToo, COVID, BLM).
- **Data:** 6,000 Reddit posts which were preprocessed to 2,600 usable samples.
- **Models Tested:** BERT, DistilBERT, VADER, TextBlob.
- **Output:** Temporal bias visualizations, intersectional heatmaps, and validation metrics.

---

##  How to Run
1. Clone the repo  
   ```bash
   git clone https://github.com/<your-username>/Temporal-Bias-Analysis.git
   cd Temporal-Bias-Analysis

2. pip install -r requirements.txt

3. python main.py
