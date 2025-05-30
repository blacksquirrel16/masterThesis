# masterThesis
Evaluating fairness in Recommender Systems on 5 recommender models in multiple rounds of recommendation

## Datasets are : 
- Modcloth (User feedback on woman clothing fit and quality). ModCloth specializes in women’s vintage clothing and accessories
- RentTheRunway( User ratings and fit feedback on rental clothing) . RentTheRunway is a plattfrom for rented clothing primarily for special occasions

## Fairness Evaluation:
- Disparate Impact (DI) – Ratio of effectiveness between low- and high-activity users.
- Group Recommendation Unfairness (GRU)– Absolute difference in effectiveness across user groups.
- User Coefficient of Variation (UCV) – Variability in recommendation quality within user groups.

## Requirements: 
Python 3.8+ 
pandas, numpy, scikit-learn, matplotlib, LightFM, scipy, tensorflow ( necessary libs are listed in the ipynb with pip so just need to run the cell)

## Recommender models

### ModCloth Data Processing and Collaborative Filtering
modcloth_m.ipynb

Focuses on preprocessing and feature engineering for the ModCloth dataset before feeding into models. And implements Collaborative Filtering user-group based.

### Data Processing of Rent the Runway Data and Collaborative Filtering
renttherunway.ipynb

Focuses on preprocessing and feature engineering for the Rent the Runway user-item interactions. And implements Collaborative Filtering user-group based.

### Content-Based Filtering Modcloth
cb.ipynb

Implements a basic content-based recommendation system using item features such as categories and descriptions.

###  Content-Based Filtering for Rent the Runway
cbrent.ipynb

Adapts the content-based filtering approach to the Rent the Runway dataset.

###  Hybrid Recommendation Model Modcloth
hybrid_mod.ipynb

Combines collaborative filtering with content-based features for the hybrid recommender system for Modcloth.

###  Hybrid Recommendation Model (Script)-> takes the longest time to execute
hybrid_rent.py

Combines collaborative filtering with content-based features for the hybrid recommender system for RentTheRunway.

###  LightFM WARP and BPR with Rent the Runway Data
lightfm_rent.ipynb

Implements LightFM model BPR and WARP on the Rent the Runway dataset

###  LightFM  WARP and BPR on ModCloth Dataset
lightfm.ipynb

Implements LightFM model BPR and WARP on the ModCloth dataset

