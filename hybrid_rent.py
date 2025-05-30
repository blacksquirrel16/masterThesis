#!/usr/bin/env python3

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import torch
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


file_path = 'renttherunway_final_data.json'
df_rent = pd.read_json(file_path, lines=True)

###############################################
# 3. Explicit Feedback
###############################################
df_rent_clean = df_rent.dropna(subset=["user_id", "item_id", "rating","age"]).drop_duplicates()
df_rent_clean['interaction'] = df_rent_clean['rating']
def calculate_sparsity(df):
    """Calculates sparsity for each numerical feature in the DataFrame."""
    numerical_features = df.select_dtypes(include=np.number).columns
    sparsity_results = {}

    for feature in numerical_features:
        sparsity = df[feature].isnull().sum() / len(df)
        sparsity_results[feature] = sparsity

    return sparsity_results

sparsity_dict = calculate_sparsity(df_rent_clean)
print(sparsity_dict)

def preprocess_data(df, sparsity_threshold=0.50):
    """Preprocesses the data by removing numerical features with high sparsity."""
    for col in df.select_dtypes(include=np.number).columns:
        sparsity = df[col].isnull().sum() / len(df)
        if sparsity > sparsity_threshold:
            print(f"Dropping sparse feature: {col} (Sparsity: {sparsity:.2f})")
            df = df.drop(col, axis=1)
    return df
df_rent_clean = preprocess_data(df_rent_clean, sparsity_threshold=0.50)


user_interaction_counts = df_rent_clean.groupby('user_id')['rating'].count()
users_with_enough_interactions = user_interaction_counts[user_interaction_counts >= 2].index
df_rent_clean = df_rent_clean[df_rent_clean["user_id"].isin(users_with_enough_interactions)]
print(f"Number of users with more than 2 interactions: {len(users_with_enough_interactions)}")

user_activity = df_rent_clean.groupby('user_id')['interaction'].sum()

vectorizer = CountVectorizer()
category_bow = vectorizer.fit_transform(df_rent_clean['category'])
###############################################
# 3. User Engagement Levels
###############################################

median_interaction_strength = df_rent_clean.groupby('user_id')['interaction'].sum().median()

df_rent_clean['user_activity'] = df_rent_clean.groupby('user_id')['interaction'].transform('sum').map(
    lambda x: 'Low' if x < median_interaction_strength else 'High'
)

print(df_rent_clean['user_activity'].value_counts())

df_rent_clean = df_rent_clean.sort_values('review_date')

### SPLIT DATA
train_size = int(0.6 * len(df_rent_clean)) # 60 %
val_size = int(0.2 * len(df_rent_clean))

train_df = df_rent_clean[:train_size]
val_df = df_rent_clean[train_size:train_size + val_size]
test_df = df_rent_clean[train_size + val_size:]


print("Train set shape:", train_df.shape)
print("Validation set shape:", val_df.shape)
print("Test set shape:", test_df.shape)

###############################################
# 3. User Item Matrix -Train
###############################################

train_user_item_matrix = train_df.pivot_table(
    index="user_id",
    columns="item_id",
    values="interaction",
    aggfunc="sum",
    fill_value=0
)
###############################################
# 3. Utility/Ranking metrics
###############################################
def dcg_at_k_recursive(relevance_scores, k, b=2):
    """
    Calculate the Discounted Cumulative Gain at rank k recursively.

    :param relevance_scores: A list or array of relevance scores for the ranked items.
    :param k: The rank at which to stop (k items to evaluate).
    :param b: The base of the logarithm (typically 2).
    :return: DCG at rank k.
    """
    k = min(k, len(relevance_scores))

    if k == 0:
        return 0.0

    if k < b:
        return np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k])])


    else:
        dcg_k_minus_1 = dcg_at_k_recursive(relevance_scores, k - 1, b)
        rel_k = relevance_scores[k - 1]
        return dcg_k_minus_1 + (rel_k / np.log2(k + 1))

def dcg_at_k(relevance_scores, k, b=2):
    """
    Calculate the Discounted Cumulative Gain at rank k (non-recursive).
    """
    dcg = 0
    for idx, rel in enumerate(relevance_scores[:k]):
        dcg += rel / np.log2(idx + 2)
    return dcg
def idcg_at_k(relevance_scores, k):
    """
    Calculate the Ideal Discounted Cumulative Gain at rank k.
    :param relevance_scores: A list or array of relevance scores for the ranked items.
    :param k: The rank at which to stop (k items to evaluate).
    :return: Ideal DCG at rank k.
    """
    relevance_scores = sorted(relevance_scores, reverse=True)
    return dcg_at_k(relevance_scores, k)

def ndcg_at_k(relevance_scores, k):
    """
    Calculate the Normalized Discounted Cumulative Gain at rank k.
    :param relevance_scores: A list or array of relevance scores for the ranked items.
    :param k: The rank at which to stop (k items to evaluate).
    :return: nDCG at rank k.
    """
    dcg = dcg_at_k_recursive(relevance_scores, k)
    idcg = idcg_at_k(relevance_scores, k)
    if idcg == 0:
        return 0
    return dcg / idcg

def hr_at_k(actual, predicted, k=5):
    """
    Computes Hit Rate (HR) at rank k.

    Args:
        actual: list of relevant items
        predicted: ranked list of items
        k: rank cutoff (default is 5)

    Returns:
        Hit Rate at rank k.
    """

    for item in predicted[:k]:
        if item in actual:
            return 1
    return 0


def hit_rate_at_k(actual, predicted, k=5):
    """
    Computes Mean Hit Rate (HR) at rank k for all users.

    Args:
        actual: list of relevant items for each user
        predicted: list of predicted ranked items for each user
        k: rank cutoff (default is 5)

    Returns:
        HR at rank k.
    """

    if not actual or len(predicted) == 0:
        return 0.0


    hr_scores = []
    for user_actual, user_predicted in zip(actual, predicted):
        hr = hr_at_k(user_actual, user_predicted, k)
        hr_scores.append(hr)


    return np.mean(hr_scores)

def rr_at_k(actual, predicted, k=5):
    """
    Computes Reciprocal Rank at rank k.

    Args:
        actual: the relevant item(s)
        predicted: the ranked list of items
        k: rank cutoff (default is 5)

    Returns:
        Reciprocal Rank at rank k.
    """

    for i, item in enumerate(predicted[:k]):
        if item == actual:
            return 1 / (i + 1)
    return 0.0


def mrr_at_k(actual, predicted, k=5):
    """
    Computes Mean Reciprocal Rank (MRR) at rank k.

    Args:
        actual: list of relevant items (can be multiple)
        predicted: ranked list of items
        k: rank cutoff (default is 5)

    Returns:
        MRR at rank k.
    """

    if actual is None or len(predicted) == 0:
        return 0.0


    rr_scores = []
    for item in actual:
        rr = rr_at_k(item, predicted, k)
        rr_scores.append(rr)

    return np.mean(rr_scores)



def calculate_disparate_impact(protected_outcomes, privileged_outcomes):
    """
    Args:
        protected_outcomes: List of binary outcomes (1=favorable) for the protected group.
        privileged_outcomes: List of binary outcomes for the privileged group.
    Returns:
        Disparate impact ratio.
    """
    protected_rate = np.mean(protected_outcomes)
    privileged_rate = np.mean(privileged_outcomes)

    if privileged_rate == 0:
        return np.inf

    return protected_rate / privileged_rate


def calculate_group_recommender_unfairness(group1_metrics, group2_metrics):
  """
    Calculates the absolute difference in mean metrics between two groups.
    This metric quantifies the unfairness of a recommender system by examining
    the absolute difference in average performance between different user groups.

    Args:
        group1_metrics (list or numpy.ndarray): A list or numpy array of metrics for group 1.
        group2_metrics (list or numpy.ndarray): A list or numpy array of metrics for group 2.

    Returns:
        float: The absolute difference in mean metrics between the two groups.
  """
  return np.abs(np.mean(group1_metrics) - np.mean(group2_metrics))


def mean_absolute_difference(group1_metrics, group2_metrics):
    """
    Calculates the mean absolute difference between two groups of metrics.

    Args:
        group1_metrics: A list or numpy array of metrics for group 1.
        group2_metrics: A list or numpy array of metrics for group 2.

    Returns:
        The mean absolute difference.
    """
    return np.mean(np.abs(np.array(group1_metrics) - np.array(group2_metrics)))

def mean_difference(group1_metrics, group2_metrics):
    return np.abs(np.mean(group1_metrics) - np.mean(group2_metrics))

def mean_absolute_deviation(metrics):
    return np.mean(np.abs(metrics - np.mean(metrics)))

def coefficient_of_variation(arr):
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 0
    return np.std(arr) / mean_val

def calculate_ucv(metric_low_group, metric_high_group):
    cv_low = coefficient_of_variation(metric_low_group) if len(metric_low_group) > 0 else 0
    cv_high = coefficient_of_variation(metric_high_group) if len(metric_high_group) > 0 else 0
    return (cv_low + cv_high) / 2


def coefficient_of_variance(group):
    """Calculates the coefficient of variance for a group."""
    return np.std(group) / np.mean(group) if np.mean(group) != 0 else 0





df_rent_clean['category'] = df_rent_clean['category'].fillna('unknown')


def create_weighted_user_profile(user_id, df, interaction_data):
    """
    Create a user profile based on the weighted interaction strengths for the categories
    of the items the user has interacted with.
    """

    user_items = interaction_data.loc[user_id, interaction_data.loc[user_id] > 0].index.tolist()


    user_items_df = df[df['item_id'].isin(user_items)]


    weighted_categories = user_items_df.groupby('category')['rating'].sum()


    user_profile = ' '.join([f'{category} ' * int(weight) for category, weight in weighted_categories.items()])

    return user_profile


def recommend_items_for_user(user_id, df, interaction_data, top_n=5):
    """
    Recommend items for a user based on their weighted profile similarity with all other items.
    """

    user_profile = create_weighted_user_profile(user_id, df, interaction_data)


    vectorizer = TfidfVectorizer(stop_words='english')


    all_profiles = df['category'].tolist() + [user_profile]


    profile_matrix = vectorizer.fit_transform(all_profiles)


    cosine_sim = cosine_similarity(profile_matrix[-1:], profile_matrix[:-1])


    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


    item_indices = [i[0] for i in sim_scores[:top_n]]
    item_scores = [i[1] for i in sim_scores[:top_n]]


    recommended_items = df.iloc[item_indices]

    return recommended_items[['item_id', 'category','rating','age']], item_scores


recommended_items, scores = recommend_items_for_user(user_id=581157, df=df_rent_clean, interaction_data=train_user_item_matrix)


print("Recommended Items for User 581157:")
print(recommended_items)


import datetime

def create_weighted_user_profile(user_id, df, interaction_data):
    if user_id not in interaction_data.index:
        return ' '

    user_items = interaction_data.loc[user_id, pd.to_numeric(interaction_data.loc[user_id], errors='coerce') > 0].index
    user_items = user_items.tolist()
    user_items_df = df[df['item_id'].isin(user_items)].copy()

    most_recent_date = user_items_df['review_date'].max()

    # Convert review dates to datetime format and compute a time decay factor
    # that gives more weight to recent interactions (exponential decay over years)
    user_items_df['review_date'] = pd.to_datetime(user_items_df['review_date'])
    most_recent_date = pd.to_datetime(most_recent_date)

    user_items_df.loc[:, 'time_decay'] = np.exp(-(most_recent_date - user_items_df['review_date']).dt.days / 365)


    weighted_categories = user_items_df.groupby('category', observed=True)[['interaction', 'time_decay']].apply(lambda group: (group['interaction'] * group['time_decay']).sum())
    weighted_categories = weighted_categories.drop(columns=['interaction', 'time_decay'])

    user_profile = ' '.join([f'{category} ' * int(weight) for category, weight in weighted_categories.items()])

    return user_profile


def recommend_items_for_user_torch(user_id, df, interaction_data, top_n=5):
    user_profile = create_weighted_user_profile(user_id, df, interaction_data)


    all_profiles = df['category'].astype(str).tolist() + [str(user_profile)]
    all_profiles = [profile for profile in all_profiles if profile.strip()]

    vectorizer = TfidfVectorizer(stop_words='english')
    profile_matrix = vectorizer.fit_transform(all_profiles)
    cosine_sim = cosine_similarity(profile_matrix[-1:], profile_matrix[:-1])
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    item_indices = [i[0] for i in sim_scores[:top_n]]
    recommended_items = df.iloc[item_indices][['item_id', 'category']]
    return recommended_items['item_id'].tolist()


def calculate_metrics_for_user(user_id, actual_item, recommended_items, user_engaged, k=5):
    hr = hr_at_k([actual_item], recommended_items, k)
    mrr = mrr_at_k([actual_item], recommended_items, k)

    relevance_scores = [1 if item == actual_item else 0 for item in recommended_items[:k]]
    dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores)])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(relevance_scores)))])
    ndcg = dcg / idcg if idcg > 0 else 0
    cv = coefficient_of_variance(relevance_scores)

    return {
        'User': user_id,
        'NDCG@K': ndcg,
        'HR@K': hr,
        'MRR@K': mrr,
        'CV': cv,
        'Engagement Group': user_engaged
    }
def update_user_profile_with_relevance(user_id, relevant_items, interaction_data, df):
    """
    Update the user profile with relevant items.
    """

    user_profile = create_weighted_user_profile(user_id, df, interaction_data)


    for item_id in relevant_items:
        category = df[df['item_id'] == item_id]['category'].values[0]
        user_profile += f' {category}'

    return user_profile


def recommend_and_update(user_id, df, interaction_data, excluded_items, top_n=5):
    """
    Recommend items for a user and update the profile based on relevance.
    """

    recommended_items = recommend_items_for_user_torch(user_id, df, interaction_data, top_n=top_n)

    recommended_items = [item for item in recommended_items if item not in excluded_items]

    relevant_items = recommended_items[:2]
    irrelevant_items = recommended_items[2:]

    updated_profile = update_user_profile_with_relevance(user_id, relevant_items, interaction_data, df)

    excluded_items.update(irrelevant_items)

    return relevant_items, updated_profile, excluded_items



excluded_items = set()
for round_num in range(3):
    print(f"Round {round_num + 1}")
    relevant_items, updated_profile, excluded_items = recommend_and_update(
        user_id=123, df=df_rent_clean, interaction_data=train_user_item_matrix, excluded_items=excluded_items
    )
    print(f"Relevant Items: {relevant_items}")
    print(f"Excluded Items: {excluded_items}")
def update_user_profile(user_id, actual_item, train_matrix, df):
    """
    Update the user profile with the actual item interacted with.
    This function assumes that the user has interacted with the actual item.
    """

    user_profile = create_weighted_user_profile(user_id, df, train_matrix)


    new_profile = user_profile + ' ' + df[df['item_id'] == actual_item]['category'].values[0]


    train_matrix.loc[user_id] = new_profile

    return train_matrix




def coef_variation(arr):
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 0
    return np.std(arr) / mean_val



def update_user_profile_with_relevance(user_id, relevant_items, interaction_data, df):
    """
    Update the user profile with relevant items.
    """

    user_profile = create_weighted_user_profile(user_id, df, interaction_data)


    for item_id in relevant_items:

        if item_id in df['item_id'].values:
            category = df[df['item_id'] == item_id]['category'].values[0]
            user_profile += f' {category}'

    return user_profile


def recommend_and_update(user_id, df, interaction_data, excluded_items, top_n=5):
    """
    Recommend items for a user and update the profile based on relevance.
    """

    recommended_items = recommend_items_for_user_torch(user_id, df, interaction_data, top_n=top_n)


    recommended_items = [item for item in recommended_items if item not in excluded_items]


    relevant_items = recommended_items[:2]
    irrelevant_items = recommended_items[2:]


    updated_profile = update_user_profile_with_relevance(user_id, relevant_items, interaction_data, df)


    excluded_items.update(irrelevant_items)

    return relevant_items, updated_profile, excluded_items






###############################################
# 3. Recommendation User based model
###############################################
def get_user_based_recommendations(user_index, neighbor_indices, train_matrix, distances, top_k=5):
    """
    Generate Recommendations for a user based on preference or similar users (neighbors) using user-based CF.

    Args:
        user_index: index of user in the user-item matrix
        neighbor_indices: indices of the nearest neighbors
        train_matrix: user-item matrix
        distances: distance matrix of nearest neighbors
        top_k: number of recommendations to generate

    Returns:
        recommendations: list of recommended items for the user
    """

    user_interacted = train_matrix.columns[train_matrix.iloc[user_index] > 0].tolist()
    rec_scores = {}

    for i, neighbor in enumerate(neighbor_indices[:10]):
        if neighbor == user_index:
            continue
        neighbor_vector = train_matrix.iloc[neighbor]

        for item, score in neighbor_vector.items():
            if score > 0 and item not in user_interacted:
                rec_scores[item] = rec_scores.get(item, 0) + score * (1 / (1 + distances[0][i]))

    if not rec_scores:
        return [(item, 0) for item in np.random.choice(train_matrix.columns, size=top_k, replace=False).tolist()]

    return sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
###############################################
# 4. Hybrid Recommender
###############################################
def hybrid_recommendations(user_id, df, train_user_item_matrix, content_weight=0.5, collaborative_weight=0.5, top_k=5):
    """
    Generate hybrid recommendations by combining content-based and collaborative filtering.

    Args:
        user_id: The ID of the user for whom to generate recommendations.
        df: The DataFrame containing item data.
        train_user_item_matrix: The user-item interaction matrix.
        content_weight: The weight to give to content-based recommendations.
        collaborative_weight: The weight to give to collaborative filtering recommendations.
        top_k: The number of recommendations to generate.

    Returns:
        A list of recommended item IDs.
    """

    content_recs, content_scores = recommend_items_for_user(user_id, df, train_user_item_matrix, top_n=top_k)  # Get content scores
    content_recs_list = content_recs['item_id'].tolist()


    user_index = train_user_item_matrix.index.get_loc(user_id)  # Get user index
    distances, neighbor_indices = nn_model.kneighbors(train_user_item_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=10)
    collaborative_recs = get_user_based_recommendations(user_index, neighbor_indices[0], train_user_item_matrix, distances, top_k=top_k)
    collaborative_recs_list = [item_id for item_id, _ in collaborative_recs]

    all_recs = {}
    for item_id in set(content_recs_list + collaborative_recs_list):

        content_score = next((score for rec_item_id, score in zip(content_recs['item_id'], content_scores) if rec_item_id == item_id), 0)
        collaborative_score = next((score for rec_item_id, score in collaborative_recs if rec_item_id == item_id), 0)
        all_recs[item_id] = content_weight * content_score + collaborative_weight * collaborative_score


    sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return sorted_recs



def calculate_metrics_for_hybrid(user_id, actual_item, recommendations, engagement_level, k=5):
    """Calculates NDCG, MRR, and HR for hybrid recommendations."""

    recommended_items = [item_id for item_id, _ in recommendations]
    relevance_scores = [1 if item == actual_item else 0 for item in recommended_items]
    ndcg = ndcg_at_k(relevance_scores, k)
    hr = hr_at_k([actual_item], recommended_items, k)
    mrr = mrr_at_k([actual_item], recommended_items, k)

    return [user_id, ndcg, hr, mrr, engagement_level]

nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
nn_model.fit(train_user_item_matrix)
## hyperparameter :

nn_model = NearestNeighbors(metric='cosine')
nn_model.fit(train_user_item_matrix)
param_grid = {
    'content_weight': [0.1,  0.5,  0.8],
    'collaborative_weight': [0.1, 0.5,0.8],
}

best_average_ndcg = -1
best_params = {}

for params in ParameterGrid(param_grid):
    total_weight = params['content_weight'] + params['collaborative_weight']
    if total_weight == 0:
        continue
    normalized_content_weight = params['content_weight'] / total_weight
    normalized_collaborative_weight = params['collaborative_weight'] / total_weight

    metrics_list = []


    unique_user_ids = df_rent_clean['user_id'].unique()

    for user_id in unique_user_ids:
        if user_id not in train_user_item_matrix.index:
            continue

        actual_item = test_items.get(user_id, None)

        if actual_item is None:
            continue

        user_data = df_rent_clean[df_rent_clean['user_id'] == user_id]
        if user_data.empty:
            continue

        user_activity = user_data['user_activity'].iloc[0]

        if user_activity == 'Low':
            user_engaged = "Low"
        else:
            user_engaged = "High"

        recommended_items, recommended_scores = hybrid_recommendations(
            user_id,
            df_rent_clean,
            train_user_item_matrix,
            nn_model,
            content_weight=normalized_content_weight,
            collaborative_weight=normalized_collaborative_weight,
            top_k=5 # in future hyperparameter for k
        )

        if recommended_items:
            metrics = calculate_metrics_for_user(
                user_id,
                actual_item,
                recommended_items,
                user_engaged,
                k=5
            )
            metrics_list.append(metrics)


    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        average_ndcg_for_params = metrics_df['NDCG@K'].mean()


        if average_ndcg_for_params > best_average_ndcg:
            best_average_ndcg = average_ndcg_for_params
            best_params = params
    else:
        print(f"No metrics calculated for params: {params}")

print(f"Best Hyperparameters: {best_params}")
print(f"Best Average NDCG: {best_average_ndcg}")

#####
# best param 0.5 0.5
distances, indices = nn_model.kneighbors(train_user_item_matrix)

def recommend_items_for_user_torch(user_id, df, interaction_data, top_n=5):
    user_profile = create_weighted_user_profile(user_id, df, interaction_data)
    vectorizer = TfidfVectorizer(stop_words='english')
    all_profiles = df['category'].tolist() + [user_profile]
    profile_matrix = vectorizer.fit_transform(all_profiles)
    cosine_sim = cosine_similarity(profile_matrix[-1:], profile_matrix[:-1])
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    item_indices = [i[0] for i in sim_scores[:top_n]]
    recommended_items = df.iloc[item_indices][['item_id', 'category']]
    return recommended_items['item_id'].tolist()

def create_weighted_user_profile(user_id, df, interaction_data):
    if user_id not in interaction_data.index:
        return ' '
    user_items = interaction_data.loc[user_id, pd.to_numeric(interaction_data.loc[user_id], errors='coerce') > 0].index
    user_items = user_items.tolist()
    user_items_df = df[df['item_id'].isin(user_items)]
    weighted_categories = user_items_df.groupby('category')['interaction'].sum()
    user_profile = ' '.join([f'{category} ' * int(weight) for category, weight in weighted_categories.items()])
    return user_profile

print("Distances shape:", distances.shape)
print("Indices shape:", indices.shape)
user_id = 623100
recommendations = hybrid_recommendations(user_id, df_rent_clean, train_user_item_matrix, top_k=5)

print(f"Hybrid Recommendations for User {user_id}:")
for item_id in recommendations:
    print(item_id)

def create_weighted_user_profile(user_id, df, interaction_data):
    if user_id not in interaction_data.index:
        return ' '
    user_items = interaction_data.loc[user_id, pd.to_numeric(interaction_data.loc[user_id], errors='coerce') > 0].index
    user_items = user_items.tolist()
    user_items_df = df[df['item_id'].isin(user_items)]
    weighted_categories = user_items_df.groupby('category')['interaction'].sum()
    user_profile = ' '.join([f'{category} ' * int(weight) for category, weight in weighted_categories.items()])
    return user_profile


def recommend_items_for_user_torch(user_id, df, interaction_data, top_n=5):
    user_profile = create_weighted_user_profile(user_id, df, interaction_data)
    vectorizer = TfidfVectorizer(stop_words='english')
    all_profiles = df['category'].tolist() + [user_profile]
    profile_matrix = vectorizer.fit_transform(all_profiles)
    cosine_sim = cosine_similarity(profile_matrix[-1:], profile_matrix[:-1])
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    item_indices = [i[0] for i in sim_scores[:top_n]]
    recommended_items = df.iloc[item_indices][['item_id', 'category','age']]
    return recommended_items['item_id'].tolist()


def coef_variation(arr):
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 0
    return np.std(arr) / mean_val
def calculate_all_metrics_for_rounds_hybrid(data, train_user_item_matrix, rounds=3, device='cuda'):
    all_rounds_metrics = []

    data['user_engaged'] = data['review_summary'].notnull()
    test_items = data.groupby("user_id")["item_id"].first().to_dict()

    for round_num in range(rounds):
        print(f"Starting round {round_num + 1} of hybrid recommendation...")

        metrics_hybrid = []
        for user_id in data['user_id'].unique():
            if user_id not in train_user_item_matrix.index:
                continue
            actual_item = test_items.get(user_id, None)
            if actual_item is None:
                continue

            user_data = data[data['user_id'] == user_id]
            user_activity = df_rent_clean.loc[df_rent_clean['user_id'] == user_id, 'user_activity'].iloc[0]

            if user_activity == 'Low':
                user_engaged = "Low"
            else:
                user_engaged = "High"

            recommendations = hybrid_recommendations(user_id, data, train_user_item_matrix, top_k=5)
            metrics_hybrid.append(calculate_metrics_for_hybrid(user_id, actual_item, recommendations, user_engaged))

        metrics_hybrid_df = pd.DataFrame(metrics_hybrid, columns=["User", "NDCG@K", "HR@K", "MRR@K", "Engagement Group"])
        grouped_metrics_hybrid = metrics_hybrid_df.groupby("Engagement Group").agg({"NDCG@K": "mean", "HR@K": "mean", "MRR@K": "mean"})
        all_rounds_metrics.append(grouped_metrics_hybrid)
        print(grouped_metrics_hybrid)
        ndcg_low = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] == 'Low']['NDCG@K'].values
        ndcg_high = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] == 'High']['NDCG@K'].values
        hr_low = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] == 'Low']['HR@K'].values
        hr_high = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] == 'High']['HR@K'].values
        mrr_low = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] == 'Low']['MRR@K'].values
        mrr_high = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] == 'High']['MRR@K'].values

        disparate_impact_ndcg = calculate_disparate_impact(ndcg_low, ndcg_high)
        gru_ndcg = calculate_group_recommender_unfairness(ndcg_low, ndcg_high)

        disparate_impact_hr = calculate_disparate_impact(hr_low, hr_high)
        gru_hr = calculate_group_recommender_unfairness(hr_low, hr_high)
        disparate_impact_mrr = calculate_disparate_impact(mrr_low, mrr_high)
        gru_mrr = calculate_group_recommender_unfairness(mrr_low, mrr_high)
        print(f"Disparate Impact for NDCG: {disparate_impact_ndcg}")
        print(f"Disparate Impact for HR: {disparate_impact_hr}")
        print(f"Disparate Impact for MRR: {disparate_impact_mrr}")
        print(f"GRU for NDCG: {gru_ndcg}")
        print(f"GRU for HR: {gru_hr}")
        print(f"GRU for MRR: {gru_mrr}")

        def get_ucv(metric_name):
            low = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] == 'Low'][metric_name].values
            high = metrics_hybrid_df[metrics_hybrid_df['Engagement Group'] != 'Low'][metric_name].values
            return (coef_variation(low) + coef_variation(high)) / 2

        ucv_ndcg = get_ucv('NDCG@K')
        ucv_hr = get_ucv('HR@K')
        ucv_mrr = get_ucv('MRR@K')
        print(f"UCV for NDCG: {ucv_ndcg}")
        print(f"UCV for HR: {ucv_hr}")
        print(f"UCV for MRR: {ucv_mrr}")
        avg_ndcg = metrics_hybrid_df['NDCG@K'].mean()
        avg_hr = metrics_hybrid_df['HR@K'].mean()
        avg_mrr = metrics_hybrid_df['MRR@K'].mean()
        print(f"Average NDCG: {avg_ndcg}")
        print(f"Average HR: {avg_hr}")
        print(f"Average MRR: {avg_mrr}")

        print(f"Round {round_num + 1} Metrics with UCV:")
        print(grouped_metrics_hybrid)

        print("\n")

    return all_rounds_metrics


device = 'cuda' if torch.cuda.is_available() else 'cpu'
all_rounds_hybrid_metrics = calculate_all_metrics_for_rounds_hybrid(df_rent_clean, train_user_item_matrix, rounds=3, device=device)


for round_num, metrics in enumerate(all_rounds_hybrid_metrics):
    print(f"\nHybrid Metrics for Round {round_num + 1}:")
    print(metrics)
metrics_df = pd.concat(all_rounds_hybrid_metrics)
metrics_df.to_csv('hybrid_metrics.csv', index=False)

def calculate_di_gru(all_round_metrics):
    """
    Calculates Disparate Impact and Group Recommender Unfairness for NDCG and HR
    for each round of metrics.

    Args:
        all_round_metrics (list): A list of DataFrames, where each DataFrame contains
                                   metrics for a single round, grouped by Engagement Group.
                                   Each DataFrame is expected to have 'NDCG@K' and 'HR@K' columns
                                   and 'Low' and 'High' index labels.
    """
    for round_num, round_metrics_df in enumerate(all_round_metrics):
        print(f"\n--- Fairness Metrics for Round {round_num + 1} ---")


        if 'Low' in round_metrics_df.index and 'High' in round_metrics_df.index:

            ndcg_low = round_metrics_df.loc["Low", "NDCG@K"]
            ndcg_high = round_metrics_df.loc["High", "NDCG@K"]
            hr_low = round_metrics_df.loc["Low", "HR@K"]
            hr_high = round_metrics_df.loc["High", "HR@K"]
            mrr_low = round_metrics_df.loc["Low", "MRR@K"]
            mrr_high = round_metrics_df.loc["High", "MRR@K"]


            di_ndcg = calculate_disparate_impact([ndcg_low], [ndcg_high])
            di_hr = calculate_disparate_impact([hr_low], [hr_high])
            di_mrr = calculate_disparate_impact([mrr_low], [mrr_high])


            gru_ndcg = calculate_group_recommender_unfairness([ndcg_low], [ndcg_high])
            gru_hr = calculate_group_recommender_unfairness([hr_low], [hr_high])
            gru_mrr = calculate_group_recommender_unfairness([mrr_low], [mrr_high])


            print(f"NDCG:")
            print(f"  Disparate Impact (Low/High): {di_ndcg:.4f}")
            print(f"  Group Recommender Unfairness (Abs Diff): {gru_ndcg:.4f}")
            print(f"MRR:")
            print(f"  Disparate Impact (Low/High): {di_mrr:.4f}")
            print(f"  Group Recommender Unfairness (Abs Diff): {gru_mrr:.4f}")

            print(f"HR@K:")
            print(f"  Disparate Impact (Low/High): {di_hr:.4f}")
            print(f"  Group Recommender Unfairness (Abs Diff): {gru_hr:.4f}")
        else:
            print("Could not calculate fairness metrics as one or both engagement groups are missing.")

calculate_di_gru(all_rounds_hybrid_metrics)
