import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
file_path = r"C:\Users\rithi\OneDrive\Desktop\intrainz\OnlineRetail.xlsx"
df = pd.read_excel(file_path)

# Remove rows with NaN values in the 'Description' column
df = df.dropna(subset=['Description'])

# Create a TF-IDF vectorizer to convert product descriptions into vectors
vectorizer = TfidfVectorizer(stop_words='english')
description_vectors = vectorizer.fit_transform(df['Description'])

# Calculate the cosine similarity between product vectors
similarity_matrix = cosine_similarity(description_vectors, description_vectors)

# Create a function to get recommendations for a given product
def get_recommendations(product_id, num_recs=5):
    # Get the index of the product in the similarity matrix
    idx = df[df['Stock Code'] == product_id].index[0]
    
    # Get the similarity scores for the product
    scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort the scores in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations
    recs = [df.iloc[i[0]]['Stock Code'] for i in scores[1:num_recs+1] if i[0] != idx]
    
    return recs

# Example usage:
product_id = '85123A'  # Replace with a product ID from the dataset
recommendations = get_recommendations(product_id)
print(recommendations)