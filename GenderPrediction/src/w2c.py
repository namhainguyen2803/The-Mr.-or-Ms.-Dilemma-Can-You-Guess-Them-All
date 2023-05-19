from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import pandas as pd
# Load name corpus
data_path= "/Users/nguyenbathiem/The-Mr.-or-Ms.-Dilemma-Can-You-Guess-Them-All/GenderPrediction/dataset/name_full.csv"
df = pd.read_csv(data_path)
df


# # Train the Word2Vec model
model = Word2Vec(sentences=sentences, min_count=1)

# # Get the vector representation of each sentence
# sentence_vectors = []
# for sentence in sentences:
#     vector = sum(model.wv[word] for word in sentence) / len(sentence)  # Average word vectors in the sentence
#     sentence_vectors.append(vector)

# # Print the sentence vectors
# for i, vector in enumerate(sentence_vectors):
#     print(f"Sentence {i+1}: {vector}")
