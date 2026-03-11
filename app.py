from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)

# -------------------------------
# 1. Load Models
# -------------------------------
score_model = joblib.load("models/score_rf_model.pkl")
rating_model = joblib.load("models/rating_cb_model.pkl")
genre_model = joblib.load("models/genre_rf_model.pkl")  # optional if we don't predict genres
nn_model = joblib.load("models/content_based_nn.pkl")

# -------------------------------
# 2. Load Dataset
# -------------------------------
df = pd.read_csv("top_anime_dataset.csv")

# Preprocess synopsis
df['synopsis'] = df['synopsis'].fillna('')

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['synopsis'])
svd = TruncatedSVD(n_components=200, random_state=42)
synopsis_features = svd.fit_transform(tfidf_matrix)

# -------------------------------
# 3. Helper Functions
# -------------------------------
def get_features(idx):
    # numeric + SVD features
    numeric_features = [
        df.loc[idx, 'episodes'],
        df.loc[idx, 'members'],
        df.loc[idx, 'favorites'],
        0,  # type_enc placeholder
        0,  # source_enc placeholder
        0,  # rating_enc placeholder
        df.loc[idx, 'year']
    ]
    return np.hstack([numeric_features, synopsis_features[idx]])

def predict_score(features):
    return round(float(score_model.predict([features])[0]), 2)

def predict_rating(features):
    return rating_model.predict([features])[0]  # returns encoded label (simplified)

def get_genres(idx):
    # Use original genres from dataset
    return df.loc[idx, 'genres_list'] if 'genres_list' in df.columns else df.loc[idx, 'genres'].split(',')

def recommend_content(title):
    idx_list = df.index[df['title'] == title].tolist()
    if not idx_list:
        return ["Anime not found!"]
    idx = idx_list[0]
    distances, indices = nn_model.kneighbors([synopsis_features[idx]])
    return df.iloc[indices[0][1:]]['title'].values.tolist()

# -------------------------------
# 4. Flask Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form["title"]

        if title in df['title'].values:
            idx = df.index[df['title'] == title][0]
            features = get_features(idx)

            score_pred = predict_score(features)
            rating_pred = predict_rating(features)
            genres_pred = get_genres(idx)
            recs = recommend_content(title)

            return render_template(
                "index.html",
                title=title,
                score=score_pred,
                rating=rating_pred,
                genres=", ".join(genres_pred),
                recommendations=recs
            )
        else:
            return render_template("index.html", error="Anime not found!")

    return render_template("index.html")

# -------------------------------
# 5. Run Flask
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
