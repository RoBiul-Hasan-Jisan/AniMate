#  Anime AI Recommender


**Anime AI Recommender** is an interactive web application designed to predict anime ratings, scores, and genres, while providing highly personalized recommendations using machine learning. Built with **Python**, **Flask**, and state-of-the-art ML models, it empowers anime fans to discover new shows based on their unique tastes.



---

##  Key Features

* **Score Prediction:** Predict the overall community score (e.g., MyAnimeList score) of any anime.
* **Rating Prediction:** Estimate the appropriate age rating (e.g., PG-13, R-17+).
* **Genre Classification:** Automatically predict multiple fitting genres for a given anime synopsis.
* **Content-Based Recommendations:** Suggest similar anime titles using natural language processing on plot summaries.
* **Collaborative Filtering *(Optional)*:** Recommend anime based on user popularity trends and interaction data.

---

##  Project Structure & Models

| File / Directory | Description |
| :--- | :--- |
| `app.py` | Main Flask application script handling backend logic and routing. |
| `templates/index.html` | Frontend HTML template for the user interface. |
| `top_anime_dataset.csv` | Scraped dataset containing rich anime metadata. |
| `models/score_rf_model.pkl` | **Random Forest** model trained for score prediction. |
| `models/rating_cb_model.pkl` | **CatBoost** model trained for age rating prediction. |
| `models/genre_rf_model.pkl` | **MultiOutput Random Forest** model for multi-label genre classification. |
| `models/content_based_nn.pkl`| **Nearest Neighbors** model for content-based similarity matching. |

---


## Usage

- Enter your favorite anime title (or a new concept) into the search/input box.

- Click the "Get Recommendations" button.

- Instantly view the generated analytics:

- Predicted community score

- Predicted age rating

- Associated genres

- A curated list of similar anime recommendations

##  Dataset
The dataset powering this engine is the **[Top Anime Dataset](https://www.kaggle.com/datasets/muhammadaqeelkabir/top-anime-csv/data)**, originally scraped from **MyAnimeList** using the **Jikan API**.

**Features included:**
* Anime ID & Title
* Type & Source
* Episodes & Status
* Community Score
* Members & Favorites
* Synopsis (Plot summary)
* Genres & Studios
* Release Year

## Technology Stack
- Backend: Python 3.12, Flask

- Machine Learning: scikit-learn, XGBoost, CatBoost

- NLP & Recommender: NLTK, NearestNeighbors algorithm

- Frontend: HTML5, Bootstrap 5
