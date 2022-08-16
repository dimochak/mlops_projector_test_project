import os

import joblib
import pandas as pd
import re
import nltk
import time

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm

from fastapi import FastAPI
from pydantic import BaseModel, constr, conlist
from typing import List


nltk.download('stopwords')
nltk.download('punkt')
app = FastAPI()


# In[]
def preprocess(data, path_to_results, type='train'):
    excerpt_processed = []
    for e in tqdm(data['excerpt']):
        # find alphabets
        e = re.sub("[^a-zA-Z]", " ", e)

        # convert to lower case
        e = e.lower()

        # tokenize words
        e = nltk.word_tokenize(e)

        # remove stopwords
        e = [word for word in e if not word in set(stopwords.words("english"))]

        # lemmatization
        lemma = nltk.WordNetLemmatizer()
        e = [lemma.lemmatize(word) for word in e]
        e = " ".join(e)

        excerpt_processed.append(e)
    joblib.dump(excerpt_processed,
                os.path.join(path_to_results, f'excerpts_preprocessed_{type}.joblib'))
    return excerpt_processed


# In[]
class UserRequest(BaseModel):
    text: constr(min_length=1)


class ModelResponse(BaseModel):
    text: str
    score: float

# In[]
def training(model, X, y, path_to_results):
    pipeline = make_pipeline(
        TfidfVectorizer(binary=True, ngram_range=(1, 1)),
        model,
    )
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    print(f'RMSE (train error): {mse(y, y_pred, squared=False)}')
    joblib.dump(pipeline[1],
                os.path.join(f'{path_to_results}', 'model.joblib'))
    joblib.dump(pipeline,
                os.path.join(f'{path_to_results}', 'pipeline.joblib'))
    with open(os.path.join(f'{path_to_results}', 'rmse.txt'), 'w+') as f:
        f.writelines(f'RMSE (train error): {mse(y, y_pred, squared=False)}')


# In[]
def init_training():
    cur_dir = os.getcwd()
    train_df = pd.read_csv(os.path.join(cur_dir, "data", "train.csv"))
    test_df = pd.read_csv(os.path.join(cur_dir, "data", "test.csv"))
    path_to_results = os.path.join(cur_dir, 'results')
    path_to_train_preprocessed = 'excerpts_preprocessed_train.joblib'

    if not os.path.exists(os.path.join(path_to_results, path_to_train_preprocessed)):
        train_df["excerpt_preprocessed"] = preprocess(train_df, path_to_results)
        test_df["excerpt_preprocessed"] = preprocess(test_df, path_to_results, type='test')
    else:
        train_df["excerpt_preprocessed"] = joblib.load(os.path.join(path_to_results,
                                                                    'excerpts_preprocessed_train.joblib'))
        test_df["excerpt_preprocessed"] = joblib.load(os.path.join(path_to_results,
                                                                   'excerpts_preprocessed_test.joblib'))

    model = Ridge()
    X = train_df["excerpt_preprocessed"]
    y = train_df['target']

    training(model, X, y, os.path.join(cur_dir, 'results'))
    return "Model is trained."


@app.get("/")
async def root():
    return {'If you see this message, it means that application has been deployed properly. Navigate to {app-name}/docs '
            'to play around with available methods'}

@app.post('/predict', response_model=ModelResponse)
async def predict(request: UserRequest):
    path_to_pipeline = os.path.join(os.getcwd(), "results", "pipeline.joblib")
    if not os.path.exists(path_to_pipeline):
        init_training()
    pipeline = joblib.load(path_to_pipeline)
    print(request.text)
    user_score = pipeline.predict([request.text])
    return ModelResponse(text=request.text, score=user_score)