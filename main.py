# libraries
from utils import *
import pandas as pd
from modal import Image,Secret,Mount, App, asgi_app, gpu, Volume
from fastapi import FastAPI, Response, Query, File, UploadFile, Request
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Annotated
import os
from io import BytesIO
import json

volume = Volume.from_name("my-volume-2")
app = App(name="api-predict-category-normalized-v1")

conda_image = (Image.micromamba()
               .micromamba_install(
                "cudatoolkit=11.2",
                "cudnn=8.1.0",
                "cuda-nvcc",
                channels=["conda-forge", "nvidia"],
                )
               .pip_install("pandas", "numpy", "matplotlib", "requests",
                            "jax", "jaxlib", "transformers", "tensorflow~=2.9.1",
                            "Unidecode", "python-dotenv", "mysql-connector-python",
                            "scikit-learn", "google-cloud-aiplatform==1.25", "python-jose[cryptography]",
                            "passlib[bcrypt]", "python-multipart", "openpyxl"))


class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers['X-Process-Time'] = str(process_time)
        return response


categories_app = FastAPI(title='CategoryPredictionsAPI',
                       summary="API for predicting the category of a sku", version="1.1",
                       contact={
                                "name": "Cristian Vergara",
                                "email": "cvergara@geti.cl",
                                })
categories_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'])
categories_app.add_middleware(MyMiddleware)
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@categories_app.get("/")
def read_root():
    return {"Root": "Root_test"}


@categories_app.get("/sku-category-prediction-web")
async def predict_category_from_website(product_title: str, product_category: str): # used to predict category from website
    try:
        print('Making prediction..\n')
        start = time.time()
        df = pd.DataFrame()
        df['category_name'] = [product_category]
        df['product_name'] = [product_title]
        df['predicted_category'] = df.apply(
            lambda row: predict_product_category(row['category_name'], row['product_name'], top=2, model=cats_model, max_length=MAX_LENGTH,
                                                 tokenizer=bert_tokenizer), axis=1)
        df['predicted_category_1'] = df['predicted_category'].apply(lambda row: row[0][0])
        df['probability_category_1'] = df['predicted_category'].apply( lambda row: row[0][1])
        df['predicted_category_2'] = df['predicted_category'].apply(lambda row: row[1][0])
        df['probability_category_2'] = df['predicted_category'].apply(lambda row: row[1][1])
        df = df.drop(columns=['predicted_category'])
        total_time = round(time.time() - start, 2)
        print('Making prediction..OK!')
        print(f'time taken: {total_time} secs')

        output = df.to_json(orient="records")
        return Response(content=output, media_type="application/json")

    except Exception as e:
        output = {
            "error": str(e),
        }
        return output


@categories_app.get("/wv-ids-categories-prediction")
async def predict_category_from_wv_ids(wv_id: List[int] = Query()):
    try:
        wv_ids_to_predict = {"web_variety_ids": wv_id}
        web_variety_ids = wv_ids_to_predict["web_variety_ids"]

        df = skus_from_wv_ids(web_variety_ids)
        df_to_print = df[['id', 'sku', 'retail_id']]
        print(f'Skus to predict the category: \n {df_to_print}')
        print('Making prediction..')
        df['brand_and_product'] = df.brand + ' ' + df.product_name + ' ' + df.variety_name
        df['brand_and_product'] = df['brand_and_product'].apply(lambda row: preprocess_products(str(row)))
        df['predicted_category'] = df.apply(
            lambda row: predict_product_category(row['category_name'], row['brand_and_product'], top=2, model=cats_model,
                                                 max_length=MAX_LENGTH, tokenizer=bert_tokenizer), axis=1)
        df['predicted_category_1'] = df['predicted_category'].apply(lambda row: row[0][0])
        df['probability_category_1'] = df['predicted_category'].apply(lambda row: row[0][1])
        df['predicted_category_2'] = df['predicted_category'].apply(lambda row: row[1][0])
        df['probability_category_2'] = df['predicted_category'].apply(lambda row: row[1][1])
        df = df.drop(columns=['predicted_category', 'brand_and_product'])
        print('Making prediction..OK!')
        output = df.to_json(orient="records")
        return Response(content=output, media_type="application/json")

    except Exception as e:
        output = {
            "error": str(e),
        }
        return output


@app.function(image=conda_image,gpu=gpu.T4(count=1),
              secret=Secret.from_name("automatch-secret-keys"),
              mounts=[Mount.from_local_file("model_categories_txt_v5_acc.h5",
                                             remote_path="/root/model_categories_txt_v5_acc.h5"),
                      Mount.from_local_file("dict_categories.pkl",
                                             remote_path="/root/dict_categories.pkl"),
                      Mount.from_local_file("dict_categories_inv.pkl",
                                             remote_path="/root/dict_categories_inv.pkl"),
                      Mount.from_local_file("automatch-309218-5f83b019f742.json",
                                             remote_path="/root/automatch-309218-5f83b019f742.json")],
              volumes={"/vol": volume}, _allow_background_volume_commits=True,  #before: shared_volumes
              timeout=999)  # schedule=Period(minutes=30)
@asgi_app(label='predict-category-normalized-v1')
def fastapi_app():
    # check available GPUs
    print(get_available_gpus())

    return categories_app
