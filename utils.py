import pandas as pd
import numpy as np
import time
import re
import unidecode
import itertools
from dotenv import load_dotenv
import os
import mysql.connector
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.python.client import device_lib
from vertexai.preview.language_models import TextEmbeddingModel
from collections import Counter
import io
import requests
import warnings
import math
import pickle
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


# envs
load_dotenv()
MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
MYSQL_PASSWORD_ASSISTED_MATCH = os.environ["MYSQL_PASSWORD_ASSISTED_MATCH"]
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'
CACHE_DIR = "/Users/cvergarabah/.cache/huggingface/hub"
PINECONE_GETI_API_KEY = os.environ["PINECONE_GETI_API_KEY"]
PINECONE_GETI_ENV = os.environ["PINECONE_GETI_ENV"]
#PINECONE_GETI_API_KEY_FREE = os.environ["PINECONE_GETI_API_KEY_FREE"]
#PINECONE_GETI_ENV_FREE = os.environ["PINECONE_GETI_ENV_FREE"]
MAX_LENGTH = 30  # model parameter


#mysql
def sku_to_data_from_sql(skus_ids_to_search: list, retail_id_to_search: int, host='db.geti.cl',
                         database='winodds', user='cvergara', password=MYSQL_PASSWORD):
    """extract data from mySQL based on the skus and retails ids. Used for starting the candidates search"""

    # formatting for the query
    if len(skus_ids_to_search) == 1:
        results_ids = tuple(skus_ids_to_search)[0]
        results_ids = f"'{results_ids}'"
        aux = '='
    elif len(skus_ids_to_search) > 1:
        results_ids = tuple(skus_ids_to_search)
        aux = 'IN'

    # Creating connection object
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    mycursor = mydb.cursor()

    query = f"SELECT wv.id, wv.retail_id, \
              wv.sku, wv.brand AS brand,\
              wv.product_name, wv.variety_name, \
              wv.web_category_id, wc.name AS category_name, \
              wv.image, wv.url \
              FROM winodds.web_varieties wv\
              INNER JOIN winodds.web_categories wc ON wv.web_category_id = wc.id \
              WHERE wv.sku {aux} {results_ids} && wv.retail_id = {retail_id_to_search} && wv.discontinued = 0"

    mycursor.execute(query)
    results = mycursor.fetchall()

    # inserting the results into a dataframe
    cols_name = [i[0] for i in mycursor.description]
    cols_and_rows = [dict(zip(cols_name, result)) for result in results]
    df = pd.DataFrame(cols_and_rows)
    return df


# DEPRECATED
def skus_from_wv_ids(wv_ids: list, retail_id: int,  host='db.geti.cl',
                     database='winodds', user='cvergara', password=MYSQL_PASSWORD):
    '''' extract data from mySQL based on the vector search result '''

    # formatting for the query
    if len(wv_ids) == 1:
        wv_ids = tuple(wv_ids)[0]
        wv_ids = f'"{wv_ids}"'
        aux = '='
    elif len(wv_ids) > 1:
        wv_ids = tuple(wv_ids)
        aux = 'IN'

    # Creating connection object
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    mycursor = mydb.cursor()

    query = f"SELECT wv.id,wv.sku, wv.retail_id, wv.state, wv.web_category_id, wc.name AS category_name,\
              wv.brand, wv.product_name, wv.variety_name, wv.url \
              FROM winodds.web_varieties wv \
              INNER JOIN winodds.web_categories wc ON wv.web_category_id = wc.id \
              WHERE wv.id {aux} {wv_ids} && wv.retaild_id={retail_id} && wv.discontinued = 0"

    mycursor.execute(query)
    results = mycursor.fetchall()

    # inserting the results into a dataframe
    cols_name = [i[0] for i in mycursor.description]
    cols_and_rows = [dict(zip(cols_name, result)) for result in results]
    df = pd.DataFrame(cols_and_rows)
    return df

# Preprocessing functions
#dict units
dict_units = {}
dict_units['u'] = ['u', 'un', 'und' 'unit', 'units', 'unidad', 'unidades']
dict_units['cm'] = ['cm', 'cms', 'centimeter', 'centimetros']
dict_units['l'] = ['l', 'lt', 'lts', 'litro', 'litros', 'litre', 'litres']
dict_units['m'] = ['m', 'mt', 'mts', 'metro', 'metros', 'meter', 'meters']
dict_units['gr'] = ['g', 'gr', 'grs', 'gramo', 'gramos', 'gram', 'grams']
dict_units['ml'] = ['ml', 'mls', 'mililitro', 'mililitros', 'millilitre', 'millilitres']
dict_units['kg'] = ['kg', 'kgs', 'kilo', 'k', 'kilos', 'kilogramo', 'kilogramos', 'kilogram', 'kilograms']
dict_units['lb'] = ['lb','lbs', 'libra', 'libras']
dict_units['cc'] = ['cc', 'ccs']
dict_units['mm'] = ['mm', 'mms', 'milimetro', 'milimetros', 'millimeter', 'millimeters']
dict_units['mg'] = ['mg', 'mgs', 'miligramo', 'miligramos', 'milligram', 'milligrams']
dict_units['gb'] = ['gb', 'gigabyte', 'gigabytes']
dict_units['kb'] = ['kb', 'kilobyte', 'kilobytes']
dict_units['mb'] = ['mb', 'megabyte', 'megabytes']
dict_units['tb'] = ['tb', 'terabyte', 'terabytes']
dict_units['w'] = ['w', 'watts']
dict_units['hz'] = ['hz', 'hertz']
dict_units['oz'] = ['oz', 'onza', 'onzas', 'ounce']

def get_key(val, dict):
    """returns the key of the dict_units dict"""
    for key, value in dict.items():
        if val in value:
            return key

def split_float(text, dict_units):
    """split skus quantity with unit, ex: 20grs, 20 grs"""
    for key, values in dict_units.items():
        pattern = '|'.join(values)
        #split skus quantity with unit, ex:
        text = re.sub(r'(\d+\.?\d*)(' + pattern + r')', r'\1 \2', text)
    return text

def split_combos(text):
    """split skus with promotions: ex: '6x100gr' , '6 x 100gr'"""
    text = re.sub(r'(\d+)x(\d+)', r'\1 x \2', text)
    return text


def remove_integer_pattern(text):
    """removes pattern of numbers (noise)"""
    # Define the regex pattern to match 4 or 5 integers followed by a space and another 4 or 5 integers
    pattern = r'\b\d{4,5} \d{4,5}\b'

    result = re.sub(pattern, '', text)
    # Remove any extra spaces left after removing the pattern
    result = re.sub(r'\s{2,}', ' ', result).strip()

    return result


def join_units(string, dict_units=dict_units):
    """joins numbers with following units (ej:250 grs = 250gr)"""

    # stopwords and dict for merging units to numbers in titles
    stop_words = ['and', 'for', 'the', 'with', 'to', 'y', 'de', 'que', 'en', 'para', 'del', 'le', 'les', 'lo', 'los',
                  'la', 'las', 'con', 'que', 'gratis', 'promo', 'promocion', 'promotion', 'oferta', 'ofertas', 'free', 'gratis',
                  'descuento', 'descuentos', 'dcto', 'pagina', 'page', 'null', 'price', 'precio', 'precios', 'producto',
                  'productos', 'product', 'products', 'combo', 'ceba'] #'especial', 'especiales', 'special', 'specials'

    dict_values = np.hstack(list(dict_units.values()))

    string_f_split = [token for token in string.split() if token not in stop_words]

    aux = []
    block_next = -1
    for index in range(len(string_f_split)):

        if index == block_next:  # skip word
            continue

        try:
            float_num = float(string_f_split[index])
            next_val = string_f_split[index + 1]
            try:
                next_float_num = float(next_val)
                if next_float_num:
                    val = string_f_split[index]
                    aux.append(val)
                    continue

            except:
                if index != len(string_f_split) and \
                        bool(re.search(r'\d', string_f_split[index + 1])) == False and \
                        string_f_split[index + 1] in dict_values:

                    val = str(float_num).rstrip('0').rstrip('.') + get_key(string_f_split[index + 1], dict_units)
                    aux.append(val)
                    block_next = index + 1
                    continue

                else:
                    val = string_f_split[index]
                    aux.append(val)
                    continue

        except:
            val = string_f_split[index]
            aux.append(val)
            continue

    formatted = ' '.join(aux)
    return formatted


def preprocess_products(row, dict_units=dict_units):
    """preprocess the product based on rules. Main function for preprocessing using the ones above"""
    row = row.lower()
    row = row.replace('unknown', '').replace(',', '.').replace('&nbsp', ' ').replace('\xa0', '').replace('"', '').\
        replace('solo delivery', '').replace('cyber monday', '').replace('&', 'y')
    # for pets
    row = row.replace("/", ' ').replace('**entrega en 2 dias habiles**', '').replace('**entrega en 4 dias habiles**','').\
        replace('**entrega en 6 dias habiles**', '').replace('**entrega en 7 dias habiles**', '').\
        replace('** entrega en 2 dias habiles**', '').replace('** entrega en 4 dias habiles**', '').\
        replace('** entrega en 6 dias habiles**', '').replace('** entrega en 7 dias habiles**', '').\
        replace(' ** entrega en 4 dias habiles **', '').replace('gabrica sas', '').replace('gabrica medicamentos', '').\
        replace('gabrica alimentos', '').replace('jaramillo pets cia sca', '').replace('synergylabs', '').\
        replace('verosa group sas', '').replace('select pets colombia sas', '').replace('italcol occidente sa', '').\
        replace('solo delivery', '').replace('colombia sas', '').replace('colombia sa', '').\
        replace('virbac colombia ltda', '').replace('concentrados norte 1 sas', '').replace('gesto agro sas','').\
        replace('industria colombiana farmaceutica icofar', '').replace('flatazor', '').\
        replace('alimentos italcol', '').replace('comercial mas bios s a', '').replace('colombia ltda', '')

    row = ''.join(re.findall('[-+]?(?:\d*\.\d+|\d+)|\d|\s|\w|\+', row))
    row = row.replace('-', ' ')
    row = re.sub(r'[_]', ' ', row)
    aux = ['mas' if token == '+' else token for token in row.split()]
    row = ' '.join(aux)

    row = unidecode.unidecode(row)  # removes special characters
    row = split_combos(row)
    row = split_float(row, dict_units)
    row = remove_integer_pattern(row)
    row = join_units(row)  # units formatting
    row = row.split()
    row = " ".join(sorted(set(row), key=row.index))  # removes duplicates

    return row


def preprocess_products_category_version(row):
    """preprocess the category name based on rules"""
    row = str(row).lower()
    row = unidecode.unidecode(row)
    row = row.replace('unknown','').replace(',','.') .replace('&nbsp', ' ').replace('\xa0','').replace('"','').\
        replace("/", ' ').replace('-', ' ').replace('|', ' ').replace('especiales','').replace('navidad','').\
        replace('ver todo', '').replace('invierno','').replace('primavera','').replace('verano','').replace('otono','').\
        replace('otoño','').replace('especial','').replace('temporadas','').\
        replace('temporada','').replace('nueva temporada','').replace('avance de temporada','').replace('cyber','').\
        replace('cyber monday','').replace('cyberday','').\
        replace('remate final','').replace('oferta','').replace('ofertas','').replace('+', '').replace('home','')  #revisar este ultimo! (home)

    row = ''.join(re.findall('[-+]?(?:\d*\.\d+|\d+)|\d|\s|\w|\+', row))
    row = join_units(row)
    row = re.sub(r'[^a-zA-Z\s]+', '', row)
    row = row.split()
    stop_words_1 = ['a', 'anos', 'ano', 'hasta', 'entre', 'meses', 'todo', 'ver', 'combo', 'combos', 'avance', 'null']
    stop_words_2 = ['u','cm','l','m','gr','ml','kg','lb','cc','mm','mg','gb','kb','mb','tb','w','hz','oz']
    stop_words_1.extend(stop_words_2)
    row = [token for token in row if token not in stop_words_1 and len(token) > 1]

    row = " ".join(sorted(set(row), key=row.index))  # removes duplicates

    if row == '':
      return 'unknown'

    if len(row) < 4:
      return 'unknown'
    return row

# Tensorflow Models
def get_available_gpus():
    """returns the available gpus for the API"""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# Loading Bert model and tokenizer from Hugging face & building automatch model
gpus_list = get_available_gpus()
# URL for getting: BERT multi & uncased from HuggingFace
bert_multi_url = 'bert-base-multilingual-uncased'

print('loading BERT Multilingual Model and tokenizer from HuggingFace...')
bert_multi_model = TFBertModel.from_pretrained(bert_multi_url, cache_dir=CACHE_DIR)
bert_tokenizer = BertTokenizer.from_pretrained(bert_multi_url, cache_dir=CACHE_DIR)

# normalized categories to predict.
with open('./dict_categories.pkl', 'rb') as f:
    dict_categories_norm = pickle.load(f)

with open('./dict_categories_inv.pkl', 'rb') as f:
    dict_categories_inv_norm = pickle.load(f)


def build_model_categories(model=bert_multi_model, target_dict=dict_categories_norm, seq_length=MAX_LENGTH):
    """Model architecture using BERT as backbone for predicting a normalized category"""
    classes = len(target_dict)

    input_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int64, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int64, name="attention_mask")

    embedding_layer = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    max_pool_txt = tf.keras.layers.GlobalMaxPool1D(name = 'max_pooling_txt')(embedding_layer)
    avg_pool_txt = tf.keras.layers.GlobalAveragePooling1D(name = 'avg_pooling_txt')(embedding_layer)
    concat_txt = tf.keras.layers.concatenate([max_pool_txt, avg_pool_txt])

    batch_norm_1_txt = tf.keras.layers.BatchNormalization(name='batch_norm1_txt')(concat_txt)
    dense_1_txt = tf.keras.layers.Dense(512, activation='relu', name='dense_1_txt')(batch_norm_1_txt)
    batch_norm_2_txt = tf.keras.layers.BatchNormalization(name='batch_norm2_txt')(dense_1_txt)
    dropout_1_txt = tf.keras.layers.Dropout(0.2, name='dropout_1_txt')(batch_norm_2_txt)

    #fully connected layer
    dense_fc = tf.keras.layers.Dense(256, activation='relu', name='dense_fc')(dropout_1_txt)
    batch_norm_fc = tf.keras.layers.BatchNormalization(name='batch_norm_fc')(dense_fc)
    dropout_fc = tf.keras.layers.Dropout(0.4, name='dropout_fc')(batch_norm_fc)

    output = tf.keras.layers.Dense(classes, activation='softmax', name='labels')(dropout_fc)

    classif_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    return classif_model

print('loading Custom Model for normalized categories predictions...')
if len(gpus_list) > 0:
    with tf.device(gpus_list[0]):
        cats_model = build_model_categories(model=bert_multi_model,target_dict=dict_categories_norm, seq_length=MAX_LENGTH)
        cats_model.load_weights('model_categories_txt_v5_acc.h5')  # loading model weights from training
else:  # if there is No GPU available -> use CPU

    cats_model = build_model_categories(model=bert_multi_model, target_dict=dict_categories_norm, seq_length=MAX_LENGTH)
    cats_model.load_weights('model_categories_txt_v5_acc.h5')


def encode_products(join, max_length=MAX_LENGTH, tokenizer=bert_tokenizer):
    """encondes an input using the BERT Tokenizer"""
    out = tokenizer(join, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    return out


def predict_product_category(cat, product, top, model, max_length=MAX_LENGTH,
                             tokenizer=bert_tokenizer, dict_categories=dict_categories_norm):
    """makes predictions of top categories based on: [brand_and_product + category_name]"""

    labels = np.array(list(dict_categories.keys()))
    cat_prepro = [preprocess_products_category_version(cat)]
    prod_prepro = [preprocess_products(product)]

    join = [f'{cat} [SEP] {prod}' for cat, prod in zip(cat_prepro, prod_prepro)]
    join_encoded = encode_products(join, max_length=max_length, tokenizer=tokenizer)

    predictions = model.predict([join_encoded['input_ids'], join_encoded['attention_mask']])

    preds = [(labels[int(cat)], val) for cat, val in enumerate(predictions[0])]
    preds.sort(key=lambda x: x[1], reverse=True)
    pred_top = preds[:top]

    return pred_top


# GCP embedding model (Not used rn. Still Ill leave it here, who knows)
# embedding_model_gcp = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# def get_embedding_txt_gcp(text, model=embedding_model_gcp):
#    """returns the embedding for a string using gcp embedding model"""
#    text = text.replace("\n", " ")
#    embedding = model.get_embeddings([text])
#    embedding = embedding[0].values
#    return embedding



