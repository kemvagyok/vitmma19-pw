import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Optional


class AnnotationTask:
    def __init__(self, raw: dict, annotator: str):
        self.annotator = annotator
        self.studentImage = None
        self.raw = raw

        self.id: int = raw.get("id")
        self.image_path: str = raw.get("data", {}).get("image")
        if len(self.image_path.split('/')) == 5: 
            self.image_name = self.image_path.split('/')[4].split('-')[1]
        else:
             self.image_name = self.image_path.split('/')[1].split('-')[1]
             self.studentImage =  self.image_path.split('/')[0]
        # Extract label (if available)
        annotations = raw.get("annotations", [])
        if annotations and "result" in annotations[0]:
            if len(annotations[0]["result"]) != 0:
                result = annotations[0]["result"][0]
                self.label: Optional[str] = result["value"]["choices"][0]
            else:
                self.label = None

        # Full metadata
        self.meta = raw.get("meta", {})
        self.created_at = raw.get("created_at")
        self.updated_at = raw.get("updated_at")

    def __repr__(self):
        return f"AnnotationTask(id={self.id}, label={self.label}, image_path='{self.image_path}')"

    def get_infos(self):
        return {
            "annotator" : self.annotator, 
            "owner_image" : self.studentImage,
            "image_name" : self.image_name, 
            "label" : self.label,
        }

def load_annotation_file(path: str, student: str) -> List[AnnotationTask]:
    """Load a Label Studio JSON export and convert to Python objects."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    tasks = [AnnotationTask(item, student) for item in items]
    return tasks


def labelsRepairing(df):
    df = df.copy()

    # 1. None sorok eltávolítása
    df = df[df['label'].notna()]

    # 2. Komplett egységesítési táblázat
    mapping = {
        # Pronáció
        'pronation': '1_Pronacio',
        '1_Pronacio': '1_Pronacio',
        '1_Pronáló': '1_Pronacio',
        '1_Pronáló ': '1_Pronacio',

        # Neutral
        'neutral': '2_Neutralis',
        '2_Neutralis': '2_Neutralis',
        '2_Neutrális': '2_Neutralis',

        # Szupináció
        'supination': '3_Szupinacio',
        '3_Szupinacio': '3_Szupinacio',
        '3_Szupináló': '3_Szupinacio',
        '3_Szupináló ': '3_Szupinacio'
    }

    # 3. Replace
    df['label'] =  df['label'].replace(mapping)
    return df


def consensusReparing(df, consensus_neptuns):# 1) owner_image kitöltése
    df['owner_image'] = None
    for neptun in consensus_neptuns:
        mask = df['image_name'].str.contains(neptun)
        df.loc[mask, 'owner_image'] = neptun

    # 2) clean_image_name létrehozása és tisztítása
    clean_names = df['image_name']

    # NEPTUN-ok eltávolítása bárhol
    for neptun in consensus_neptuns:
        clean_names = clean_names.str.replace(
            rf"{neptun}_?|_{neptun}", 
            "",
            regex=True
        )
    ## resztvevo -> reszvevo javítás
    #clean_names = clean_names.str.replace("resztvevo", "reszvevo", regex=True)

    # dupla _ javítása és elejéről/végéről _ eltávolítása
    clean_names = clean_names.str.replace("__", "_", regex=False).str.strip("_")

    df['clean_image_name'] = clean_names

    #3.) Extra

    mask_extra = df['image_name'].str.contains('[0-9]_D6AE9F.jpg', regex=True)
    df.loc[mask_extra, 'owner_image'] = 'OJHGS8'

    mask_extra = df['annotator'].str.contains('GK1XQ4')
    df.loc[mask_extra, 'owner_image'] = 'D6AE9F'

    mask_extra = df['image_name'].str.contains('resztvevo')
    df.loc[mask_extra, 'owner_image'] = 'FO6K58'
    #4.) Extra 2: Az én elnevezési hibám miatt ki tudtam deríteni, hogy bizonyos én fájlaimhoz melyik címkék tartoznak: sajat_resztvevo_03, _04, és _05.
    #Ellenőriztem is, hogy a mások nem követtek el ilyen hibát.
    mask_extra = df['image_name'].str.contains('sajat_reszvevo_0[3-5]_*', regex=True)
    df.loc[mask_extra, 'owner_image'] = 'ECSGGY'

    return df


def consensusFiltering(df):
    neptuns_as_filter = df['owner_image'].unique()[df['owner_image'].unique()!=None]
    pattern = "|".join(neptuns_as_filter)
    return df[df['image_name'].str.contains(pattern, regex=True)]



def consesusLabelsPipeline(tasksList, consensus_neptuns):
    df_consensus = pd.DataFrame(tasksList)
    df_consensus = consensusReparing(df_consensus, consensus_neptuns)
    df_consensus = consensusFiltering(df_consensus)
    df_consensus = labelsRepairing(df_consensus) 
    return df_consensus


def ownLabelsPipeline(tasksList):
    df_own_neptun = pd.DataFrame(tasksList)
    df_own_neptun['owner_image'] = df_own_neptun['annotator']
    df_own_neptun = labelsRepairing(df_own_neptun)
    df_own_neptun['clean_image_name'] = df_own_neptun['image_name']
    df_own_neptun = df_own_neptun.drop(['annotator','image_name'], axis= 1) 
    return df_own_neptun


def load_image(path, target_size=(256, 256)):
    # próbálkozás JPEG dekódolással
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
    except tf.errors.InvalidArgumentError:
        # ha nem JPEG, próbáljuk PNG-vel
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
        except tf.errors.InvalidArgumentError:
            print(f"Nem sikerült betölteni a képet: {path}")
            return None
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size)
    return img