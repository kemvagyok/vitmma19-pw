import requests
import zipfile
import io
import os
import json
from typing import List, Optional

import pandas as pd
import tensorflow as tf

import config

# ======================
# Data downloading utilities
# ======================

def ensure_data_dir():
    zip_url = 'https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A?e=Uu6CTj&download=1'


    zipping_data_dir = config.DATA_ROOT_DIR

    # --- Downloading és extracting ---

    try:
        # 1. Data downloading with request
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()  # Error raising, if it is 200 (for ex. 404, 500)

        # 2. The downloaded data is for zipfile extracting
        zip_in_memory = io.BytesIO(response.content)

        # 3. Extracting
        with zipfile.ZipFile(zip_in_memory, 'r') as zip_ref:
            # Create the target directory if it does not already exist
            if not os.path.exists(zipping_data_dir):
                os.makedirs(zipping_data_dir)
                print(f"Destination directory created: {zipping_data_dir}")

            # Extract to the specified destination directory
            zip_ref.extractall(zipping_data_dir)
            
            print("\n Download and extraction successful!")
            print(f"The files can be found here: {os.path.abspath(zipping_data_dir)}")

    except requests.exceptions.RequestException as e:
        print(f"Error in downloading: (for ex. bad URL, error network): {e}")
    except zipfile.BadZipFile:
        print("Error in the ZIP file extracting: The downloaded file is not good or not good ZIP archive.")
    except Exception as e:
        print(f":Unknown error {e}")
    

# ======================
# Annotation Task
# ======================
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
            self.studentImage = self.image_path.split('/')[0]

        # Extract label if available
        annotations = raw.get("annotations", [])
        self.label: Optional[str] = None
        if annotations and "result" in annotations[0] and len(annotations[0]["result"]) > 0:
            self.label = annotations[0]["result"][0]["value"]["choices"][0]

        # Metadata
        self.meta = raw.get("meta", {})
        self.created_at = raw.get("created_at")
        self.updated_at = raw.get("updated_at")

    def __repr__(self):
        return f"AnnotationTask(id={self.id}, label={self.label}, image_path='{self.image_path}')"

    def get_infos(self):
        return {
            "annotator": self.annotator,
            "owner_image": self.studentImage,
            "image_name": self.image_name,
            "label": self.label,
        }


def load_annotation_file(path: str, annotator: str) -> List[AnnotationTask]:
    """Betölt egy Label Studio JSON exportot AnnotationTask objektumokba."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [AnnotationTask(item, annotator) for item in items]

# ======================
# Label & Consensus Processing
# ======================
class LabelProcessor:
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

    @staticmethod
    def repair_labels(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df['label'].notna()]
        df['label'] = df['label'].replace(LabelProcessor.mapping)
        return df

    @staticmethod
    def repair_consensus(df: pd.DataFrame, consensus_neptuns: List[str]) -> pd.DataFrame:
        df = df.copy()
        df['owner_image'] = None
        for neptun in consensus_neptuns:
            mask = df['image_name'].str.contains(neptun)
            df.loc[mask, 'owner_image'] = neptun

        clean_names = df['image_name']
        for neptun in consensus_neptuns:
            clean_names = clean_names.str.replace(rf"{neptun}_?|_{neptun}", "", regex=True)
        clean_names = clean_names.str.replace("__", "_", regex=False).str.strip("_")
        df['clean_image_name'] = clean_names

        # Extra corrections (legacy hacks)
        df.loc[df['image_name'].str.contains('[0-9]_D6AE9F.jpg', regex=True), 'owner_image'] = 'OJHGS8'
        df.loc[df['annotator'].str.contains('GK1XQ4'), 'owner_image'] = 'D6AE9F'
        df.loc[df['image_name'].str.contains('resztvevo'), 'owner_image'] = 'FO6K58'
        df.loc[df['image_name'].str.contains('sajat_reszvevo_0[3-5]_?', regex=True), 'owner_image'] = 'ECSGGY'

        return df

    @staticmethod
    def filter_consensus(df: pd.DataFrame) -> pd.DataFrame:
        neptuns_as_filter = df['owner_image'].dropna().unique()
        pattern = "|".join(neptuns_as_filter)
        return df[df['image_name'].str.contains(pattern, regex=True)]

    @staticmethod
    def consensus_pipeline(tasks_list: List[dict], consensus_neptuns: List[str]) -> pd.DataFrame:
        df = pd.DataFrame(tasks_list)
        df = LabelProcessor.repair_consensus(df, consensus_neptuns)
        df = LabelProcessor.filter_consensus(df)
        df = LabelProcessor.repair_labels(df)
        return df

    @staticmethod
    def own_pipeline(tasks_list: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(tasks_list)
        df['owner_image'] = df['annotator']
        df = LabelProcessor.repair_labels(df)
        df['clean_image_name'] = df['image_name']
        df = df.drop(['annotator','image_name'], axis=1)
        return df


# ======================
# Image Utilities
# ======================
def load_image(path: str) -> Optional[tf.Tensor]:
    """Betölt egy képet JPEG vagy PNG formátumban és átméretezi."""
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
    except tf.errors.InvalidArgumentError:
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
        except tf.errors.InvalidArgumentError:
            print(f"Unsuccesful loading image: {path}")
            return None
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, config.TARGET_IMAGE_SIZE)
    return img


# ======================
# Dataset Utilities
# ======================
def creating_dataset(df: pd.DataFrame, data_dir: str) -> tf.data.Dataset:
    """DataFrame alapján TF Dataset létrehozása a képek és címkék párosításával."""
    images = []
    labels = []

    for idx, row in df.iterrows():
        owner = row['owner_image']
        img_name = row['clean_image_name']
        label = row['label']

        possible_paths = [
            os.path.join(data_dir, owner, img_name),
            os.path.join(data_dir, owner, img_name.replace('.jpg', '.jpeg')),
            os.path.join(data_dir, owner, img_name.replace('.jpeg', '.jpg')),
        ]
        #Képek betöltése
        for path in possible_paths:
            if os.path.exists(path):
                img = load_image(path)
                if img is not None:
                    if len(img.shape) == 4 and img.shape[0] == 1:
                        img = tf.squeeze(img, axis=0)
                    images.append(img)
                    labels.append(label)
                break

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset


def split_dataset(dataset, batch_size: int, train_ratio: float = 0.8):
    """TF Dataset feldarabolása train/val/test-re."""
    dataset_size = sum(1 for _ in dataset)
    dataset = dataset.shuffle(buffer_size=dataset_size, seed=42, reshuffle_each_iteration=False)

    train_size = int(train_ratio * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset = dataset.take(train_size).batch(batch_size, drop_remainder=True)
    val_dataset = dataset.skip(train_size).take(val_size).batch(batch_size, drop_remainder=False)
    test_dataset = dataset.skip(train_size + val_size).take(test_size).batch(batch_size, drop_remainder=False)

    return train_dataset, val_dataset, test_dataset
