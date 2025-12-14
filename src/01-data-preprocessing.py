# preprocess.py
# Data preprocessing script
from utils import setup_logger
from dataUtils import (
    load_annotation_file,
    LabelProcessor,
    creating_dataset,
    ensure_data_dir
)
import os
import config
import pandas as pd
import tensorflow as tf

logger = setup_logger()


def get_consensus_data(consensus_dir: str, exclude: set, consensus_neptuns: list) -> pd.DataFrame:
    """Betölti a konszenzus adatokat és pipeline-on keresztül tisztítja."""
    files = [f for f in os.listdir(consensus_dir) if f not in exclude]

    annotators_tasks = []
    for filename in files:
        annotator = filename.rsplit('.', 1)[0].upper()
        tasks = load_annotation_file(os.path.join(consensus_dir, filename), annotator)
        annotators_tasks.extend([t.get_infos() for t in tasks])

    df_consensus = LabelProcessor.consensus_pipeline(annotators_tasks, consensus_neptuns)
    return df_consensus


def get_own_labels(data_dir: str, exclude: set) -> pd.DataFrame:
    """Betölti az egyéni annotációkat."""
    neptuns = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d not in exclude
    ]

    annotators_tasks = []
    for neptun in neptuns:
        folder = os.path.join(data_dir, neptun)
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]

        if not json_files:
            logger.warning(f"Nincs JSON fájl: {neptun}")
            continue

        json_path = os.path.join(folder, json_files[0])
        tasks = load_annotation_file(json_path, neptun)
        annotators_tasks.extend([t.get_infos() for t in tasks])

    df_own = LabelProcessor.own_pipeline(annotators_tasks)
    return df_own


def preprocess():
    logger.info("Preprocessing data...")
    #--- Data downloading ---
    ensure_data_dir()
    # --- Config ---
    exclude_consensus = {'anklealign-consensus.txt', 'DPMC6W.json', 'ITWQ3V.json'}
    exclude_ownLabels = {'consensus', 'sample', 'NC1O2T', 'ECSGGY', 'GI9Y8B'}
    consensus_neptuns = ['D6AE9F', 'ECSGGY', 'FO6K58']

    # --- Load data ---
    df_consensus = get_consensus_data(os.path.join(config.DATA_DIR, 'consensus'),
                                      exclude_consensus,
                                      consensus_neptuns)
    df_own = get_own_labels(config.DATA_DIR, exclude_ownLabels)

    # --- Konszenzus voting ---
    vote_counts = (
        df_consensus.groupby(['owner_image', 'image_name', 'clean_image_name', 'label'])
        .size()
        .reset_index(name='votes')
    )

    final_df_list = []
    for neptun in vote_counts['owner_image'].unique():
        idx = (
            vote_counts[vote_counts['owner_image'] == neptun]
            .groupby('clean_image_name')['votes']
            .idxmax()
        )
        top_labels = vote_counts.loc[idx]
        final_df_list.append(top_labels)

    final_df = pd.concat(final_df_list, ignore_index=True)
    final_df = final_df.drop(['votes', 'image_name'], axis=1)

    # --- Merge with own labels ---
    if config.WITH_OWN_LABELS:
        final_df = pd.concat([final_df, df_own], ignore_index=True)

    logger.info(f"{len(final_df)} labels after consensus voting (with own labels: {config.WITH_OWN_LABELS})")
    # --- Create TF Dataset ---
    dataset = creating_dataset(final_df, config.DATA_DIR)
    logger.info(f"{len(list(dataset))} images processed and added to the dataset")
    logger.info(f"{len(final_df) - len(list(dataset))} was skipped due to errors or missing files.")
    # --- Save TF Dataset ---
    dataset_path = os.path.join(config.DATA_DIR, 'dataset')
    tf.data.Dataset.save(dataset, dataset_path)
    logger.info(f"Preprocessing complete. Dataset saved to {dataset_path}")


if __name__ == "__main__":
    preprocess()
