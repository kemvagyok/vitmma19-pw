# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
from utils import setup_logger
from preprocessingTools import AnnotationTask,load_annotation_file, labelsRepairing, consesusLabelsPipeline, ownLabelsPipeline, load_image
import os
import config
import pandas as pd
import tensorflow as tf

logger = setup_logger()

def getConsensusData(consensus_dir, exclude, consensus_neptuns):
    files = [
            f for f in os.listdir(consensus_dir)
            if f not in exclude
        ]

    annotators_tasks = []

    for filename in files:
        annotator = filename.rsplit('.', 1)[0].upper()
        tasks = load_annotation_file(f"{consensus_dir}/{filename}", annotator)
        annotators_tasks.extend([t.get_infos() for t in tasks])

    df_consensus = consesusLabelsPipeline(annotators_tasks, consensus_neptuns)

    return df_consensus


def getOwnLabels(data_dir, exclude):
    neptuns = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d not in exclude
    ]

    annotators_tasks = []

    for neptun in neptuns:
        folder = os.path.join(data_dir, neptun)
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]

        if not json_files:
            print(f"Nincs JSON fájl: {neptun}")
            continue

        json_path = os.path.join(folder, json_files[0])

        tasks = load_annotation_file(json_path, neptun)
        annotators_tasks.extend(t.get_infos() for t in tasks)

        # --- DataFrame ---
    df_ownLabels = ownLabelsPipeline(annotators_tasks)
    return df_ownLabels
    

def getDataset(df, data_dir):
    images = []
    #owners = []
    #names = []
    labels = []
    for index in range(len(df)):
        owner = df['owner_image'].iloc[index]
        img_name = df['clean_image_name'].iloc[index]
        label = df['label'].iloc[index]
        # próbáljuk .jpg-t és .jpeg-t is
        possible_paths = [
            os.path.join(data_dir, owner, img_name),
            os.path.join(data_dir, owner, img_name.replace('.jpg', '.jpeg')),
            os.path.join(data_dir, owner, img_name.replace('.jpeg', '.jpg'))
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                img = load_image(path)
                if img is not None:
                    images.append(img)
                    #owners.append(owner)
                    #names.append(img_name)
                    labels.append(label)
                break
                
    # Egységesítés
    images_clean = []
    for img in images:
        # Ha van extra dimenzió (pl. [1,256,256,3]), azt eltávolítjuk
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = tf.squeeze(img, axis=0)
        images_clean.append(img)

    #owners_ = pd.DataFrame(owners)
    #names_ = pd.DataFrame(names)
    dataset = tf.data.Dataset.from_tensor_slices((images_clean, labels))
    return dataset

def preprocess():
    logger.info("Preprocessing data...")
    exclude_consensus = {'anklealign-consensus.txt', 'DPMC6W.json', 'ITWQ3V.json'}
    exclude_ownLabels = {'consensus', 'sample', 'NC1O2T', 'ECSGGY', 'GI9Y8B'}
    consensus_neptuns = ['D6AE9F','ECSGGY','FO6K58']
    
    df_consensus = getConsensusData(f"{config.DATA_DIR}/consensus", exclude_consensus, consensus_neptuns)
    df_ownLabels = getOwnLabels(config.DATA_DIR,exclude_ownLabels)


    vote_counts = (
            df_consensus.groupby(['owner_image', 'image_name', 'clean_image_name', 'label'])
            .size()
            .reset_index(name='votes')
        )
    
    final_df = []

    for neptun in vote_counts['owner_image'].unique():
        idx = (
            vote_counts[vote_counts['owner_image'] == neptun]
            .groupby('clean_image_name')['votes']
            .idxmax()
        )
        top_labels = vote_counts.loc[idx]
        final_df.append(top_labels)

    final_df = pd.concat(final_df, ignore_index=True)
    final_df = final_df.drop(['votes', 'image_name'], axis=1)
    final_df = pd.concat([final_df, df_ownLabels])
    print(len(df_consensus),len(df_ownLabels),(len(final_df)))
    dataset = getDataset(final_df,config.DATA_DIR)
    print(len(dataset))
    dataset.save(f"{config.DATA_DIR}/dataset")
    logger.info("Preprocessing complete")

if __name__ == "__main__":
    preprocess()
