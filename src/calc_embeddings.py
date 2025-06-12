import csv
import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from WordTransformer import WordTransformer
from WordTransformer import InputExample
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate embeddings for DWUG English dataset")
    parser.add_argument("--input_dir", type=str, default="data/dwug_en", help="Input directory containing DWUG data")
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory for embeddings")
    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WordTransformer("pierluigic/xl-lexeme", device=device)

    src_gid = 1
    tgt_gid = 2
    source_token2vecs = defaultdict(list)
    target_token2vecs = defaultdict(list)

    for lemma_dir in tqdm(sorted((input_dir / "data").iterdir())):
        lemma = lemma_dir.stem
        csv_path = lemma_dir / "uses.csv"
        df = pd.read_table(csv_path, quoting=csv.QUOTE_NONE)
        
        for gid in [src_gid, tgt_gid]:
            if gid == src_gid:
                token2vecs = source_token2vecs
            else:
                token2vecs = target_token2vecs
            
            sentences = df[df["grouping"]==gid]["context"].tolist()
            indexes = df[df["grouping"]==gid]["indexes_target_token"].tolist()
            for sentence, index in tqdm(list(zip(sentences, indexes))):
                L, R = map(int, index.split(":"))
                input_example = InputExample(texts=sentence, positions=[L, R])
                vec = model.encode(input_example)
                token2vecs[lemma].append(vec)

    with open(output_dir/"dwug_en_embeddings.pkl", "wb") as f:
        pickle.dump((source_token2vecs, target_token2vecs), f)

if __name__ == "__main__":
    main()