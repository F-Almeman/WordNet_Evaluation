import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from MirrorWiC.evaluation_scripts.src.helper import get_embed


def encode_example(df):
    new_examples = []
    punctuations = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“"
    for _, row in df.iterrows():
        word = row['WORD'].strip().translate(str.maketrans('', '', punctuations))
        for w in row['EXAMPLE'].split():
            if w.translate(str.maketrans('', '', punctuations)).strip().startswith(word):
                new_example = row['EXAMPLE'].replace(w, f'[ {w} ]', 1)
                break
        new_examples.append(new_example)
    return new_examples


def make_example_embeddings(input_df, tokenizer, model):
    vecs = []
    for i, row in input_df.iterrows():
        encoded_example = row.NEW_EXAMPLE
        try:
            vec = get_embed([encoded_example], tokenizer, model, flag='token', layer_start=9, layer_end=13, maxlen=64)
            vecs.append(vec)
        except:
            vecs.append(np.zeros((1, 768)))
            continue
        if i % 1000 == 0:
            print('Done ', i, 'of', len(input_df))
    return vecs


def evaluate_similarity(datasets, dictionaries):
    all_res = []
    for dataset_name, dataset in datasets:
        for _, row in dataset.iterrows():
            word1 = row["word1"]
            word2 = row["word2"]
            gold_sim = row.sim
            for dname, d in dictionaries:
                result_word1 = d[d['WORD'] == word1]
                result_word2 = d[d['WORD'] == word2]
                cand_vecs_w1 = result_word1.VECTOR.values
                cand_examples_w1 = result_word1.NEW_EXAMPLE.values
                cand_vecs_w2 = result_word2.VECTOR.values
                cand_examples_w2 = result_word2.NEW_EXAMPLE.values

                best_sim = -9999
                ex1_for_best_sim = ''
                ex2_for_best_sim = ''
                for vec1_idx, vec1 in enumerate(cand_vecs_w1):
                    for vec2_idx, vec2 in enumerate(cand_vecs_w2):
                        this_sim = 1 - cosine(vec1[0], vec2[0])
                        if this_sim > best_sim:
                            best_sim = this_sim
                            ex1_for_best_sim = cand_examples_w1[vec1_idx]
                            ex2_for_best_sim = cand_examples_w2[vec2_idx]

                all_res.append({
                    'word1': word1,
                    'word2': word2,
                    'sim': gold_sim,
                    'pred_sim': best_sim,
                    'ex_w1': ex1_for_best_sim,
                    'ex_w2': ex2_for_best_sim,
                    'dataset': dataset_name,
                    'dictionary': dname
                })
    return pd.DataFrame(all_res)


def compute_correlations(results_df, output_file='corr.txt'):
    with open(output_file, 'a') as file:
        for dataset in results_df['dataset'].unique():
            df_dataset = results_df[results_df['dataset'] == dataset]
            print(f"Calculating for dataset: {dataset}", file=file)
            for dictionary in df_dataset['dictionary'].unique():
                df_dict = df_dataset[df_dataset['dictionary'] == dictionary]
                sim_values = df_dict['sim']
                pred_sim_values = df_dict['pred_sim']
                pearson_corr, _ = pearsonr(sim_values, pred_sim_values)
                spearman_corr, _ = spearmanr(sim_values, pred_sim_values)
                print(f"  For dictionary: {dictionary}", file=file)
                print(f"    Pearson Correlation: {pearson_corr}", file=file)
                print(f"    Spearman Correlation: {spearman_corr}", file=file)


def main(args):
    model = AutoModel.from_pretrained(args.model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    df = pd.read_csv(args.input_words)
    df['WORD'] = df['WORD'].str.strip()
    dfgpt = pd.read_csv(args.input_gpt)
    sim_df = pd.read_csv(args.similarity_file)
    sim_df['word1'] = sim_df['word1'].str.strip()
    sim_df['word2'] = sim_df['word2'].str.strip()

    wn_df = df[df["SOURCE"] == "WordNet"]
    cha_df = df[df["SOURCE"] == "CHA"]
    gptsimple_df = dfgpt[dfgpt["SOURCE"] == "gpt-simple"]
    gptgdex_df = dfgpt[dfgpt["SOURCE"] == "gpt_gdex"]

    sim_words = set(sim_df["word1"]).union(set(sim_df["word2"]))

    simlex = sim_df[sim_df["dataset"] == "simlex"]
    men = sim_df[sim_df["dataset"] == "men"]
    simverb = sim_df[sim_df["dataset"] == "simverb"]
    scws = sim_df[sim_df["dataset"] == "scws"]

    def prepare_embeddings(source_df, label):
        print(f"-- {label} --")
        df_filtered = source_df[source_df['WORD'].isin(sim_words)]
        df_filtered['NEW_EXAMPLE'] = encode_example(df_filtered)
        df_filtered = df_filtered.reset_index(drop=True)
        embeddings = make_example_embeddings(df_filtered, tokenizer, model)
        df_filtered['VECTOR'] = embeddings
        return df_filtered

    wn_df_sim = prepare_embeddings(wn_df, 'wn')
    cha_df_sim = prepare_embeddings(cha_df, 'cha')
    gptsimple_df = prepare_embeddings(gptsimple_df, 'gptsimple')
    gptgdex_df = prepare_embeddings(gptgdex_df, 'gptgdex')

    datasets = [('simlex', simlex), ('men', men), ('simverb', simverb), ('scws', scws)]
    dictionaries = [('wn', wn_df_sim), ('cha', cha_df_sim), ('gptsimple', gptsimple_df), ('gptgdex', gptgdex_df)]

    results_df = evaluate_similarity(datasets, dictionaries)
    results_df.to_csv(args.output_csv, index=False)

    compute_correlations(results_df, args.output_corr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run similarity evaluation using MirrorWiC embeddings.")
    parser.add_argument('--input_words', type=str, required=True, help='Path to wn_cha_common.csv')
    parser.add_argument('--input_gpt', type=str, required=True, help='Path to sim_examples_gpt_simple_and_gdex.csv')
    parser.add_argument('--similarity_file', type=str, required=True, help='Path to similarity_datasets.csv')
    parser.add_argument('--model_name', type=str, default='cambridgeltl/mirrorwic-bert-base-uncased', help='Model name or path')
    parser.add_argument('--output_csv', type=str, default='results_similarity_experiment.csv', help='Output CSV file for results')
    parser.add_argument('--output_corr', type=str, default='corr.txt', help='Output file for correlation results')
    args = parser.parse_args()

    main(args)

