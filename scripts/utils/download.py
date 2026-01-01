# import torchtext; torchtext.disable_torchtext_deprecation_warning()

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
# os.environ['https_proxy'] = '127.0.0.1:7897'


from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
# from torchtext.datasets import WikiText103

import argparse

from Timer import tprint

def download_longbench(root):
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", \
            "gov_report", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    for dataset in datasets:
        load_dataset('THUDM/LongBench', dataset).save_to_disk(root + dataset)
        
def download_longbench_e(root):
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    for dataset in datasets:
        dataset = dataset + "_e"
        data = load_dataset('THUDM/LongBench', dataset).save_to_disk(root + dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download models and datasets')
    parser.add_argument('--dataset', type=str, help='Dataset name to download')
    # parser.add_argument('--root', type=str, help='Root directory to save models and datasets')
    args = parser.parse_args()

    file_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(file_path))
    
    if args.dataset:
        if args.dataset == 'longbench':
            download_longbench(root='../../datasets/')
        elif args.dataset == 'longbench_e':
            download_longbench_e(root='../../datasets/') 
        else:
            raise ValueError('Dataset not supported')