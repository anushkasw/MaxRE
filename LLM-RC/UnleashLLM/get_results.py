import json
import os
import math
import argparse

from data_loader import get_train_example, get_labels

def main(args):
    store_path = f'{args.output_dir}/{args.data_name}/{args.data_seed}-{args.k}'
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)

    # Relations
    with open(f'{args.data_dir}/{args.data_name}/rel2id.json', "r") as f:
        rel2id = json.loads(f.read())

    # Train / Demostration Set
    train_path = f'{args.data_dir}/{args.data_name}/k-shot/seed-{args.data_seed}/{args.k}-shot/train.json'
    label_list = get_train_example(train_path, rel2id)

    # Label words
    rels = list(rel2id.keys())
    rel2labelword = get_labels(args, rel2id)

    labelword2rel = {}
    for k, v in rel2labelword.items():
        labelword2rel[v] = k

    # Test Set
    test_path = f'{args.data_dir}/{args.data_name}/k-shot/seed-{args.data_seed}/test.json'
    with open(test_path, 'r') as f:
        test = json.load(f)

    test_examples = {}
    for sublist in test:
        test_examples[sublist['id']] = sublist

    batch_path = f'{args.input_dir}/{args.data_name}/k-shot/output/{args.data_seed}-{args.k}.jsonl'

    with open(batch_path) as f:
        batch = f.read().splitlines()

    batch = [json.loads(line) for line in batch if line != '']

    test_res = []
    norels = []

    for test_sample in batch:
        sent_id = test_sample['custom_id']
        probs = test_sample['response']['body']['choices'][0]['logprobs']
        result = test_sample['response']['body']['choices'][0]['message']['content']

        try:
            pred = result[:-1]
            pred_rel = labelword2rel[pred]
            prob = math.exp(probs['content'][0]['logprob'])
            rel = test_examples[sent_id]['relation']

            test_res.append({
                "id": sent_id,
                "label_true": rel,
                "label_pred": pred_rel,
                "probs": {pred_rel: prob}
            })

        except:
            norels.append(sent_id)
            print(f"Predicted Rel was: {result}")

    with open(f'{store_path}/DeepKE_test.jsonl', 'w') as f:
        for res in test_res:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(res, f)

    with open(f'{store_path}/nores.jsonl', 'w') as f:
        for res in norels:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(res, f)

    print('\n Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dir', type=str, required=True, help="The path of training / test data.")
    parser.add_argument('--data_name', '-t', type=str, required=True, help="Dataset name")
    parser.add_argument('--input_dir', '-is', type=str, required=True,
                        help="The output directory of successful ICL samples.")
    parser.add_argument('--output_dir', '-os', type=str, required=True,
                        help="The output directory of successful ICL samples.")
    parser.add_argument('--k', type=int, default=1, help="k-shot demonstrations")
    parser.add_argument('--data_seed', type=int, default=1, help="few shot seed")
    args = parser.parse_args()

    main(args)
