import json
import os
import random
import math

from arguments import get_args_parser
from shared.const import get_id2prompt

def get_test_example(example_path, reltoid):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            for dict_ in tmp_dict:
                if dict_["relation"] in ['no_relation', 'Other']:
                    rel = "NONE"
                else:
                    rel = dict_["relation"]
                example_dict[reltoid[rel]].append(dict_)
    return example_dict


def run(reltoid, idtoprompt, store_path, args):
    test_path = f'{args.data_dir}/{args.task}/k-shot/seed-{args.data_seed}/test.json'
    test_dict = get_test_example(test_path, reltoid)

    test_examples = {}
    for sublist in test_dict.values():
        for item in sublist:
            test_examples[item['id']] = item

    batch_path = f'{args.reason_dir}/{args.task}/k-shot/output/{args.data_seed}-{args.k}.jsonl'

    with open(batch_path) as f:
        batch = f.read().splitlines()

    batch = [json.loads(line) for line in batch if line != '']

    test_res = []
    norels = []

    idtorel = {v: k for k, v in reltoid.items()}

    for test_sample in batch:
        sent_id = test_sample['custom_id']
        probs = test_sample['response']['body']['choices'][0]['logprobs']
        result = test_sample['response']['body']['choices'][0]['message']['content']

        choice = None
        for key in idtoprompt.keys():
            if idtoprompt[key].lower() in result.lower():
                choice = key

        if choice != None:
            pred = int(choice)
            prob = math.exp(probs['content'][0]['logprob'])
            rel = test_examples[sent_id]['relation']

            # if idtorel[pred]=='NONE':
            #     if args.task in ["sem_eval_task_8", "GIDS", "semeval_nodir"]:
            #         label_pred = 'Other'
            #     elif args.task in ["tacred", "tacrv", "retacred", "dummy_tacred", "kbp37"]:
            #         label_pred = 'no_relation'

            test_res.append({
                "id": sent_id,
                "label_true": rel,
                "label_pred": idtorel[pred],
                "probs": {idtorel[pred]: prob}
            })
        else:
            norels.append(sent_id)
            print(f"Predicted Rel was: {result[:20]}")

    print('Saving output')
    with open(f'{store_path}/GPT-RE_test.jsonl', 'w') as f:
        for res in test_res:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(res, f)

    with open(f'{store_path}/nores.jsonl', 'w') as f:
        for res in norels:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(res, f)


def main():
    args = get_args_parser()
    random.seed(args.seed)

    store_path = f'{args.output_dir}/{args.task}/{args.data_seed}-{args.k}'
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)

    with open(f'{args.data_dir}/{args.task}/rel2id.json', "r") as f:
        rel2id = json.loads(f.read())

    if args.task in ["sem_eval_task_8", "GIDS", "semeval_nodir"]:
        rel2id['NONE'] = rel2id.pop('Other')
        args.na_idx = rel2id['NONE']
    elif args.task in ["tacred", "tacrv", "retacred", "dummy_tacred", "kbp37"]:
        rel2id['NONE'] = rel2id.pop('no_relation')
        args.na_idx = rel2id['NONE']

    id2prompt = get_id2prompt(rel2id, args)
    run(rel2id, id2prompt, store_path, args)

    print('\n Done.')

if __name__ == "__main__":
    main()
