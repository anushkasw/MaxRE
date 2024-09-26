import json
import random
from tqdm import tqdm
import argparse
import os

from data_loader import get_train_example, get_labels


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def f1_score(true, pred_result, rel2id):
    correct = 0
    total = len(true)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable']:
        if name in rel2id:
            neg = rel2id[name]
            break
    for i in range(total):
        golden = true[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    return result


def main(args):
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

    batch = []
    for input in tqdm(test):
        sent_id = input['id']

        random.shuffle(rels)
        if "text" in args.prompt:
            prompt = "There are candidate relations: " + ', '.join(labelword2rel.keys()) + ".\n"
        else:
            prompt = "Given a context, a pair of head and tail entities in the context, decide the relationship between the head and tail entities from candidate relations: " + \
                     ', '.join(labelword2rel.keys()) + ".\n"
        for rel in rels:
            random.shuffle(label_list[rel])
            kshot = label_list[rel][:args.k]
            for data in kshot:
                ss, se = data['subj_start'], data['subj_end']
                head = ' '.join(data['token'][ss:se + 1])
                headtype = data['subj_type'].lower().replace('_', ' ')
                if headtype == "misc":
                    headtype = "miscellaneous"
                elif headtype == 'O':
                    headtype = "unkown"
                os, oe = data['obj_start'], data['obj_end']
                tail = ' '.join(data['token'][os:oe + 1])
                tailtype = data['obj_type'].lower().replace('_', ' ')
                if tailtype == "misc":
                    tailtype = "miscellaneous"
                elif tailtype == 'O':
                    tailtype = "unkown"
                sentence = ' '.join([convert_token(token) for token in data['token']])
                relation = rel2labelword[data['relation']]
                if "schema" in args.prompt:
                    prompt += "Context: " + sentence + " The relation between " + headtype + " '" + head + "' and " + tailtype + " '" + tail + "' in the context is " + relation + ".\n"
                else:
                    prompt += "Context: " + sentence + " The relation between '" + head + "' and '" + tail + "' in the context is " + relation + ".\n"
                # prompt += " The relation between '" + head + "' and '" + tail + "' in the context '" + sentence + "' is " + relation + ".\n"

        tss, tse = input['subj_start'], input['subj_end']
        testhead = ' '.join(input['token'][tss:tse + 1])
        testheadtype = input['subj_type'].lower().replace('_', ' ')
        if testheadtype == "misc":
            testheadtype = "miscellaneous"
        elif testheadtype == 'O':
            testheadtype = "unkown"
        tos, toe = input['obj_start'], input['obj_end']
        testtail = ' '.join(input['token'][tos:toe + 1])
        testtailtype = input['obj_type'].lower().replace('_', ' ')
        if testtailtype == "misc":
            testtailtype = "miscellaneous"
        elif testtailtype == 'O':
            testtailtype = "unkown"
        testsen = ' '.join(input['token'])
        if "schema" in args.prompt:
            prompt += "Context: " + testsen + " The relation between " + testheadtype + " '" + testhead + "' and " + testtailtype + " '" + testtail + "' in the context is "
        else:
            prompt += "Context: " + testsen + " The relation between '" + testhead + "' and '" + testtail + "' in the context is "
            # prompt += " The relation between '" + testhead + "' and '" + testtail + "' in the context '" + testsen + "' is "
        # print(prompt)

        batch_dict = {"custom_id": sent_id, "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": "gpt-3.5-turbo",
                  "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 128, "logprobs": True}}

        batch.append(batch_dict)

    with open(f'{args.output_dir}/{args.data_name}/k-shot/input/{args.data_seed}-{args.k}.jsonl', 'w') as f:
        for line in batch:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dir', type=str, required=True, help="The path of training / test data.")
    parser.add_argument('--data_name', '-t', type=str, required=True, help="Dataset name")
    parser.add_argument('--output_dir', '-os', type=str, required=True,
                        help="The output directory of successful ICL samples.")
    parser.add_argument('--prompt', type=str, required=True,
                        choices=["text", "text_schema", "instruct", "instruct_schema"])
    parser.add_argument('--k', type=int, default=1, help="k-shot demonstrations")
    parser.add_argument('--data_seed', type=int, default=1, help="few shot seed")
    args = parser.parse_args()

    path = f'{args.output_dir}/{args.data_name}/k-shot/input'
    os.makedirs(path, exist_ok=True)

    main(args)