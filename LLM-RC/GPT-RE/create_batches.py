import json
import os
import random
import time

from shared.prompt import generate_select_auto_prompt
from arguments import get_args_parser
from data_loader import get_train_example, get_test_example, auto_generate_example, \
    generate_ft_example, generate_lm_example, generate_knn_example
from knn_simcse import get_demonstrations
from shared.const import get_id2prompt


def run(reltoid, idtoprompt, store_path, args):
    demo = None
    train_path = f'{args.data_dir}/{args.task}/train.json'
    test_path = f'{args.data_dir}/{args.task}/k-shot/seed-{args.data_seed}/test.json'

    example_dict = get_train_example(train_path, reltoid)  # train data
    test_dict = get_test_example(test_path, reltoid)
    test_examples = [item for sublist in test_dict.values() for item in sublist]

    with open(f'{args.reason_dir}/{args.task}/output.jsonl') as f:
        batch = f.read().splitlines()

    train_reason = [json.loads(line) for line in batch if line!=""]
    reason_list = {}
    for reason in train_reason:
        reason_list[reason['custom_id']] = reason['response']['body']['choices'][0]['message']['content']

    ## Create training prompts and demonstration retrieval models
    ft_dict, gpu_index_flat, train_dict, train_sentences, knn_model = get_demonstrations(args, example_dict, reltoid)

    # Testing starts here
    print("Number of test examples", len(test_examples))
    print('\n')

    start_time = time.time()

    os.makedirs(f'{store_path}/input', exist_ok=True)
    os.makedirs(f'{store_path}/demo_ids', exist_ok=True)

    whole_knn = {}

    for tmp_dict in test_examples:
        sent_id = tmp_dict['id']
        if tmp_dict["relation"] == "NONE" and args.no_na:
            continue
        if tmp_dict["relation"] != "NONE" and tmp_dict[
            "relation"] == "Other" and args.no_na:
            continue
        if tmp_dict["relation"] != "NONE" and tmp_dict[
            "relation"] != "Other" and args.null:
            continue
        if not args.fixed_example and not args.use_knn:
            example_prompt = auto_generate_example(example_dict, reltoid, idtoprompt, args.num_per_rel, args.num_na,
                                                   args.random_label, args.reasoning, demo)
        if args.use_knn:
            if args.use_ft:
                example_prompt, tmp_knn, label_other, knn_list = generate_ft_example(tmp_dict, ft_dict, reltoid,
                                                                                     idtoprompt, demo, args)
            elif args.lm_mask:
                example_prompt, tmp_knn, label_other, knn_list = generate_lm_example(gpu_index_flat, tmp_dict,
                                                                                     train_dict, train_sentences,
                                                                                     args.k, reltoid, idtoprompt,
                                                                                     args.num_per_rel, args.num_na,
                                                                                     args.random_label,
                                                                                     args.reasoning, demo, args.var,
                                                                                     args)
            else:
                example_prompt, tmp_knn, label_other, knn_list = generate_knn_example(knn_model, tmp_dict,
                                                                                      train_dict, args.k,
                                                                                      reltoid,
                                                                                      idtoprompt,
                                                                                      args.num_per_rel,
                                                                                      args.num_na,
                                                                                      args.random_label,
                                                                                      args.reasoning, demo,
                                                                                      args.var, args, reason_list)

        prompt_list, subject, target = generate_select_auto_prompt(tmp_dict, example_prompt, reltoid, args.no_na,
                                                                   args.reasoning, args)

        whole_knn[sent_id] = tmp_knn

        batch_dict = {"custom_id": sent_id, "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": args.model,
                               "messages": [{"role": "user", "content": prompt_list}], "temperature": 0,
                               "max_tokens": 256, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0,
                               "logprobs": True}}

        with open(f'{store_path}/input/{args.data_seed}-{args.k}.jsonl', 'a') as f:
            if f.tell() > 0:  # Check if file is not empty
                f.write('\n')
            json.dump(batch_dict, f)

    with open(f'{store_path}/demo_ids/{args.data_seed}-{args.k}.json', 'w') as file:
        json.dump(whole_knn, file)

    elapsed_time = time.time() - start_time
    print(f'Total time - {elapsed_time / 60}')


def main():
    args = get_args_parser()
    random.seed(args.seed)

    store_path = f'{args.output_dir}/{args.task}/k-shot'
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)

    try:
        if os.path.exists(f'{store_path}/input/{args.data_seed}-{args.k}.jsonl') and not args.overwrite_output_dir:
            raise FileExistsError(
                f"\n\nThe output file '{store_path}/input/{args.data_seed}-{args.k}.jsonl' already exists.\nUse the over_write flag or delete the existing file.")

        with open(f'{args.data_dir}/{args.task}/rel2id.json', "r") as f:
            rel2id = json.loads(f.read())

        if args.task in ["semeval_nodir", "GIDS"]:
            rel2id['NONE'] = rel2id.pop('Other')
            args.na_idx = rel2id['NONE']
        elif args.task in ["tacred", "tacrev", "retacred", "dummy_tacred", "kbp37_nodir"]:
            rel2id['NONE'] = rel2id.pop('no_relation')
            args.na_idx = rel2id['NONE']

        id2prompt = get_id2prompt(rel2id, args)

        run(rel2id, id2prompt, store_path, args)

    except FileExistsError as e:
        print(e)

    print('\nDone.')


if __name__ == "__main__":
    main()
