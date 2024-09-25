import json
import random

from shared.prompt import instance
from utils import compute_variance
from knn_simcse import find_knn_example, find_lmknn_example

import openai


def get_train_example(example_path, reltoid):
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

## The following functions generate prompts for demonstrations
def auto_generate_example(example_dict, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo):
    # #ratio = 0.5
    # #num_per_rel = 4
    # num_example = num_per_rel * (len(example_dict.keys()) - 1) + num_na
    #
    #
    # #select_dict = {"0":0, "A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
    # #reltoalpha = {0:"0", 1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F"}
    # #reltoalpha = {0:"NONE", 1:"Physical", 2:"General and affiliation", 3:"Person and social", 4:"Organization and affiliation", 5:"Part and whole", 6:"Agent and artifact"}
    # #reltoalpha = {0:"NONE", 1:"PHYSICAL", 2:"GENERAL AND AFFILIATION", 3:"PERSON AND SOCIAL", 4:"ORGANIZATION AND AFFILIATION", 5:"PART AND WHOLE", 6:"AGENT AND ARTIFACT"}
    #        #else:
    #         #    if random.random() > 0.9:
    #         #        example_list.append(tmp_dict)
    #         #    else:
    #         #        continue
    # #examples = [item for k,v in example_dict.items() for item in v]
    # examples = []
    # for relid in example_dict.keys():
    #     if relid == 0:
    #         examples.append(random.sample(example_dict[relid], num_na))
    #     else:
    #         examples.append(random.sample(example_dict[relid], num_per_rel))
    #
    #
    # flat_examples = [item for sublist in examples for item in sublist]
    # #print(len(examples))
    # example_list = random.sample(flat_examples, num_example)
    # #assert False

    example_list = [item for sublist in example_dict.values() for item in sublist]
    example_prompt = str()
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["token"])
        sub_head = tmp_dict["subj_start"]
        sub_tail = tmp_dict["subj_end"] + 1


        obj_head = tmp_dict["obj_start"]
        obj_tail = tmp_dict["obj_end"] + 1

        entity1 = " ".join(tmp_dict["token"][sub_head:sub_tail])
        entity1_type = tmp_dict["subj_type"]
        entity2 = " ".join(tmp_dict["token"][obj_head:obj_tail])
        entity2_type = tmp_dict["obj_type"]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        else:
            rel = tmp_dict["relation"]


        if not reasoning:
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:"
            #print(prompt_query)
            #assert False

            results, probs = demo.get_multiple_sample(tmp_query)
            prompt_query = prompt_query + results[0] +"\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt


def generate_ft_example(tmp_dict, ft_dict, reltoid, idtoprompt, demo, args):
    tmp_example = instance(tmp_dict)

    example_list = ft_dict[tmp_example.id]
    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()
    if args.var:
        knn_distribution = []
        for tmp_dict in example_list:
            if tmp_dict["relations"] == [[]]:
                rel = 'NONE'
            else:
                rel = tmp_dict["relations"][0][0][4]
            knn_distribution.append(reltoid[rel])
        label_other = compute_variance(knn_distribution)
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1


        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if args.random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]
        tmp_knn.append(reltoid[rel])

        tmp_example = instance(tmp_dict)
        if not args.reasoning or label_other == 1:
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        elif args.self_error:
            prompt_query = tmp_example.get_self_error(tmp_dict, demo, reltoid, idtoprompt, args)
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            #print(prompt_query)
            #assert False

            while(True):
                try:
                    results, probs = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            #prompt_query = prompt_query + results[0] +"\n"
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list



def generate_lm_example(gpu_index_flat, tmp_dict, train_dict, train_sentences, k, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo, var, args):
    #train_list = [x for y in train_dict.values() for x in y]
    #print(tmp_dict)
    #assert False
    #print(len(train_list))
    example_list = find_lmknn_example(gpu_index_flat, tmp_dict,train_dict,train_sentences, k)

    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()
    if var:
        knn_distribution = []
        for tmp_dict in example_list:
            if tmp_dict["relations"] == [[]]:
                rel = 'NONE'
            else:
                rel = tmp_dict["relations"][0][0][4]
            knn_distribution.append(reltoid[rel])
        label_other = compute_variance(knn_distribution)
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1


        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]
        tmp_knn.append(reltoid[rel])

        tmp_example = instance(tmp_dict)
        if not reasoning or label_other == 1:
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            #print(prompt_query)
            #assert False

            while(True):
                try:
                    results, probs = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            #prompt_query = prompt_query + results[0] +"\n"
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list



def generate_knn_example(knn_model, tmp_dict, train_dict, k, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo, var, args, reason_list=None):
    '''
    For a test sample, this function creates prompts with knn demonstartions
    :param knn_model:
    :param tmp_dict:
    :param train_dict:
    :param k:
    :param reltoid:
    :param idtoprompt:
    :param num_per_rel:
    :param num_na:
    :param random_label:
    :param reasoning:
    :param demo:
    :param var:
    :param args:
    :return: example_prompt - (main prompt+context), tmp_knn - labels of the knn demonstrations, label_other, example_list - list of all demonstrations
    '''
    #train_list = [x for y in train_dict.values() for x in y]
    #print(tmp_dict)
    #assert False
    #print(len(train_list))
    example_list = find_knn_example(knn_model, tmp_dict,train_dict,k, args.entity_info)

    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()
    if var:
        knn_distribution = []
        for tmp_dict in example_list:
            if tmp_dict["relations"] == [[]]:
                rel = 'NONE'
            else:
                rel = tmp_dict["relations"][0][0][4]
            knn_distribution.append(reltoid[rel])
        label_other = compute_variance(knn_distribution)
    for tmp_dict in example_list:
        sent_id = tmp_dict['id']
        string = " ".join(tmp_dict["token"])
        sub_head = tmp_dict["subj_start"]
        sub_tail = tmp_dict["subj_end"] + 1


        obj_head = tmp_dict["obj_start"]
        obj_tail = tmp_dict["obj_end"] + 1

        entity1 = " ".join(tmp_dict["token"][sub_head:sub_tail])
        entity1_type = tmp_dict["subj_type"]
        entity2 = " ".join(tmp_dict["token"][obj_head:obj_tail])
        entity2_type = tmp_dict["obj_type"]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relation"] in ['no_relation', 'Other']:
            rel = 'NONE'
        else:
            rel = tmp_dict["relation"]
        tmp_knn.append(reltoid[rel])

        tmp_example = instance(tmp_dict)
        if not reasoning or label_other == 1:
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
                        #print(prompt_query)
            #assert False
            if reason_list:
                results = [reason_list[sent_id]]
            else:
                while(True):
                    try:
                        tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + \
                                    idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
                        results, _ = demo.get_multiple_sample(tmp_query, no_prob=True)
                        break
                    except openai.OpenAIError as e:
                        print(f"OpenAI API error: {e}")
                        raise
                    # except:
                    #     break
            # prompt_query = prompt_query + results +"\n"
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list