from simcse import SimCSE
import faiss

from transformers import pipeline
import numpy as np
from tqdm import tqdm

from shared.prompt import instance
from utils import generate_ft_dict


def get_demonstrations(args, example_dict, reltoid):
    '''
    Trains the demonstration models
    :param args:
    :param example_dict:
    :param reltoid:
    :return:
    '''
    ft_dict, gpu_index_flat, train_dict, train_sentences, knn_model = None, None, None, None, None
    if args.use_ft:
        ft_dict = generate_ft_dict(args)
    elif args.use_knn:
        train_list = [x for y in example_dict.values() for x in y]
        if args.no_na:
            if args.task == "semeval":
                train_list = [x for x in train_list if reltoid[x["relations"][0][0][4]] != 0]
            else:

                train_list = [x for x in train_list if x["relations"] != [[]]]
        if not args.lm_mask:
            if args.entity_info:
                train_dict = {instance(x).reference: x for x in train_list}
                train_sentences = [instance(x).reference for x in train_list]
            else:
                train_dict = {instance(x).sentence: x for x in train_list}
                train_sentences = [instance(x).sentence for x in train_list]

            knn_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
            # knn_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
            knn_model.build_index(train_sentences, device="cuda")
        else:
            train_dict = {instance(x).lm_mask: x for x in train_list}
            train_sentences = [instance(x).lm_mask for x in train_list]

            res = faiss.StandardGpuResources()

            index_flat = faiss.IndexFlatL2(1024)
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

            extractor = pipeline(model="roberta-large", task="feature-extraction")
            embed_array = []
            for item in tqdm(train_sentences):
                result = extractor(item, return_tensors=True)

                embeds = result[0].detach().numpy().copy()
                embed_array.append(embeds[-3, :])

            embed_list = np.array(embed_array)
            gpu_index_flat.add(embed_list)
    return ft_dict, gpu_index_flat, train_dict, train_sentences, knn_model


def find_knn_example(model, test_dict, train_dict, k, entity_info):
    '''
    Finds k closest train sentences to the test example
    :param model:
    :param test_dict:
    :param train_dict:
    :param k:
    :param entity_info:
    :return:
    '''
    if entity_info:
        test_sentences = instance(test_dict).reference
    else:
        test_sentences = " ".join(test_dict["token"])
    test_id = test_dict["id"]
    label_other = 0
    # train_dict = {" ".join(x["sentences"][0]):x for x in train_list}
    # train_sentences = [x for x in train_dict.keys()]

    # print(len(test_sentences))
    # print(len(train_sentences))
    # model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    # model.build_index(train_sentences, device="cpu")

    # for x in test_sentences:

    #    knn_result = model.search(x, device="cpu", threshold=0.3, top_k=3)
    #    print(knn_result)
    #    assert False
    knn_result = model.search(test_sentences, device="cpu", threshold=0.0, top_k=k)
    # print(knn_result)
    knn_list = [train_dict[x[0]] for x in knn_result]
    # if var and not no_na:
    #    label_other = knn_variance(knn_list)

    # print(train_sentences[0])
    # print(knn_list)
    # assert False
    return knn_list


def find_lmknn_example(gpu_index_flat, test_dict, train_dict, train_sentences, k):
    test_sentence = instance(test_dict).lm_mask
    extractor = pipeline(model="roberta-large", task="feature-extraction")
    result = extractor(test_sentence, return_tensors=True)

    embed = result.detach().numpy().copy()
    xq = np.array([embed[0][-3]])

    print(xq.shape)
    D, I = gpu_index_flat.search(xq, k)
    print(I)

    knn_list = [train_dict[train_sentences[i]] for i in I[0, :k]]

    return knn_list
