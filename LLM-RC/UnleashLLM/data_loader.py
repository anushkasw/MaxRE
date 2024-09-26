import json
import re

def flatten_list(labels):
    flattened = []
    for item in labels:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened

def get_train_example(example_path, reltoid):
    example_dict = {k:list() for k in reltoid.keys()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            for dict_ in tmp_dict:
                rel = dict_["relation"]
                example_dict[rel].append(dict_)
    return example_dict

def get_labels(args, rel2id):
    temps = {}
    for name, id in rel2id.items():
        if args.data_name=='wiki80':
            labels = name.split(' ')

        elif args.data_name == 'semeval_nodir':
            labels = name.split('-')

        elif args.data_name == 'FewRel':
            labels = name.split('_')

        elif args.data_name in ['NYT10', 'GIDS']:
            if name == 'Other':
                labels = ['None']
            elif name == '/people/person/education./education/education/institution':
                labels = ['person', 'and', 'education', 'institution']
            elif name == '/people/person/education./education/education/degree':
                labels = ['person', 'and', 'education', 'degree']
            else:
                labels = name.split('/')
                labels[-1] = "and_"+labels[-1]
                labels = labels[2:]
                for idx, lab in enumerate(labels):
                    if "_" in lab:
                        labels[idx] = lab.split("_")
                labels = flatten_list(labels)

        elif args.data_name == 'WebNLG':
            name_mod = re.sub(r"['()]", '', name)
            labels = name_mod.split(' ')

            if len(labels) == 1:
                label0 = labels[0]
                if "_" in label0:
                    labels = label0.split("_")

                    for idx, lab in enumerate(labels):
                        if any(char.isupper() for char in lab) and not lab.isupper():
                            l = re.split(r'(?=[A-Z])', lab)
                            if l[0] == "":
                                l = l[1:]
                            labels[idx] = l

                    labels = flatten_list(labels)

                elif any(char.isupper() for char in label0):
                    labels = re.split(r'(?=[A-Z])', label0)

        elif args.data_name == 'crossRE':
            if name == "win-defeat":
                labels = ['win', 'or', 'defeat']
            else:
                labels = name.split('-')

        elif args.data_name in ['tacred', 'tacrev', 'retacred', 'dummy_tacred']:
            labels = [name.lower().replace("_", " ").replace("-", " ").replace("per", "person").replace("org",
                                                                                                                  "organization").replace("stateor", "state or ")]

        labels = [item.lower() for item in labels]
        temps[name] = ' '.join(labels)

    return temps