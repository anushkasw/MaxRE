import re

def flatten_list(labels):
    flattened = []
    for item in labels:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened

def get_id2prompt(rel2id, args):
    id2prompt = {}
    for name, id in rel2id.items():
        if args.task == 'wiki80':
            labels = name.split(' ')

        elif args.task == 'semeval_nodir':
            labels = name.split('-')

        elif args.task == 'FewRel':
            labels = name.split('_')

        elif args.task in ['NYT10', 'GIDS']:
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

        elif args.task == 'WebNLG':
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

        elif args.task == 'crossRE':
            if name == "win-defeat":
                labels = ['win', 'or', 'defeat']
            else:
                labels = name.split('-')

        elif args.task in ['tacred', 'tacrev', 'retacred', 'dummy_tacred', 'kbp37']:
            labels = [name.lower().replace("_", " ").replace("-", " ").replace("per", "person").replace("org",
                                                                                                        "organization").replace(
                "stateor", "state or ")]

        labels = [item.lower() for item in labels]

        if args.task == 'semeval_nodir':
            id2prompt[id] = ' and '.join(labels).upper()
        else:
            id2prompt[id] = ' '.join(labels).upper()
    return id2prompt

wiki_reltoid = {wiki20m_rel[i]:int(i+1) for i in range(len(wiki20m_rel))}
tacred_reltoid = {"NONE": 0, "per:title": 1, "per:city_of_death": 2, "org:shareholders": 3, "per:origin": 4, "org:top_members/employees": 5, "org:city_of_headquarters": 6, "per:religion": 7, "per:city_of_birth": 8, "per:employee_of": 9, "per:date_of_death": 10, "per:other_family": 11, "org:website": 12, "per:cause_of_death": 13, "org:subsidiaries": 14, "org:stateorprovince_of_headquarters": 15, "per:countries_of_residence": 16, "per:siblings": 17, "per:stateorprovinces_of_residence": 18, "org:alternate_names": 19, "per:spouse": 20, "per:parents": 21, "org:country_of_headquarters": 22, "per:age": 23, "per:date_of_birth": 24, "per:country_of_death": 25, "per:schools_attended": 26, "org:member_of": 27, "per:children": 28, "org:parents": 29, "per:cities_of_residence": 30, "per:stateorprovince_of_birth": 31, "per:charges": 32, "org:founded": 33, "org:founded_by": 34, "per:stateorprovince_of_death": 35, "org:members": 36, "per:country_of_birth": 37, "per:alternate_names": 38, "org:number_of_employees/members": 39, "org:dissolved": 40, "org:political/religious_affiliation": 41}
ace05_reltoid = {"NONE":0,"PHYS":1,"GEN-AFF":2,"PER-SOC":3,"ORG-AFF":4,"PART-WHOLE":5,"ART":6}
ace05_idtoprompt = {0:"NONE",1:"PHYSICAL",2:"GENERAL AND AFFILIATION",3:"PERSON AND SOCIAL",4:"ORGANIZATION AND AFFILIATION", 5:"PART AND WHOLE", 6:"AGENT AND ARTIFACT"}

# semeval_reltoid = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}

# semeval_idtoprompt = {0:"NONE",1:"CAUSE AND EFFECT", 2:"COMPONENT AND WHOLE", 3:"ENTITY AND DESTINATION",4:"ENTITY AND ORIGIN",5:"PRODUCT AND PRODUCER",6:"MEMBER AND COLLECTION",7:"MESSAGE AND TOPIC",8:"CONTENT AND CONTAINER",9:"INSTRUMENT AND AGENCY"}

scierc_reltoid = {"NONE": 0, "PART-OF": 1, "USED-FOR": 2, "FEATURE-OF": 3, "CONJUNCTION": 4, "EVALUATE-FOR": 5, "HYPONYM-OF": 6, "COMPARE": 7}

# query_list = generate_query(entity1_type, entity2_type, relation_list, query_dict)
# for query in query_list:
# task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output the number 0.\nChoice A: Physical Relationship\nChoice B: General-Affiliation Relationship\nChoice C: Person-Social Relationship\nChoice D: Organization-Affiliation Relationship\nChoice E: Part-Whole Relationship\nChoice F: Agent-Artifact Relationship\n"
# ace_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context" \
#                  ", I'll output the most precise relation between two entities choosing from the following six possible relations" \
#                  ".\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location" \
#                  "\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student " \
#                  "alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT:" \
#                  " user, owner, inventor, manufacturer\n"
# ace_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider " \
#                     "whether the most precise relation between two entities belongs to the following six possible relations. If yes, I will " \
#                     "output the most precise relation, otherwise I will output NONE.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, " \
#                     "resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION:" \
#                     " employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical," \
#                     "subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
# scierc_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the " \
#                        "most precise relation between two entities belongs to the following seven possible relations. If yes," \
#                        " I will output the most precise relation, otherwise I will output NONE.\n\nPART-OF: a part of\nUSED-FOR: " \
#                        "based on, models, trained on, used for\nFEATURE-OF: belong to, a feature of\nCONJUNCTION: similar role or incorporate with" \
#                        "\nEVALUATE-FOR: evaluate for\nHYPONYM-OF: a hyponym of, a type of\nCOMPARE: comapre with others\n"
# scierc_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation between two entities choosing from the following seven possible relations.\n\nPART-OF: a part of\nUSED-FOR: based on, models, trained on, used for\nFEATURE-OF: belong to, a feature of\nCONJUNCTION: similar role or incorporate with\nEVALUATE-FOR: evaluate for\nHYPONYM-OF: a hyponym of, a type of\nCOMPARE: comapre with others\n"

# task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
# choice_def = "CAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\n"
choice_def = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\nENTITY AND ORIGIN\nPRODUCT AND " \
             "PRODUCER\nMEMBER AND COLLECTION\nMESSAGE AND TOPIC\nCONTENT AND CONTAINER\nINSTRUMENT AND " \
             "AGENCY\n "
# choice_def = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\n"
choice_def_na = "CAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a " \
                "component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a " \
                "destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT " \
                "AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a " \
                "nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or " \
                "spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a " \
                "delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\nOTHER: other " \
                "possible relation types excluding these nine relations "
choice_reason = "CAUSE AND EFFECT\nCOMPONENT AND WHOLE\nENTITY AND DESTINATION\nENTITY AND ORIGIN\nPRODUCT " \
                "AND PRODUCER\nMEMBER AND COLLECTION\nMESSAGE AND TOPIC\nCONTENT AND CONTAINE\nINSTRUMENT AND" \
                " AGENCY\n "

# task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\n"
tacred_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, " \
                       "I'll output the most precise relation between two entities. If there is no relation between them, I will output NONE\n\n"
# task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities belongs to nine possible relations. If yes, I will output the most precise relation, otherwise I will output NONE.\n\n"
task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the " \
                     "context, " \
                     "I'll first consider whether the most precise relation between two entities belongs to following nine possible relations. " \
                     "If yes, I will output the most precise relation, otherwise I will output NONE.\n\n"

# task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll first consider whether the most precise relation between two entities. If yes, I will output the most precise relation, otherwise I will output OTHER.\n\n"
# task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, I'll output the most precise relation of the subject towards the object based on the context, choosing from nine possible relations. If all relations are not proper, I will output OTHER.\n\nCAUSE AND EFFECT: an event or object yields an effect\nCOMPONENT AND WHOLE: an object is a component of a larger whole\nENTITY AND DESTINATION: an entity is moving towards a destination\nENTITY AND ORIGIN: an entity is coming or is derived from an origin\nPRODUCT AND PRODUCER: a producer causes a product to exist\nMEMBER AND COLLECTION: a member forms a nonfunctional part of a collection\nMESSAGE AND TOPIC: an act of communication, written or spoken, is about a topic\nCONTENT AND CONTAINER: an object is physically stored in a delineated area of space\nINSTRUMENT AND AGENCY: an agent uses an instrument\n"
task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the " \
                  "context, " \
                  "I'll output the most precise relation between two entities based on the context, choosing from nine possible relations:\n"
tacred_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the " \
                    "context, " \
                    "I'll output the most precise relation between two entities based on the context\n"
# task_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all relations are not proper, I will output NONE.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
# task_def_choice = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from six possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\n"
task_def_others = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output the most precise relation of the subject towards the object based on the context, choosing from seven possible relations.\n\nPHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\nPERSON AND SOCIAL: business,family,lasting personal\nORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\nPART AND WHOLE: artifact,geographical,subsidiary\nAGENT AND ARTIFACT: user, owner, inventor, manufacturer\nOTHERS: the relation does not belongs to the previous six choices\n"
task_def = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, the subject and the object, I'll output one character of the most precise relation of the subject towards the object based on the context, choosing from six possible relations. If all choices are not proper, I will output None."

# gids_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context" \
#                  "I'll first consider whether the most precise relation between two entities belongs to following four possible relations. " \
#                      "If yes, I will output the most precise relation, otherwise I will output NONE." \
#                 "\n\n/people/deceased_person/place_of_death: person died in the place\n/people/person/place_of_birth: person was born in the place" \
#                 "\n/people/person/education./education/education/institution: person studied at the institution\n" \
#                      "/people/person/education./education/education/degree: person received the degree"

# gids_def_choice_na = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context" \
#                  "I'll first consider whether the most precise relation between two entities belongs to following four possible relations. " \
#                      "If yes, I will output the most precise relation, otherwise I will output NONE." \
#                 "\n\n/people/deceased_person/place_of_death: person died in the place\n/people/person/place_of_birth: person was born in the place" \
#                 "\n/people/person/education./education/education/institution: person studied at the institution\n" \
#                      "/people/person/education./education/education/degree: person received the degree"