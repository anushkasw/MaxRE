import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, required=True,
                        choices=["dummy_data", "ace05", "semeval_nodir", "tacred", "scierc", 'tacrev',
                                 'retacred', 'wiki80', 'dummy_tacred',
                                 'FewRel', 'GIDS', 'WebNLG', 'crossRE', 'NYT10', "kbp37_nodir"])
    parser.add_argument("--model", default=None, type=str, required=False)
    parser.add_argument("--api_key", default=None, type=str, required=False)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default=None, required=True)
    parser.add_argument("--fixed_example", type=int, default=1)
    parser.add_argument("--fixed_test", type=int, default=1)
    parser.add_argument("--num_per_rel", type=int, default=2)
    parser.add_argument("--num_na", type=int, default=0)
    parser.add_argument("--no_na", type=int, default=0)
    parser.add_argument("--num_run", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_label", type=int, default=0)
    parser.add_argument("--reasoning", type=int, default=0)
    parser.add_argument("--use_knn", type=int, default=0)
    parser.add_argument("--lm_mask", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--bert_sim", type=int, default=1)
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--reverse", type=int, default=0)
    parser.add_argument("--verbalize", type=int, default=0)
    parser.add_argument("--entity_info", type=int, default=0)
    parser.add_argument("--structure", type=int, default=0)
    parser.add_argument("--use_ft", type=int, default=0)
    parser.add_argument("--self_error", type=int, default=0)
    parser.add_argument("--use_dev", type=int, default=0)
    parser.add_argument("--store_error_reason", type=int, default=0)
    parser.add_argument("--discriminator", type=int, default=0)
    parser.add_argument("--name", type=str, default=0)
    parser.add_argument("--null", type=str, default=1)
    parser.add_argument("--na_idx", type=int, default=None)

    parser.add_argument("--output_dir", default='./result', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--reason_dir", default='./batches_reason', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    # TODO: make dynamic


    args = parser.parse_args()
    if args.null == 1:
        args.null = True
    else:
        args.null = False
    if args.lm_mask == 1:
        args.lm_mask = True
    else:
        args.lm_mask = False
    if args.verbalize == 1:
        args.verbalize = True
    else:
        args.verbalize = False

    if args.entity_info == 1:
        args.entity_info = True
    else:
        args.entity_info = False
    if args.reverse == 1:
        args.reverse = True
    else:
        args.reverse = False
    if args.var and args.no_na:
        raise Exception("Sorry, if focus on no NA examples, please turn var into 0")
    if args.var:
        args.var = True
    else:
        args.var = False
    if args.fixed_example and args.use_knn:
        print("Can't use fixed examples and knn at the same time")
        assert False
    if args.fixed_example == 1:
        args.fixed_example = True
    else:
        args.fixed_example = False

    if args.fixed_test == 1:
        args.fixed_test = True
    else:
        args.fixed_test = False

    if args.reasoning == 1:
        args.reasoning = True
    else:
        args.reasoning = False

    if args.no_na == 1:
        args.no_na = True
    else:
        args.no_na = False

    if args.random_label == 1:
        args.random_label = True
    else:
        args.random_label = False
    if args.no_na and args.num_na != 0:
        print(args.no_na)
        print(args.num_na)
        assert False

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS