import json
import os
from pathlib import Path
import random
import argparse
import time

import openai

from gpt_api import Demo

import logging


def setup_logger(log_file_path_and_name, logger_name, level=logging.ERROR, formatter=None):
    log_path, _ = os.path.split(log_file_path_and_name)

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a file handler
    os.makedirs(log_path, exist_ok=True, mode=0o755)
    # file_handler = RotatingFileHandler(log_file_path_and_name)
    file_handler = logging.FileHandler(log_file_path_and_name)
    file_handler.setLevel(level)

    # Create a logging format
    if formatter:
        formatter = logging.Formatter(formatter)
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    return logger


def main(args):
    random.seed(args.seed)

    demo = Demo(
        api_key=args.api_key,
        engine=args.model,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=True,
    )

    store_path = f'{args.output_dir}/{args.task}/k-shot'
    os.makedirs(f'{store_path}/output', exist_ok=True)

    # Setup loggers
    logging_dir = f'{store_path}/logs/output'
    os.makedirs(logging_dir, exist_ok=True)
    config_done_logger = setup_logger(f'{logging_dir}/output_done_paths.log', 'config_done_logger', level=logging.INFO, formatter='%(message)s')
    config_redo_logger = setup_logger(f'{logging_dir}/output_redo_paths.log', 'config_redo_logger', level=logging.INFO, formatter='%(message)s')

    # done_samples, redo_samples = [], []
    # if os.path.exists(f'{logging_dir}/output_done_paths.log'):
    #     with open(f'{logging_dir}/output_done_paths.log', 'r') as file:
    #         log_content = file.read()
    #     done_samples = [line.split("Sample: ")[1] for line in log_content.strip().split('\n')]
    #
    # if os.path.exists(f'{logging_dir}/output_redo_paths.log'):
    #     with open(f'{logging_dir}/output_redo_paths.log', 'r') as file:
    #         log_content = file.read()
    #     redo_samples = [line.split("Sample: ")[1] for line in log_content.strip().split('\n')]


    batch_files = list(Path(f'{store_path}/input').glob('*-1.jsonl')) # only 1 shot batches
    print(f'Total batches = {len(batch_files)}')

    for file in batch_files:
        if not os.path.exists(f'{store_path}/output/{file.name}'):
            print(f'Processing: {file}')
            start_time = time.time()
            result = demo.process_batch(file)
            if result:
                config_done_logger.info(f'{file},{(time.time()-start_time)/60}\n')
                result_file_name = f'{store_path}/output/{file.name}'
                with open(result_file_name, 'w') as f:
                    f.write(result + '\n')
            else:
                config_redo_logger.info(f'{file}: error')
        else:
            print(f'Output already present for file - {file}')

    print('\n Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, required=True,
                        choices=["ace05", "sem_eval_task_8", "tacred", "scierc", 'tacrev',
                         'retacred', 'wiki80', 'dummy_tacred',
                         'FewRel', 'GIDS', 'WebNLG', 'crossRE', 'NYT10'])
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--api_key", default=None, type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default='./batches', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")


    args = parser.parse_args()
    main(args)