import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils

# In order to run on Windows, we had to create this if name==main guard
# to prevent the "start a new process before the current process has finished bootstrapping" error
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
    argp.add_argument('variant', help="Choose vanilla or perceiver")
    argp.add_argument('--bottleneck_dim', type=int, default=32)
    argp.add_argument('pretrain_corpus_path', default=None)
    argp.add_argument('--reading_params_path',default=None)
    argp.add_argument('--writing_params_path',default=None)
    argp.add_argument('--finetune_corpus_path', default=None)
    argp.add_argument('--eval_corpus_path', default=None)
    argp.add_argument('--outputs_path', default=None)
    argp.add_argument('--pretrain_lr', default=6e-3, type=float)
    argp.add_argument('--finetune_lr', default=6e-4, type=float)
    argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                      default='run')
    args = argp.parse_args()

    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)} (device {torch.cuda.current_device()})")
    else:
        print("Using CPU")
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/%s/%s_%s_%d_pt_lr_%f_ft_lr_%f' % (
        args.function,
        args.tb_expt_name,
        args.variant,
        args.bottleneck_dim,
        args.pretrain_lr,
        args.finetune_lr))

    # Keep the block size 128
    # Why is the pretraining corpus always required (even if we're not pretraining?)
    # It's because we're using it as a hack to always have the same vocabulary
    # (that is, the same mapping from character to integer, and we build the
    # vocab from the pretraining corpus.)
    block_size = 128
    text = open(args.pretrain_corpus_path, encoding='utf-8').read()
    pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

    # We don't suggest you change these hyperparameters, as they're known to work.
    # use them for both the vanilla and the perceiver models
    mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
        n_layer=4, n_head=8, n_embd=256)

    """
    Don't change above here; write your code below
    """

    # define models.
    # note: models should moved to device defined on line 34.
    gpt_model = None
    if args.variant == 'vanilla':
        gpt_model = model.GPT(mconf).to(device)
    elif args.variant == 'perceiver':
        # set mconf.perceiver, and mconf.bottleneck_dim parameters appropriately.
        pass # Make some other model here
    else:
        raise ValueError("Unknown model variant")

    # Perform pretraining, finetuning, or evaluation
    if args.function == 'pretrain':
        assert args.writing_params_path is not None
        # - Given:
        #     1. A corpus specified in args.pretrain_corpus_path
        #     2. An output path args.writing_params_path for the model parameters
        # - Goals:
        #     1. Pretrain the model on this corpus
        #     2. Save the resulting model in args.writing_params_path

        # - Make sure to use the following hyperparameters for pretraining:
        # Hyperparameters for pretraining:
        # max_epochs=650
        # batch_size=128
        # learning_rate=args.pretrain_lr
        # lr_decay=True
        # warmup_tokens=512*20
        # final_tokens=200*len(pretrain_dataset)*block_size
        # num_workers=4
        # writer=writer
        pt_config = trainer.TrainerConfig(
            max_epochs = 650,
            batch_size = 128,
            learning_rate = args.pretrain_lr,
            lr_decay = True,
            warmup_tokens = 512*20,
            final_tokens = 200*len(pretrain_dataset)*block_size,
            num_workers = 4,
            writer = writer,
        )
        pt_config.ckpt_path = args.writing_params_path
        # pt_config.ckpt_path = None   # disable per-epoch autosaves
        trainer_obj = trainer.Trainer(
            model = gpt_model,
            train_dataset = pretrain_dataset,
            test_dataset = None,
            config = pt_config
        )
        trainer_obj.train()
        # one final save:
        torch.save(gpt_model.state_dict(), args.writing_params_path)
        print("Pretraining complete, checkout ", args.writing_params_path)
    elif args.function == 'finetune':
        assert args.writing_params_path is not None
        assert args.finetune_corpus_path is not None
        # - Given:
        #     1. A finetuning corpus specified in args.finetune_corpus_path
        #     2. A path args.reading_params_path containing pretrained model
        #         parameters, or None if finetuning without a pretrained model
        #     3. An output path args.writing_params_path for the model parameters
        # - Goals:
        #     1. If args.reading_params_path is specified, load these parameters
        #         into the model
        #     2. Finetune the model on this corpus
        #     3. Save the resulting model in args.writing_params_path
        # - Make sure to use the following hyperparameters:
        #     Hyperparameters for finetuning WITHOUT a pretrained model:
        #         max_epochs=75
        #         batch_size=256
        #         learning_rate=args.finetune_lr
        #         lr_decay=True
        #         warmup_tokens=512*20
        #         final_tokens=200*len(pretrain_dataset)*block_size
        #         num_workers=4
        #         writer=writer
        #     Hyperparameters for finetuning WITH a pretrained model:
        #         max_epochs=10
        #         batch_size=256
        #         learning_rate=args.finetune_lr
        #         lr_decay=True
        #         warmup_tokens=512*20
        #         final_tokens=200*len(pretrain_dataset)*block_size
        #         num_workers=4
        #         writer=writer
        #     You can use the args.reading_params_path flag to switch between the
        #     number of epochs for each case.
        # If a pretrained checkpoint is provided, load it:
        if args.reading_params_path is not None:
            # if not os.path.exists(args.reading_params_path):
            #     torch.save(gpt_model.state_dict(), args.reading_params_path)
            print(f"Loading pretrained weights from {args.reading_params_path} ...")
            checkpoint = torch.load(args.reading_params_path, map_location=device)
            gpt_model.load_state_dict(checkpoint)
            print("Pretrained weights loaded.")
        # Build the NameDataset (uses the same stoi/itos and block_size as pretrain_dataset):
        train_data = open(args.finetune_corpus_path, encoding='utf-8').read()
        name_dataset = dataset.NameDataset(pretrain_dataset, train_data)
        # Choose hyperparameters depending on whether we loaded a checkpoint or not:
        if args.reading_params_path is None:
            # Hyperparameters for finetuning WITHOUT a pretrained model:
            ft_max_epochs = 75
        else:
            # Hyperparameters for finetuning WITH a pretrained model:
            ft_max_epochs = 10
        ft_batch_size = 256
        ft_learning_rate = args.finetune_lr
        ft_lr_decay = True
        ft_warmup_tokens = 512 * 20
        ft_final_tokens = 200 * len(pretrain_dataset) * block_size
        ft_num_workers = 4
        ft_config = trainer.TrainerConfig(
            max_epochs=ft_max_epochs,
            batch_size=ft_batch_size,
            learning_rate=ft_learning_rate,
            lr_decay=ft_lr_decay,
            warmup_tokens=ft_warmup_tokens,
            final_tokens=ft_final_tokens,
            num_workers=ft_num_workers,
            writer=writer,
        )
        ft_config.ckpt_path = args.writing_params_path
        # we don't run epoch('test') - we’ll evaluate later
        ft_trainer = trainer.Trainer(gpt_model, name_dataset, None, ft_config)
        ft_trainer.train()
        ft_trainer.save_checkpoint()
    elif args.function == 'evaluate':
        assert args.outputs_path is not None
        assert args.reading_params_path is not None
        assert args.eval_corpus_path is not None
        gpt_model.load_state_dict(torch.load(args.reading_params_path))
        correct = 0
        total = 0
        with open(args.outputs_path, 'w', encoding='utf-8') as fout:
            predictions = []
            for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
                x = line.split('\t')[0]
                x = x + '⁇'
                x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
                pred = utils.sample(gpt_model, x, 32, sample=False)[0]
                completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
                pred = completion.split('⁇')[1]
                predictions.append(pred)
                fout.write(pred + '\n')
            total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
        if total > 0:
          print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
        else:
            print('Predictions written to {}; no targets provided'
                    .format(args.outputs_path))

