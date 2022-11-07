#in main.py
import argparse
import datetime
import json
import os
import random
import time
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

import util.dist as dist
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.clevrref import ClevrRefEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator
from datasets.phrasecut_eval import PhrasecutEvaluator
from datasets.refexp import RefExpEvaluator
from engine import evaluate, train_one_epoch
from models import build_model
from models.postprocessors import build_postprocessors


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument("--do_qa", action="store_true", help="Whether to do question answering")
    parser.add_argument(
        "--predict_final",
        action="store_true",
        help="If true, will predict if a given box is in the actual referred set. Useful for CLEVR-Ref+ only currently.",
    )
    parser.add_argument("--no_detection", action="store_true", help="Whether to train the detector")
    parser.add_argument(
        "--split_qa_heads", action="store_true", help="Whether to use a separate head per question type in vqa"
    )
    parser.add_argument(
        "--combine_datasets", nargs="+", help="List of datasets to combine for training", default=["flickr"]
    )
    parser.add_argument(
        "--combine_datasets_val", nargs="+", help="List of datasets to combine for eval", default=["flickr"]
    )

    parser.add_argument("--coco_path", type=str, default="")
    parser.add_argument("--vg_img_path", type=str, default="")
    parser.add_argument("--vg_ann_path", type=str, default="")
    parser.add_argument("--clevr_img_path", type=str, default="")
    parser.add_argument("--clevr_ann_path", type=str, default="")
    parser.add_argument("--phrasecut_ann_path", type=str, default="")
    parser.add_argument(
        "--phrasecut_orig_ann_path",
        type=str,
        default="",
    )
    parser.add_argument("--modulated_lvis_ann_path", type=str, default="")

    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr_drop", default=35, type=int)
    parser.add_argument(
        "--epoch_chunks",
        default=-1,
        type=int,
        help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk",
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" frames',
    )

    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # Segmentation
    parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--masks", action="store_true")

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--set_loss",
        default="hungarian",
        type=str,
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss",
    )

    parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
    parser.add_argument(
        "--no_contrastive_align_loss",
        dest="contrastive_align_loss",
        action="store_false",
        help="Whether to add contrastive alignment loss",
    )

    parser.add_argument(
        "--contrastive_loss_hdim",
        type=int,
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )

    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # Loss coefficients
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--qa_loss_coef", default=1, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )
    parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

    # Run specific

    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    parser.add_argument("--output-dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--load", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=5, type=int)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser

parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
args = parser.parse_args(['--batch_size', '1', '--dataset_config', 'configs/gqa.json',
                          '--do_qa', '--ema', '--split_qa_heads', '--no_contrastive_align_loss',
                          '--resume' ,'https://zenodo.org/record/4721981/files/gqa_resnet101_checkpoint.pth',
                          '--eval'])
d = vars(args)
with open(args.dataset_config, "r") as f:
    cfg = json.load(f)
    d.update(cfg)
print(args)    
model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
model.load_state_dict(checkpoint["model"])
torch.save(model, 'gqa_model.pt')
model.cuda()
model.eval()
#model = torch.load('clevr_model.pt')
#model.eval()
