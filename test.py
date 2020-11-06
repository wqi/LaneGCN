# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pickle
import sys
from importlib import import_module

import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import ArgoTestDataset
from utils import Logger, load_pretrain


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="angle90", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model()

    # load pretrain model
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    # Data loader for evaluation
    dataset = ArgoTestDataset(args.split, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # begin inference
    metrics = []
    for ii, data in tqdm(enumerate(data_loader)):
        data = dict(data)

        with torch.no_grad():
            output = net(data)
            results = [x[0:1].detach().cpu() for x in output["reg"]]

        for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
            preds = pred_traj.squeeze()
            truth = data["gt_preds"][i][0] if "gt_preds" in data else None
            truth = truth.unsqueeze(0).repeat(preds.shape[0], 1, 1)

            # Compute metrics for all agents and scenarios
            l2_all = torch.sqrt(torch.sum((preds - truth)**2, dim=-1))
            ade_all = torch.sum(l2_all, dim=-1) / preds.size(-2)
            fde_all = l2_all[..., -1]
            min_fde_idx = torch.argmin(fde_all, dim=-1).unsqueeze(-1)
            fde = torch.gather(fde_all, -1, min_fde_idx).squeeze(-1)
            ade = torch.gather(ade_all, -1, min_fde_idx).squeeze(-1)
            miss = int(fde > 2)
            metrics.append((argo_idx, (fde, ade, miss)))

    # Compute per-agent metrics
    agent_mean_fde = np.mean(np.array([x[1][0] for x in metrics]))
    agent_mean_ade = np.mean(np.array([x[1][1] for x in metrics]))
    agent_mr = np.mean(np.array([x[1][2] for x in metrics]))

    print(f'Agent FDE: {agent_mean_fde}')
    print(f'Agent ADE: {agent_mean_ade}')
    print(f'Agent MR: {agent_mr}')

    # Compute per-scenario metrics
    scenario_metrics = defaultdict(list)
    for entry in metrics:
        scenario_metrics[entry[0]].append(entry[1])

    wc_fde = np.array([np.max([y[0] for y in x]) for x in scenario_metrics.values()])
    wc_mean_fde = np.mean(wc_fde)
    wc_mean_ade = np.mean([np.max([y[1] for y in x]) for x in scenario_metrics.values()])
    wc_mr = np.sum(wc_fde > 2) / wc_fde.shape[0]

    print(f'Worst-Case Scenario FDE: {wc_mean_fde}')
    print(f'Worst-Case Scenario ADE: {wc_mean_ade}')
    print(f'Worst-Case Scenario MR: {wc_mr}')


if __name__ == "__main__":
    main()
