import os
import argparse
import torch
import torch_pruning as tp
import copy
import matplotlib.pyplot as plt
from models.yolo import Model
from models.yolo import Detect
from pathlib import Path
import sys
from utils.general import intersect_dicts


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def load_model(args):
    # Load the checkpoint
    ckpt = torch.load(args.weights, map_location=args.device)  # load checkpoint
    model = Model(ckpt['model'].yaml).to(args.device)  # create model
    state_dict = ckpt['model'].float().state_dict()  # convert to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect state_dicts
    model.load_state_dict(state_dict, strict=False)  # load model state
    
    # Set all parameters to require gradients
    for name, param in model.named_parameters():
        param.requires_grad = True

    return model


def prune_model(model, args):
    example_inputs = torch.randn(1, 3, 640, 640).to(args.device)
    imp = tp.importance.MagnitudeImportance(p=2)  # L2 norm pruning

    ignored_layers = []
    from models.yolo import Detect
    for m in model.modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)

    iterative_steps = 1  # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5,  # remove 50% channels
        ignored_layers=ignored_layers,
    )
    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for g in pruner.step(interactive=True):
        print(g)
        g.prune()

    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    print("Before Pruning: MACs=%f G, #Params=%f G" % (base_macs / 1e9, base_nparams / 1e9))
    print("After Pruning: MACs=%f G, #Params=%f G" % (pruned_macs / 1e9, pruned_nparams / 1e9))

    # Set all parameters to require gradients
    for name, param in model.named_parameters():
        param.requires_grad = True

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--save_path', default=ROOT / 'pruned_model.pt', type=str, help='path to save the pruned and fine-tuned model')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  

    args = parser.parse_args()

    # Load the model
    model = load_model(args)

    # Prune the model
    pruned_model = prune_model(model, args)

    # Save the pruned model
    ckpt = {
        'model': copy.deepcopy(pruned_model),
        'optimizer': None,
        'epoch': -1,
    }
    torch.save(ckpt, args.save_path)
    del ckpt
    print("Saved", args.save_path)
