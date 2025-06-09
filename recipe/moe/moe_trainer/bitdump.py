# from dill import pickle as dill_pickle
import copyreg
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy, deepcopy

import dill
import dill.settings
import torch
import torch.distributed



class NoneReducer:
    @staticmethod
    def reduce(pg):
        return (NoneReducer.rebuild, (None,))

    @staticmethod
    def rebuild(state):
        return None


copyreg.pickle(torch._C._distributed_c10d.ProcessGroup, NoneReducer.reduce)
copyreg.pickle(torch.cuda.Stream, NoneReducer.reduce)
copyreg.pickle(torch._C._functions.AccumulateGrad, NoneReducer.reduce)
copyreg.pickle(torch.cuda.Event, NoneReducer.reduce)
copyreg.pickle(torch._C.DispatchKeySet, NoneReducer.reduce)


dill.settings["byref"] = True
dill.settings["recurse"] = True
dill.settings["ignore"] = True


def get_children_layers(model: torch.nn.Module, name=""):
    named_children = dict(model.named_children())
    children = named_children.values()
    names = named_children.keys()
    if len(children) == 0:
        output_names, output_children = [name], [model]
    else:
        output_names, output_children = [name], [model]
        for n, c in zip(names, children):
            res_n, res_c = get_children_layers(c, n)
            for ni, ci in zip(res_n, res_c):
                full_name = f"{name}.{ni}" if name != "" else ni
                output_names.append(full_name)
                output_children.append(ci)

    return output_names, output_children


def remove_attrs(m, names):
    for n in names:
        if hasattr(m, n):
            delattr(m, n)


def remove_hook_from_module(module: torch.nn.Module, recurse=False):
    if hasattr(module, "_old_forward"):
        module.forward = module._old_forward
        module.forward.__self__.forward = module._old_forward
        delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)


mbs_ids = {}


def hook_fwd_bwd_to_module(model: torch.nn.Module, names=None, prefix="", is_hf=False):
    def name_fn(name, direction="forward", is_hf=False):
        def fn(module, input_features, output_features):

            flag = True
            node = torch._C._current_autograd_node()
            if flag and name is not None and name != "" and name != " ":
                print(f"===== dump {name} datas {node=}")
                if prefix and not os.path.exists(prefix):
                    os.makedirs(prefix, exist_ok=True)

                key = (name, direction)
                mbs_ids.setdefault(key, 0)
                print(
                    f"{prefix}{name}-iter-mbs{mbs_ids[key]}-{direction}-input.pt"
                )
                torch.save(
                    input_features,
                    f"{prefix}{name}-iter-mbs{mbs_ids[key]}-{direction}-input.pt",
                    pickle_module=dill,
                )
                torch.save(
                    output_features,
                    f"{prefix}{name}-iter-mbs{mbs_ids[key]}-{direction}-output.pt",
                    pickle_module=dill,
                )
                mbs_ids[key] += 1

        return fn

    if isinstance(names, str):
        names = [names]

    all_names, _ = get_children_layers(model)

    new_names = []
    if names is None:
        new_names = all_names
    else:
        for n in all_names:
            for t in names:
                if t.endswith("*"):
                    if n.startswith(t[:-1]):
                        new_names.append(n)
                    if n == t[:-2]:
                        new_names.append(n)
                else:
                    if n == t:
                        new_names.append(n)

    modules = dict(model.named_modules())
    for name in new_names:
        if name in modules.keys():
            modules[name].register_forward_hook(name_fn(name, is_hf=is_hf))
            modules[name].register_full_backward_hook(
                name_fn(name, "backward", is_hf=is_hf), prepend=True
            )
