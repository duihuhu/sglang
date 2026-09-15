from __future__ import annotations
from . import native, pd, af, pdaf
from .recipes import RECIPES
from .matrix_workloads import (matrix_workloads_4gpu,
                               matrix_workloads_af_mooncake_tp)
from ..topology import PortAllocator
RECIPES["matrix_workloads_4gpu"] = matrix_workloads_4gpu
RECIPES["matrix_workloads_af_mooncake_tp"] = matrix_workloads_af_mooncake_tp
BUILDERS={"native":native.build,"pd":pd.build,"af":af.build,"pdaf":pdaf.build}
def build_plan(cluster, model, point, run_tag="dryrun"):
    cluster = dict(cluster)
    cluster["_run_tag"] = run_tag
    recipe = point.get("recipe")
    if recipe:
        try: return RECIPES[recipe](cluster, model, point)
        except KeyError as exc: raise ValueError(f"unknown deployment recipe: {recipe}") from exc
    return BUILDERS[point["architecture"]](cluster,model,point,PortAllocator())
