from .augmentations import SSDAugmentation
from lib.utils.evaluate_utils import EvalVOC, EvalCOCO, EvalCLS

eval_solver_map = {'VOC0712': EvalVOC,
                   'COCO2014': EvalCOCO,
                  'DEEPV': EvalVOC,
                  'CLS': EvalCLS}


def eval_solver_factory(loader, cfg):
    Solver = eval_solver_map[cfg.DATASET.NAME]
    eval_solver = Solver(loader, cfg)
    return eval_solver
