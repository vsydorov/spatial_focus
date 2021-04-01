import logging
import pandas as pd

import vst

from avid.tools import snippets
from avid.data.external_dataset import (
        DatasetHMDB51, DatasetCharades, DatasetCharadesTest)

log = logging.getLogger(__name__)


def precompute_dataset_stats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset: [~, ['charades', 'charades_test', 'hmdb51']]
    """)
    cf = cfg.parse()

    if cf['dataset'] == 'charades':
        dataset = DatasetCharades()
    elif cf['dataset'] == 'charades_test':
        dataset = DatasetCharadesTest()
    else:
        dataset = DatasetHMDB51()
    dataset.precompute_to_folder(out)
