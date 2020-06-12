from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .fcn_gsf import *
from .psp import *
from .psp_gsf import *
from .encnet import *
from .deeplabv3 import *
from .deeplabv3_gsf import *
from .deeplabv3plus import*

from .gsnet import *
from .psaa import *
from .gsnet_aspp_base import *
from .gsnet_nosa import *
from .psaa_nosa import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'fcn_gsf': get_fcn_gsf,
        'encnet': get_encnet,
        'deeplabv3plus': get_deeplabv3plus,
        'deeplabv3': get_deeplabv3,
        'deeplabv3_gsf': get_deeplabv3_gsf,
        'psp': get_psp,
        'psp_gsf': get_psp_gsf,
        'gsnet': get_gsnetnet,
        'psaa': get_psaanet,
        'gsnet_aspp_base': get_gsnet_aspp_basenet,
        'gsnet_nosa': get_gsnet_nosanet,
        'psaa_nosa': get_psaa_nosanet,
    }
    return models[name.lower()](**kwargs)
