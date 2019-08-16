# from .resnet_models import resnext101_32x8d as resnext101_32x8d
# from .resnet_models import resnext50_32x4d as resnext50_32x4d
from .resnet_models import resnet152 as resnet152
from .resnet_models import resnet101 as resnet101
from .resnet_models import resnet50 as resnet50
from .resnet_models import resnet34 as resnet34
from .resnet_models import resnet18 as resnet18

from .resnet_utils import resnet_save as save
from .resnet_utils import resnet_load as load
from .resnet_utils import resnet_train as train
from .resnet_utils import resnet_eval as eval
from .resnet_utils import resnet_representation as represent

__version__ = '1.0.0'
__author__ = 'Ali Hedayatnia, B.Sc student @ University of Tehran'