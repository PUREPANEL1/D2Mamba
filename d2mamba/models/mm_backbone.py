"""MMSegmentation wrapper registering the D²Mamba encoder.

The encoder is the VSSD hierarchy (``Backbone_VMAMBA2``) whose stages 1-3
use the Spatial Deformable Mamba (SDM) mixer -- NC-SSD plus the DefMamba
deformable branch (DMSM) -- and whose stage 4 uses standard MHSA.
"""
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .vssd_backbone import Backbone_VMAMBA2


@MODELS.register_module()
class D2MambaBackbone(BaseModule, Backbone_VMAMBA2):

    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VMAMBA2.__init__(self, *args, **kwargs)
