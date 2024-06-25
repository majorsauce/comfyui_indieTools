from .cutByMask import CutByMask
from .pasteImage import PasteImage
from .localScale import LocalScale
from .solidify import Solidify
from .yolo_detect import YoloDetector

NODE_CLASS_MAPPINGS = { 
	"IndCutByMask": CutByMask,
	"IndPastImage": PasteImage,
	"IndLocalScale": LocalScale,
	"IndSolidify": Solidify,
	"IndYoloDetector": YoloDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = { 
	"IndCutByMask": "[Indie] Cut by Mask",
	"IndPastImage": "[Indie] Paste Image",
	"IndLocalScale": "[Indie] Local Scale",
	"IndSolidify": "[Indie] Solidify",
	"IndYoloDetector": "[Indie] Yolo Detector"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']