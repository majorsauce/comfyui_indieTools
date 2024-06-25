import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .utils import normalize_image_tensor, normalize_mask_tensor

class Solidify:

	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"optional": {
				"images": ("IMAGE",),
				"masks": ("MASK",),
			},
			"required": {},
		}

	RETURN_TYPES = ("IMAGE","MASK")
	RETURN_NAMES = ("images","masks")
	FUNCTION = "solidify"
	CATEGORY = "IndieTools"
	#OUTPUT_NODE = False


	def solidify(self, images=None, masks=None):

		if images is not None:
			images = normalize_image_tensor(images)
			images = images.round()

		if masks is not None:
			masks = normalize_mask_tensor(masks)
			masks = masks.round()

		return (images, masks)
		
