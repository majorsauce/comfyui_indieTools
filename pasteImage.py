import cv2
import numpy as np
from .utils import normalize_image_tensor, normalize_mask_tensor
import torch.nn.functional as F
import torch

class PasteImage:

	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"destination": ("IMAGE",),
				"source": ("IMAGE",),
				"cut_info": ("CUT_INFO",)
			},
			"optional": {
				"masks": ("MASK",)
			}
		}

	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("images",)
	FUNCTION = "paste_images"
	CATEGORY = "IndieTools"
	#OUTPUT_NODE = False


	def paste_images(self, destination, source, cut_info, masks=None):
		destination = normalize_image_tensor(destination)
		source = normalize_image_tensor(source)

		dest_num_images = destination.size(0)
		source_num_images = source.size(0)

		if masks is not None:
			masks = normalize_mask_tensor(masks)

		if not len(cut_info) == dest_num_images:
			raise ValueError(f"The cut info includes {len(cut_info)} entries but you try to paste to {dest_num_images} images.")

		if not dest_num_images == source_num_images:
			raise ValueError(f"Invalid relation betwen number of source ({source_num_images}) and destination ({dest_num_images}) images")

		paste_results = []
		source_image_index = 0

		for i, pi in enumerate(cut_info):
			source_image = source[source_image_index]
			dest_image = destination[i]

			# Skip paste if empty mask was provided on cut
			if pi == None:
				paste_results.append(dest_image)
				continue

			mask = None
			if masks is not None and len(masks) > 0:
				if len(masks) == source_num_images:
					mask = masks[i]
				else:
					mask = masks[0]
				mask = mask.unsqueeze(2).repeat(1, 1, 3) != 0

			if "pad_height" in pi:
				source_width, source_height, _ = source_image.shape
				source_image = source_image[0:source_height - pi["pad_height"], 0:source_width - pi["pad_width"], :]
				if mask is not None:
					mask = mask[0:source_height - pi["pad_height"], 0:source_width - pi["pad_width"]]


			source_image = F.interpolate(source_image.unsqueeze(0).permute(0, 3, 1, 2), size=(pi["y_max"] - pi["y_min"], pi["x_max"] - pi["x_min"]), mode="bilinear").permute(0, 2, 3, 1).squeeze(0)

			if mask is None:
				dest_image[pi["y_min"]:pi["y_max"], pi["x_min"]:pi["x_max"], :] = source_image
			else:
				edit_layer = dest_image[pi["y_min"]:pi["y_max"], pi["x_min"]:pi["x_max"], :]
				edit_layer[mask] = source_image[mask]
				dest_image[pi["y_min"]:pi["y_max"], pi["x_min"]:pi["x_max"], :] = edit_layer
			paste_results.append(dest_image)
			source_image_index += 1

		return (torch.stack(paste_results), )