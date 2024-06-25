import cv2
import numpy as np
from .utils import normalize_image_tensor, normalize_mask_tensor
import torch
import torch.nn.functional as F

class CutByMask:

	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE",),
				"masks": ("MASK",),
				"padding": ("INT", {
					"default": 10,
					"min": 0,
					"max": 1000,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"target_width": ("INT", {
					"default": 512,
					"min": 0,
					"max": 4096,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"target_height": ("INT", {
					"default": 512,
					"min": 0,
					"max": 4096,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"kernel_size": ("INT", {
					"default": 4,
					"min": 0,
					"max": 128,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"resize_mode": (["Scale", "Resize", "Resize Keep Ratio"],),
			}
		}

	RETURN_TYPES = ("IMAGE", "MASK", "CUT_INFO")
	RETURN_NAMES = ("images", "masks", "cut info")
	CATEGORY = "IndieTools"
	FUNCTION = "cut_for_fullres"
	#OUTPUT_NODE = False

	def cut_for_fullres(self, images, masks, padding, target_width, target_height, kernel_size, resize_mode):
		if len(images.size()) == 3:
			images = images.unsqueeze(0)
		if len(masks.size()) == 2:
			masks = masks.unsqueeze(0)

		images = normalize_image_tensor(images)
		masks = normalize_mask_tensor(masks)

		num_images = images.size(0)
		num_masks = masks.size(0)
		image_height, image_width = images.shape[1:3]

		mask = masks[0]
		result_images = []
		result_masks = []
		cut_info = []

		for i, image in enumerate(images):
			if num_masks == num_images: mask = masks[i]
			else: mask = masks[0]

			mask_nonzero_indices = torch.nonzero(mask)
			
			if mask_nonzero_indices.size(0) == 0:
				# Skip if mask is empty
				cut_info.append(None)
				continue
			
			y_min, x_min = torch.min(mask_nonzero_indices, dim=0).values.numpy()
			y_max, x_max = torch.max(mask_nonzero_indices, dim=0).values.numpy()
			y_min = max(y_min - padding, 0)
			y_max = min(y_max + padding, image_height)
			x_min = max(x_min - padding, 0)
			x_max = min(x_max + padding, image_width)
			height = y_max - y_min
			width = x_max - x_min

			interpolate_height = target_height
			interpolate_width = target_width

			if resize_mode == "Scale":
				height_ratio = target_height / height
				width_ratio = target_width / width
				
				scale_factor = height_ratio
				if height_ratio <= width_ratio: 
					scale_factor = height_ratio
				else:
					scale_factor = width_ratio

				interpolate_height = height * scale_factor
				interpolate_width = width * scale_factor

			elif resize_mode == "Resize":
				pass
			elif resize_mode == "Resize Keep Ratio":
				height = y_max - y_min
				width = x_max - x_min
				target_ratio = target_width / target_height
				current_ratio = width / height
				adjustment_ratio = target_ratio / current_ratio
				if adjustment_ratio > 0:
					adjusted_width = adjustment_ratio * width
					half = int((adjusted_width - width) / 2)
					x_min -= half
					if x_min < 0:
						x_max += x_min * -1
						x_min = 0
					x_max += half
					if x_max > image_width:
						x_overflow = x_max - image_width
						if x_min < x_overflow:
							raise ValueError(f"Cannot keep the dimension for the passed mask. Need to cut width of {x_max} but image is only {width} wide")	
						x_min -= x_overflow
						x_max -= x_overflow
						
				elif adjustment_ratio < 0:
					adjusted_height = 1 / adjustment_ratio * height
					half = int((adjusted_height - height) / 2)
					y_min -= half
					if y_min < 0:
						y_max += y_min * -1
						y_min = 0
					if y_max > image_height:
						y_overflow = y_max - image_height
						if y_min < y_overflow:
							raise ValueError(f"Cannot keep the dimension for the passed mask. Need to cut height of {y_max} but image is only {height} wide")	
						y_min -= y_overflow
						y_max -= y_overflow

			image = image[y_min:y_max, x_min:x_max]
			mask = mask[y_min:y_max, x_min:x_max]
			image = F.interpolate(image.unsqueeze(0).permute(0, 3, 1, 2), size=(int(interpolate_height), int(interpolate_width)), mode="bilinear").permute(0, 2, 3, 1).squeeze(0)
			mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(int(interpolate_height), int(interpolate_width)), mode="bilinear").squeeze(0).squeeze(0)
			cut_info.append({"y_min": y_min, "y_max": y_max, "x_min": x_min, "x_max": x_max})
			
			# Pad images if required
			pad_height = target_height - image.size(0)
			pad_width = target_width - image.size(1)
			if pad_width != 0 or pad_height != 0:
				image = F.pad(image, (0, 0, 0, pad_width, 0, pad_height))
				mask = F.pad(mask, (0, pad_width, 0, pad_height))
				cut_info[-1]["pad_width"] = pad_width
				cut_info[-1]["pad_height"] = pad_height
			
			result_images.append(image)
			result_masks.append(mask)
		
		if len(result_images) == 0:
			return (torch.zeros(1, target_height, target_width, 3), torch.zeros(1, target_height, target_width), cut_info)
		return (torch.stack(result_images), torch.stack(result_masks), cut_info)
