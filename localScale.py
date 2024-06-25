import torch
import torch.nn.functional as F
from .utils import normalize_image_tensor, normalize_mask_tensor

class LocalScale:

	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"base_images": ("IMAGE",),
				"masks": ("MASK",),
				"scale_x": ("FLOAT", {
					"default": 1.0,
					"min": 0.0,
					"max": 10.0,
					"step": 0.01,
					"round": 0.01,
					"display": "number"}),
				"scale_y": ("FLOAT", {
					"default": 1.0,
					"min": 0.0,
					"max": 10.0,
					"step": 0.01,
					"round": 0.01,
					"display": "number"}),
				"offset_x": ("INT", {
					"default": 0,
					"min": -1000,
					"max": 1000,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"offset_y": ("INT", {
					"default": 0,
					"min": -1000,
					"max": 1000,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"sample_mode": (["Bilinear", "Nearest", "Bicubic", "Area"],),
				"mask_edit": (["Region of Interest", "None"],),
				"align_to": (["Mask", "Base Image"],),
				"alignment_anchor": (["Top Left", "Top Center", "Top Right", "Center Left", "Center Center", "Center Right", "Bottom Left", "Bottom Center", "Bottom Right"],),
				"scaled_anchor": (["Top Left", "Top Center", "Top Right", "Center Left", "Center Center", "Center Right", "Bottom Left", "Bottom Center", "Bottom Right"],),
			},
			"optional": {
				"paste_masks": ("MASK",),
			},
		}

	RETURN_TYPES = ("IMAGE","MASK")
	RETURN_NAMES = ("images","masks")
	FUNCTION = "scale_locally"
	CATEGORY = "IndieTools"
	#OUTPUT_NODE = False

	def scale_locally(self, base_images, masks, scale_x, scale_y, offset_x, offset_y, sample_mode, mask_edit, align_to, alignment_anchor, scaled_anchor, paste_masks=None):

		base_images = normalize_image_tensor(base_images)
		masks = normalize_mask_tensor(masks)
		paste_masks = normalize_mask_tensor(paste_masks) if paste_masks is not None else None

		n_images = base_images.size(0)
		n_masks = masks.size(0)
		n_paste_masks = paste_masks.size(0) if paste_masks is not None else 0

		if not n_images == n_masks and not n_masks == 1:
			raise ValueError(f"The number of passed in masks ({n_masks}) must either be 1 or match the amount of base images ({n_images})")

		if not paste_masks is None and not n_paste_masks == n_images and not n_paste_masks == 1:
			raise ValueError(f"The number of passed in paste masks ({n_paste_masks}) must either be 1 or match the amount of base images ({n_images})")

		res = []
		res_masks = []

		for i, cur_img in enumerate(base_images):
			mask = masks[i] if n_masks == n_images else masks[0]
			if n_paste_masks > 0:
				paste_mask = paste_masks[i] if n_paste_masks == n_images else paste_masks[0]

			result_mask = mask.clone()
			image_height, image_width, _ = cur_img.size()
			
			if mask_edit == "Region of Interest":
				mask_nonzero_indices = torch.nonzero(result_mask)
				if mask_nonzero_indices.size()[0] == 0:
					res.append(cur_img.clone())
					res_masks.append(torch.zeros_like(result_mask))
					continue

				y_min, x_min = torch.min(mask_nonzero_indices, dim=0).values.cpu().numpy()
				y_max, x_max = torch.max(mask_nonzero_indices, dim=0).values.cpu().numpy()
				mask = mask[y_min:y_max, x_min:x_max]
			else:
				y_min, x_min = 0
				y_max, x_max = mask.size()

			mask_height, mask_width = mask.size()
			edit_layer = cur_img[y_min:y_max, x_min:x_max, :] * mask.unsqueeze(-1)

			edit_layer = F.interpolate(edit_layer.unsqueeze(0).permute(0, 3, 1, 2), size=(int(mask_height * scale_y), int(mask_width * scale_x)), mode=sample_mode.lower()).permute(0, 2, 3, 1).squeeze(0)
			scaled_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(int(mask_height * scale_y), int(mask_width * scale_x)), mode=sample_mode.lower()).squeeze(0).squeeze(0)
			scaled_height, scaled_width, _ = edit_layer.size()

			y_anchor, x_anchor = alignment_anchor.lower().split(" ")
			if y_anchor == "center": y_anchor = "v_center"
			if x_anchor == "center": x_anchor = "h_center"
			y_paste_anchor, x_paste_anchor = scaled_anchor.lower().split(" ")
			if y_paste_anchor == "center": y_paste_anchor = "v_center"
			if x_paste_anchor == "center": x_paste_anchor = "h_center"

			# Catch useless combinations that would render the images out of bounds
			if y_anchor == "top" and y_paste_anchor == "bottom" or y_anchor == "bottom" and y_paste_anchor == "top":
				raise ValueError(f"The combination of alignment_anchor '{y_anchor}' and scaled_anchor {y_paste_anchor} is out of bounds")
			# Catch useless combinations that would render the images out of bounds
			if x_anchor == "left" and x_paste_anchor == "right" or x_anchor == "right" and x_paste_anchor == "left":
				raise ValueError(f"The combination of alignment_anchor '{y_anchor}' and scaled_anchor {y_paste_anchor} is out of bounds")

			base_anchor = {
				"Mask": {
					"top": y_min,
					"v_center": y_min + mask_height // 2,
					"bottom": y_min + mask_height,
					"left": x_min,
					"h_center": x_min + mask_width // 2,
					"right": x_min + mask_width
				},
				"Base Image": {
					"top": 0,
					"v_center": image_height // 2,
					"bottom": image_height,
					"left": 0,
					"h_center": image_width // 2,
					"right": image_width
				}
			}

			base_anchor_y, base_anchor_x = base_anchor[align_to][y_anchor], base_anchor[align_to][x_anchor]
			base_anchor_x += offset_x
			base_anchor_y += offset_y

			paste_anchor = {
				"top": (base_anchor_y, base_anchor_y + scaled_height),
				"v_center": (base_anchor_y - scaled_height // 2 - (0 if scaled_height % 2 == 0 else 1), base_anchor_y + scaled_height // 2),
				"bottom": (base_anchor_y - scaled_height, base_anchor_y),
				"left": (base_anchor_x, base_anchor_x + scaled_width),
				"h_center": (base_anchor_x - scaled_width // 2 - (0 if scaled_width % 2 == 0 else 1), base_anchor_x + scaled_width // 2),
				"right": (base_anchor_x - scaled_width, base_anchor_x)
			}

			y_min_paste, y_max_paste = paste_anchor[y_paste_anchor]
			x_min_paste, x_max_paste = paste_anchor[x_paste_anchor]

			y_min_overflow = abs(min(y_min_paste, 0))
			y_max_overflow = max(y_max_paste - image_height, 0)
			x_min_overflow = abs(min(x_min_paste, 0))
			x_max_overflow = max(x_max_paste - image_width, 0)

			y_min_paste = max(y_min_paste, 0)
			y_max_paste = min(y_max_paste, image_height)
			x_min_paste = max(x_min_paste, 0)
			x_max_paste = min(x_max_paste, image_width)

			pasted_image = cur_img.clone()
			edit_layer = edit_layer[y_min_overflow:scaled_height - y_max_overflow, x_min_overflow:scaled_width - x_max_overflow]
			scaled_mask = scaled_mask[y_min_overflow:scaled_height - y_max_overflow, x_min_overflow:scaled_width - x_max_overflow]
			mask_to_apply = (edit_layer.sum(dim=-1) != 0)
			
			if n_paste_masks > 0:
				mask_to_apply = mask_to_apply & (paste_mask[y_min_paste:y_max_paste, x_min_paste:x_max_paste] == 0)

			result_mask[int(y_min_paste):int(y_max_paste), int(x_min_paste):int(x_max_paste)][mask_to_apply] = scaled_mask.squeeze(-1)[mask_to_apply]
			mask_to_apply = mask_to_apply.unsqueeze(-1).expand_as(edit_layer)
			pasted_image[int(y_min_paste):int(y_max_paste), int(x_min_paste):int(x_max_paste), :][mask_to_apply] = edit_layer[mask_to_apply]

			res.append(pasted_image)
			res_masks.append(result_mask)
		
		return (torch.stack(res), torch.stack(res_masks))
