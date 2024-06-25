import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .utils import normalize_image_tensor, normalize_mask_tensor
import folder_paths
from ultralytics import YOLO
import os
from PIL import Image
import torch.nn.functional as F
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from torchvision import transforms

imgToTensor = transforms.ToTensor()

if not "yolo8" in folder_paths.folder_names_and_paths:
	folder_paths.folder_names_and_paths["yolo8"] = ([os.path.join(folder_paths.models_dir, "yolo8")], folder_paths.supported_pt_extensions)

class YoloDetector:

	@classmethod
	def INPUT_TYPES(self):
		return {
			"required": {
				"images": ("IMAGE", ),
				"model_name": (folder_paths.get_filename_list("yolo8"), ),
				"confidence": ("INT", {
					"default": 70,
					"min": 0,
					"max": 100,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"combine_mode": (["First", "Add", "Intersect", "Single"],),
				"model_mode": (["YOLO8", "FastSAM"],),
				"max_detections": ("INT", {
					"default": 100,
					"min": 0,
					"max": 1000,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"intersection_union": ("INT", {
					"default": 7,
					"min": 0,
					"max": 100,
					"step": 1,
					"round": 1,
					"display": "number"}),
				"detection_scale": ("INT", {
					"default": 100,
					"min": 0,
					"max": 100,
					"step": 1,
					"round": 1,
					"display": "number"}),				
			},
		}

	RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
	RETURN_NAMES = ("previews","images", "boxes", "segments")
	FUNCTION = "run_detection"
	CATEGORY = "IndieTools"
	#OUTPUT_NODE = False

	def run_detection(self, images, model_name, confidence, combine_mode, model_mode, max_detections, intersection_union, detection_scale):
		initial_alloc = torch.cuda.memory_allocated()
		print(f"Initially allocated: {initial_alloc / (1024 ** 2)}")
		result = self.detect(images, model_name, confidence, combine_mode, model_mode, max_detections, intersection_union, detection_scale)
		torch.cuda.empty_cache()
		final_alloc = torch.cuda.memory_allocated()
		print(f"Final allocation: {final_alloc / (1024 ** 2)}. Diff: {(final_alloc - initial_alloc) / (1024 ** 2)}")
		return result


	def detect(self, images, model_name, confidence, combine_mode, model_mode, max_detections, intersection_union, detection_scale):
		images = normalize_image_tensor(images)

		stride_adjusted = (torch.tensor(images.shape[1:3]).cpu() * detection_scale / 100 / 32).to(torch.int16) * 32

		if model_mode == "FastSAM":
			model = FastSAM(folder_paths.get_full_path("yolo8", model_name))
		else:
			model = YOLO(folder_paths.get_full_path("yolo8", model_name))
		
		results = model([arr for arr in (images[:, :, :, [2, 1, 0]] * 255).to(torch.uint8).cpu().numpy()], conf=confidence / 100, imgsz=stride_adjusted.numpy().tolist(), iou=intersection_union / 100, retina_masks=True, max_det=max_detections)

		del model

		previews = []
		result_images = []
		boxes = []
		segments = []

		for i, result in enumerate(results):
			previews.append(torch.tensor(result.plot()).cpu())
			image = images[i]

			result_box = None
			for box in result.boxes if result.boxes is not None else []:

				box_mask = torch.zeros(result.orig_shape).cuda()
				xMin, yMin, xMax, yMax = box.xyxy[0].to(torch.int32).cpu().numpy()
				box_mask[yMin:yMax, xMin:xMax] = 1

				if result_box is None:
					result_box = box_mask

				if combine_mode == "First":
					break
				elif combine_mode == "Single":
					boxes.append(box_mask)
					result_images.append(image)
					result_box = False
				elif combine_mode == "Add":
					result_box = (result_box.to(torch.bool) | box_mask.to(torch.bool)).to(torch.uint8)
				elif combine_mode == "Intersect":
					result_box = (result_box.to(torch.bool) & box_mask.to(torch.bool)).to(torch.uint8)

			if result_box is None:
				boxes.append(torch.zeros(result.orig_shape).cuda())
				result_images.append(image)
			elif result_box is not False:
				boxes.append(result_box)
				result_images.append(image)

			result_mask = None
			for mask in result.masks.data if result.masks is not None else []:

				mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=result.orig_shape, mode="bilinear", align_corners=False).cuda().squeeze(0).squeeze(0)

				if result_mask is None:
					result_mask = mask
				
				if combine_mode == "First":
					break
				elif combine_mode == "Single":
					segments.append(mask)
					result_mask = False
				if combine_mode == "Add":
					result_mask = (result_mask.to(torch.bool) | mask.to(torch.bool)).to(torch.uint8)
				elif combine_mode == "Intersect":
					result_mask = (result_mask.to(torch.bool) & mask.to(torch.bool)).to(torch.uint8)

			if result_mask is None:
				segments.append(torch.zeros(result.orig_shape).cuda())
			elif result_mask is not False:
				segments.append(result_mask)

		result = ((torch.from_numpy(np.array(previews)[:, :, :, [2, 1, 0]]).to(torch.float32) / 255).cpu(), torch.stack(result_images), torch.stack(boxes).cpu(), torch.stack(segments).cpu())
		return result

