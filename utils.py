

def normalize_image_tensor(images):
	if images == None:
		raise ValueError("No images passed")

	images = images.clone().cpu()

	for _ in range(4 - len(images.size())):
		images = images.unsqueeze(0)

	return images


def normalize_mask_tensor(masks):
	masks = masks.clone().cpu()
		
	# In case the mask is in form height, width, 1 channel > Squeeze to width, height shape
	for _ in range(len(masks.size()) - 3):
		if masks.size(-1) == 1:
			masks = masks.squeeze(-1)
		elif masks.size(0) == 1:
			masks = masks.squeeze(0)
		elif masks.size(1) == 1:
			masks = masks.squeeze(1)

	# Will add extra dimensions as we expect format [batchMasks, height, width]
	for _ in range(3 - len(masks.size())):
		masks = masks.unsqueeze(0)

	return masks
