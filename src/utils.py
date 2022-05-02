import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, spatial
import torch
from monai.inferers import sliding_window_inference
from models import UNetPooling, UNetStriding


def compute_euclidean_distance_map(segmentation, centerline):
	seg = np.array(segmentation.get_fdata(), dtype=np.bool_)
	cen = np.array(centerline.get_fdata(), dtype=np.bool_)
	map = ndimage.distance_transform_edt(np.logical_not(cen))

	map_max = np.max(map[seg > 0])
	map /= map_max
	map[seg == 0] = 1
	map = np.log2(map, out=np.zeros_like(map), where=(map!=0))
	map_min = np.min(map)
	map[cen>0] = map_min
	map -= np.min(map)
	map /= np.max(map)

	return map


def make_adjacency_matrix(segmentation):
	seg = np.copy(segmentation)
	#seg[seg==0] = np.max(seg)
	ids = np.where(seg==1) # get all pixels in the segmentation mask
	N = len(ids[0])
	tree = spatial.KDTree(list(zip(ids[0],ids[1],ids[2])))
	A = np.zeros(N,N)
	for i in range(N):
		_, ids = tree.query([ids[0][i], id[1][i], ids[2][i]], k=27)
		for j in ids:
			A[i,j] = 1

	return A


def inference(input, model, USE_AMP=False):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128,128,128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    if USE_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


def map(image, weights_file, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_dict = torch.load("../trained_models/"+weights_file)
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items() if k.startswith(prefix)}
    if torch.cuda.is_available():
        model.load_state_dict(adapted_dict)
    else:
        model.load_state_dict(adapted_dict, map_location=torch.device("cpu"))
    model.to(device)
    model.eval()
    with torch.no_grad():
        input = image["image"].unsqueeze(0).to(device)
        #input = image.unsqueeze(0).to(device)
        map = inference(input, model)
    
    #return inference_transform(map[0,0,:,:,:])
    return map[0,0,:,:,:]



def plot_results(dataset_pair, weights_file, model, z_slice=50):

	output = map(dataset_pair, weights_file, model).detach().cpu()
	target = dataset_pair["label"][0,:,:,:].detach().cpu()

	plt.figure(figsize=(12,12))
	plt.imshow(target[:,:,z_slice], vmin=0, vmax=1)
	plt.colorbar()

	plt.figure(figsize=(12,12))
	plt.imshow(output[:,:,z_slice], vmin=0, vmax=1)
	plt.colorbar()

	plt.figure(figsize=(12,12))
	plt.imshow(np.abs(output[:,:,z_slice] - target[:,:,z_slice]), vmin=0, vmax=1)
	plt.colorbar()