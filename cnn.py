import matplotlib.pyplot as plt
import torch
from dcmImage import DicomImage
import numpy as np
import torchvision
from pathlib import Path
from PIL import Image
import sys


def test_png(np_arr, filename, save_location):
	pngName = filename + '.png'
	pngPath = save_location / pngName
	img = Image.open(pngPath)
	new_np_array = np.asarray(img)
	print(filename, "-- Equal:", np.array_equal(np_arr, new_np_array))
	display_png(np_arr, new_np_array)


def display_png(original_array, new_array):
	fig, ax = plt.subplots(2, 2)
	ax[0, 0].set_title("Original Image")
	ax[0, 0].imshow(original_array, cmap='gray')
	ax[0, 0].axis('off')
	ax[0, 1].set_title("Saved Image")
	ax[0, 1].imshow(new_array, cmap='gray')
	ax[0, 1].axis('off')
	ax[1, 0].hist(original_array.flatten(), bins=250, range=(original_array.min(), original_array.max()))
	ax[1, 0].set_yscale('log')
	ax[1, 1].hist(new_array.flatten(), bins=250, range=(new_array.min(), new_array.max()))
	ax[1, 1].set_yscale('log')
	plt.tight_layout()
	plt.show()


def test_torch_tensors(np_arr):
	tens = torch.from_numpy(np_arr)
	print("Type:", type(tens), "Size:", tens.size(), "Data Size:", tens.dtype)
	print("Estimated Size:", sys.getsizeof(tens.storage()))
	print("Estimated Size as NP array:", sys.getsizeof(np_arr))
	img = Image.fromarray(np_arr)
	print("Estimated SIze as Image:", sys.getsizeof(img))


def save_tensors(np_arr, filename, save_location):
	tens = torch.from_numpy(np_arr)
	tensorName = filename + '.pt'
	tensorPath = save_location / tensorName
	torch.save(tens, tensorPath)
	# File size 1 channel = 451 KB
	# File size 3 channel = 676 KB


def save_numpy_as_png(np_arr, filename, save_location):
	if len(np_arr.shape) == 2:
		arr_min = np.amin(np_arr)
		np_arr -= arr_min
	pngName = filename + '.png'
	pngPath = save_location / pngName
	img = Image.fromarray(np_arr)
	img.save(pngPath)
	# File size 1 channel range <= 68 KB - 112 KB
	# File size 3 channel <= 74 KB - 129 KB


def torch_device():
	try:
		torch.cuda.init()
	except Exception as e:
		print("failed", e)

	if torch.cuda.is_available():
		print('GPU')
	else:
		print('CPU')


if __name__ == '__main__':
	folder = "exampleImages_S00"
	di = DicomImage(folder)
	f1 = "ID_000000e27.dcm"
	im1 = di.load_Dicom(f1)
	# di.show_Dicom(im1)
	# test_torch_Tensors(di.dicom_to_np_array(im1))
	torch_device()
