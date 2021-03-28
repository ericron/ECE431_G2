import torch
from dcmImage import DicomImage
import numpy as np
import torchvision
import sys

def test_torch_Tensors(np_arr):
	tens = torch.from_numpy(np_arr)
	print(type(tens))
	print(tens.size())
	print(tens.dtype)
	print(tens.element_size() * tens.nelement())
	print(sys.getsizeof(tens.storage()))

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
