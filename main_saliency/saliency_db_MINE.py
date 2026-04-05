import torch.utils.data as data
import os
import random
from tqdm import tqdm
DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

from videollama2.mm_utils import salmap_loader

def make_dataset(image_path, salmap_path, phase, ratio):
	image_names = os.listdir(image_path)
	dataset = []

	for i, img in tqdm(enumerate(image_names)):
		i_path = os.path.join(image_path, img)
		sal_path = os.path.join(salmap_path, img)

		if not os.path.exists(i_path):
			continue
		if not os.path.exists(sal_path) and not phase == 'test':
			continue

		sample = {
			'path': i_path,
			'name': img,
			'salmap': sal_path,
		}
		dataset.append(sample)

	random.shuffle(dataset)
	ratio = float(ratio)
	if phase == 'train':
		return dataset[int(ratio * len(dataset)) :]
	else:
		return dataset[: int(ratio * len(dataset))]


class saliency_db_MINE(data.Dataset):
	def __init__(self,
		image_path,
		salmap_path,
		precessor,
		phase = 'train',
     	version = 'ver0',
		instruct = 'What is the salient region in this image?',
		backbone = None,
		ratio = 0.2,
	):
		self.phase = phase
		self.data = make_dataset(image_path, salmap_path, phase, ratio)
		self.precessor = precessor
		self.modal = 'image'
		self.instruct = instruct
		self.version = version
		self.backbone = backbone

	def __getitem__(self, index):
		img_path = self.data[index]['path']
		sal_path = self.data[index]['salmap']

		target = {}
		if not self.phase == "test":
			target = {'salmap':[]}
			target['salmap'] = salmap_loader(sal_path, self.version, self.backbone)	# torch.Size([416, 416])
 
		if self.phase == "test" or self.phase == "val":
			target['name'] = self.data[index]['name']
   
		data = {'vision':[], 'audio':[], 'modal':[], 'instruct':[]}
		clip = self.precessor[self.modal](img_path) # torch.Size([1, 3, 384, 384])
		data['vision'] = clip
		data['modal'] = self.modal
		data['instruct'] = self.instruct

		return data, target

	def __len__(self):
		return len(self.data)
