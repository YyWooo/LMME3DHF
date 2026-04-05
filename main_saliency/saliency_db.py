import logging
import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import copy
import numpy as np
from numpy import median
import cv2
import scipy.io as sio
import torchaudio
from torchvision import transforms
from  torchvision import utils as vutils

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds



def salmap_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	img = cv2.imread(path, 0) # Load as grayscale # [480,960]
	img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA) 
	if np.max(img) != 0:
		img = (img-np.min(img))/(np.max(img)-np.min(img))
	img = torch.FloatTensor(img)
	return img



def video_loader(video_dir_path, frame_indices):
	video = []
	for i in frame_indices:
		image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
		if os.path.exists(image_path):
			data_transform = transforms.Compose(
            	[
					transforms.Resize(
						[224,224], interpolation=transforms.InterpolationMode.BICUBIC
					),
					# transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize(
						mean=(0.48145466, 0.4578275, 0.40821073),
						std=(0.26862954, 0.26130258, 0.27577711),
					),
				]
			)
			with open(image_path, "rb") as fopen:
				image = Image.open(fopen).convert("RGB")
			image = data_transform(image)
			video.append(image)
		else:
			return video
	return torch.stack(video, dim=0)



def read_sal_text(txt_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(txt_file,'r') as f:
		for line in f:
			word=line.split()
			test_list['names'].append(word[0])
			test_list['nframes'].append(word[1])
			test_list['fps'].append(word[2])
	return test_list


def make_dataset(root_path, annotation_path, salmap_path, audio_path,
				 step, step_duration, sample_rate=16000):
	data = read_sal_text(annotation_path)
	video_names = data['names']
	video_nframes = data['nframes']
	video_fps = data['fps']
	dataset = []
	audiodata= []
	for i in range(len(video_names)):
		if i % 100 == 0:
			print('dataset loading [{}/{}]'.format(i, len(video_names)))

		video_path = os.path.join(root_path, video_names[i])
		annot_path = os.path.join(salmap_path, video_names[i], 'maps')
		annot_path_bin = os.path.join(salmap_path, video_names[i])
		if not os.path.exists(video_path):
			continue
		if not os.path.exists(annot_path):
			continue
		if not os.path.exists(annot_path_bin):
			continue

		n_frames = int(video_nframes[i])
		if n_frames <= 1:
			continue

		begin_t = 1
		end_t = n_frames

		audio_wav_path = os.path.join(audio_path,video_names[i],video_names[i]+'.wav')
		if not os.path.exists(audio_wav_path):
			continue
		audiowav, sr = torchaudio.load(audio_wav_path)
		if sample_rate != sr:
			audiowav = torchaudio.functional.resample(
                audiowav, orig_freq=sr, new_freq=sample_rate
            ) # 音频重采样成 16000Hz
		# n_samples = sample_rate/float(video_fps[i])
		# starts=np.zeros(n_frames+1, dtype=int)
		# ends=np.zeros(n_frames+1, dtype=int)
		# starts[0]=0
		# ends[0]=0
		# for videoframe in range(1,n_frames+1):
		# 	startemp=max(0,((videoframe-1)*(1.0/float(video_fps[i]))*sample_rate)-n_samples/2)
		# 	starts[videoframe] = int(startemp)
		# 	endtemp=min(audiowav.shape[1],abs(((videoframe-1)*(1.0/float(video_fps[i]))*sample_rate)+n_samples/2))
		# 	ends[videoframe] = int(endtemp)

		audioinfo = {
			'audiopath': audio_path,
			'video_id': video_names[i],
			'Fs' : sample_rate,
			'wav' : audiowav,
			# 'starts': starts,
			# 'ends' : ends
		}
		audiodata.append(audioinfo)

		sample = {
			'video': video_path,
			'segment': [begin_t, end_t],
			'n_frames': n_frames,
			'fps': video_fps[i],
			'video_id': video_names[i],
			'salmap': annot_path,
			'binmap': annot_path_bin
		}
		step=int(step)
		for j in range(1, n_frames, step):
			sample_j = copy.deepcopy(sample)
			sample_j['frame_indices'] = list(range(j, min(n_frames + 1, j + step)))
			dataset.append(sample_j)

	return dataset, audiodata


class saliency_db(data.Dataset):

	def __init__(self,
				 root_path,
				 annotation_path,
				 subset,
				 audio_path,
				 temporal_transform = None,
				 target_transform = None,
				 exhaustive_sampling = False,
				 sample_duration = 16,
				 step_duration = 90,
				 sample_rate = 16000):
		
		self.sample_rate = sample_rate

		if exhaustive_sampling:
			self.exhaustive_sampling = True
			step = 5
			step_duration = sample_duration
		else:
			self.exhaustive_sampling = False
			step = 50

		self.data, self.audiodata = make_dataset(
			root_path, annotation_path, subset, audio_path,
			step, step_duration, sample_rate)

		self.temporal_transform = temporal_transform
		self.target_transform = target_transform
		# self.loader = get_loader()
		self.audio_win = int(2*sample_rate)

	def __getitem__(self, index):

		path = self.data[index]['video']
		annot_path = self.data[index]['salmap']
		annot_path_bin = self.data[index]['binmap']

		n_frames = self.data[index]['n_frames']
		frame_indices = self.data[index]['frame_indices']
		if self.temporal_transform is not None:
			frame_indices = self.temporal_transform(frame_indices)

		video_name=self.data[index]['video_id']
		video_fps = self.data[index]['fps']
		flagexists=0
		for iaudio in range(0, len(self.audiodata)):
			if (video_name == self.audiodata[iaudio]['video_id']):
				audioind = iaudio
				flagexists = 1
				break

		audioexcer  = torch.zeros(self.audiodata[audioind]['wav'].shape[0],self.audio_win)  ## maximum audio excerpt duration
		curr_indice = max(frame_indices)
		data = {'vision':[], 'audio':[]}
		valid = {}
		valid['audio']=0
		if flagexists:
			excerptend = (curr_indice-0.5) * (self.sample_rate / float(video_fps))
			excerptend = int(excerptend)
			audioexcer_tmp = self.audiodata[audioind]['wav'][:, max(excerptend - 2*self.sample_rate, 0) : min(excerptend, self.audiodata[audioind]['wav'].shape[1])]
			if excerptend - 2*self.sample_rate < 0:
				zeros = torch.zeros([audioexcer_tmp.shape[0], abs(excerptend - 2*self.sample_rate)], dtype=audioexcer_tmp.dtype, device=audioexcer_tmp.device)
				audioexcer_tmp = torch.cat((zeros, audioexcer_tmp), dim=1)
			if excerptend > self.audiodata[audioind]['wav'].shape[1]:
				zeros = torch.zeros([audioexcer_tmp.shape[0], abs(excerptend - self.audiodata[audioind]['wav'].shape[1]) ], dtype=audioexcer_tmp.dtype, device=audioexcer_tmp.device)
				audioexcer_tmp = torch.cat((audioexcer_tmp, zeros), dim=1)
			audioexcer = audioexcer_tmp

		waveform_melspec = waveform2melspec(audioexcer, self.sample_rate, num_mel_bins=128, target_length=204)
		normalize = transforms.Normalize(mean=-4.268, std=9.138)
		waveform_melspec = normalize(waveform_melspec)
		data['audio'] = waveform_melspec.unsqueeze(0)

		
		target = {'salmap':[]}
		# target = {'salmap':[],'binmap':[]}
		target['salmap'] = salmap_loader(os.path.join(annot_path, 'eyeMap_{:05d}.jpg'.format(curr_indice)))
		# tmp_mat = sio.loadmat(os.path.join(annot_path_bin, 'fixMap_{:05d}.mat'.format(med_indices)))
		# binmap_np = np.array(Image.fromarray(tmp_mat['eyeMap'].astype(float)).resize((320, 240), resample = Image.BILINEAR)) > 0
		# target['binmap'] = Image.fromarray((255*binmap_np).astype('uint8'))
		if self.exhaustive_sampling:
			target['video'] = self.data[index]['video_id']
		clip = video_loader(path, frame_indices)

		# if self.spatial_transform is not None:
			# target['binmap'] = self.spatial_transform_sal(target['binmap'])
			# target['binmap'] = torch.gt(target['binmap'], 0.0).float()

		valid['sal'] = 1
		data['vision'] = clip.squeeze(0) # [3,16,112,112]

		return data, target, video_fps #, valid # [16,3,224,224]和[1,128,204]

	def __len__(self):
		return len(self.data)

