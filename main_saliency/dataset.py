from saliency_db_MINE import saliency_db_MINE

def get_training_set(opt, processor, temporal_transform):
	print('Creating training dataset: {}'.format(opt.dataset))

	training_data = saliency_db_MINE(
		opt.image_path_MINE,
		opt.salmap_path_MINE,
		processor,
		phase = 'train',
		version = opt.ver,
		instruct=opt.instruct,
   		backbone = opt.backbone
	)

	return training_data


def get_validation_set(opt, processor, temporal_transform):
	print('Creating validation dataset: {}'.format(opt.dataset))

	validation_data = saliency_db_MINE(
		opt.image_path_MINE,
		opt.salmap_path_MINE,
		processor,
		phase = 'val',
		version = opt.ver,
		instruct=opt.instruct,
   		backbone = opt.backbone
	)

	return validation_data


def get_test_set(opt, processor, temporal_transform):
	print('Creating testing dataset: {}'.format(opt.dataset))

	test_data = saliency_db_MINE(
		opt.image_path_MINE,
		opt.salmap_path_MINE,
		processor,
		phase = 'test',
		version = opt.ver,
		instruct=opt.instruct,
   		backbone = opt.backbone
	)

	return test_data
