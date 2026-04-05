import random


class LoopPadding(object): # 最后不够用时循环重复一下

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out



class TemporalRandomCrop(object): # 训练用
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, opt):
        self.size = opt.sample_num
        # self.step = opt.sample_step

    def __call__(self, frame_indices, num_frames, min_frame_number=None, max_frame_number=None): # 16帧的取法：1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0 1 0 1 0 ★ 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0 0 0 1 
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        curr_indice = random.choice(frame_indices)
        out = [curr_indice]
        
        step = 2
        index_f = index_b = curr_indice
        for i in range(1, self.size//2+1):
            index_f = index_f - step
            index_b = index_b + step
            if min_frame_number is not None:
                if index_f >= min_frame_number:
                    out.insert(0, index_f)
                else:
                    out.insert(0, min_frame_number)
            else:  # min_frame_number == None
                if  index_f > 0:
                    out.insert(0, index_f)
                else:
                    out.insert(0, 1)
            if i < 8:
                if max_frame_number is not None:
                    if index_b <= max_frame_number:
                        out.append(index_b)
                    else:
                        out.append(max_frame_number)
                else: # max_frame_number == None
                    if index_b <= num_frames:
                        out.append(index_b)
                    else:
                        out.append(num_frames)
            if i == 4:
                step = 4
           
        return out, curr_indice


class TemporalCenterCrop(object): # val用
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, opt):
        self.size = opt.sample_num
        # self.step = opt.sample_step

    def __call__(self, frame_indices, num_frames, min_frame_number=None, max_frame_number=None): # 16帧的取法：1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0 1 0 1 0 ★ 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0 0 0 1 
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        curr_indice = frame_indices[len(frame_indices)//2]
        out = [curr_indice]
        
        step = 2
        index_f = index_b = curr_indice
        for i in range(1, self.size//2+1):
            index_f = index_f - step
            index_b = index_b + step
            if min_frame_number is not None:
                if index_f >= min_frame_number:
                    out.insert(0, index_f)
                else:
                    out.insert(0, min_frame_number)
            else:  # min_frame_number == None
                if  index_f > 0:
                    out.insert(0, index_f)
                else:
                    out.insert(0, 1)
            if i < 8:
                if max_frame_number is not None:
                    if index_b <= max_frame_number:
                        out.append(index_b)
                    else:
                        out.append(max_frame_number)
                else: # max_frame_number == None
                    if index_b <= num_frames:
                        out.append(index_b)
                    else:
                        out.append(num_frames)
            if i == 4:
                step = 4
           
        return out, curr_indice
    
    
class TemporalStartCrop(object): # 推理用
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, opt):
        self.size = opt.sample_num
        # self.step = opt.sample_step

    def __call__(self, frame_indices, num_frames, min_frame_number=None, max_frame_number=None): # 16帧的取法：1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0 1 0 1 0 ★ 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0 0 0 1 
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        curr_indice = min(frame_indices)
        out = [curr_indice]
        
        step = 2
        index_f = index_b = curr_indice
        for i in range(1, self.size//2+1):
            index_f = index_f - step
            index_b = index_b + step
            if min_frame_number is not None:
                if index_f >= min_frame_number:
                    out.insert(0, index_f)
                else:
                    out.insert(0, min_frame_number)
            else:  # min_frame_number == None
                if index_f > 0:
                    out.insert(0, index_f)
                else:
                    out.insert(0, 1)
            if i < 8:
                if max_frame_number is not None:
                    if index_b <= max_frame_number:
                        out.append(index_b)
                    else:
                        out.append(max_frame_number)
                else: # max_frame_number == None
                    if index_b <= num_frames:
                        out.append(index_b)
                    else:
                        out.append(num_frames)
            if i == 4:
                step = 4
           
        return out, curr_indice


if __name__ == "__main__":
    from opts import parse_opts

    opt = parse_opts()
    tem = TemporalCenterCrop(opt)
    # out, curr_indice = tem([8,9,10,11,12,13,14,15,16,17,18,19],26,2,24)
    out, curr_indice = tem([8,9,10,11,12,13,14,15,16,17,18,19],26)
    print(out)
    print(curr_indice)