import glob
from typing import List
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image as imwrite
import os
from os.path import join, exists
import utility
import numpy as np
import ast
import time
from ldm.models.autoencoder import * 
from torchvision.transforms import ToPILImage


class TripletTestSet:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #outptu tensor in [-1,1]

    def eval(self, model, sample_func, metrics=['PSNR', 'SSIM'], output_dir=None, output_name='output.png', resume=False):
        results_dict = {k : [] for k in metrics}

        start_idx = 0
        if resume:
            # fill in results_dict with prev results and find where to start from
            assert os.path.exists(join(output_dir, 'results.txt')), 'no res file found to resume from!'
            with open(join(output_dir, 'results.txt'), 'r') as f:
                prev_lines = f.readlines()
                for line in prev_lines:
                    if len(line) < 2:
                        continue
                    cur_res = ast.literal_eval(line.strip().split('-- ')[1].split('time')[0]) #parse dict from string
                    for k in metrics:
                        results_dict[k].append(float(cur_res[k]))
                    start_idx += 1
        
        logfile = open(join(output_dir, 'results.txt'), 'a')
        for idx in range(len(self.im_list)):
            if resume and idx < start_idx:
                assert os.path.exists(join(output_dir, self.im_list[idx], output_name)), f'skipping idx {idx} but output not found!'
                continue

            print(f'Evaluating {self.im_list[idx]}')
            t0 = time.time()
            if not exists(join(output_dir, self.im_list[idx])):
                os.makedirs(join(output_dir, self.im_list[idx]))

            with torch.no_grad():
                with model.ema_scope():
                    # form condition tensor and define shape of latent rep
                    xc = {'prev_frame': self.input0_list[idx], 'next_frame': self.input1_list[idx]}
                    c, phi_prev_list, phi_next_list = model.get_learned_conditioning(xc)
                    shape = (model.channels, c.shape[2], c.shape[3])
                    # run sampling and get denoised latent rep
                    out = sample_func(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None)
                    if isinstance(out, tuple): # using ddim
                        out = out[0]
                    # reconstruct interpolated frame from latent
                    out = model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
                    out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]

            gt = self.gt_list[idx]

            for metric in metrics:
                score = getattr(utility, 'calc_{}'.format(metric.lower()))(gt, out, [self.input0_list[idx], self.input1_list[idx]])[0].item()
                results_dict[metric].append(score)

            imwrite(out, join(output_dir, self.im_list[idx], output_name), value_range=(-1, 1), normalize=True)

            msg = '{:<15s} -- {}'.format(self.im_list[idx], {k: round(results_dict[k][-1],3) for k in metrics}) + f'    time taken: {round(time.time()-t0,2)}' + '\n'
            print(msg, end='')
            logfile.write(msg)

        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]),3) for k in metrics}) + '\n\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()

class c001_a(TripletTestSet):
    def __init__(self, db_dir):
        super(c001_a, self).__init__()
        
        slice_idx_start = 5200  # 첫 번째 축에서 선택할 슬라이스 시작 인덱스
        slice_idx_end = 5200  # 첫 번째 축에서 선택할 슬라이스 끝 인덱스
        step_size = 20  # 슬라이스 간격

        # 확대할 영역의 크기 정의
        window_size = 224

        # 확대할 영역의 중심 좌표 선택
        x_index_start = 200
        x_index_end = 800
        x_step_size = 50

        y_index_start = 200
        y_index_end = 500
        y_step_size = 50
        
        self.im_list = []
        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        
        pseudo_colored_tensor = torch.load(os.path.join(db_dir, 'false_color.pt'))

        for slice_idx in range(slice_idx_start, slice_idx_end + 1):
            for x_index in range(x_index_start, x_index_end + 1, x_step_size):
                for y_index in range(y_index_start, y_index_end + 1, y_step_size):
                    self.im_list.append(f"{slice_idx}_{x_index}_{y_index}_{step_size}")
                    self.input0_list.append(self.transform(ToPILImage()(pseudo_colored_tensor[slice_idx, y_index-window_size//2:y_index+window_size//2, x_index-window_size//2:x_index+window_size//2, :].permute(2, 0, 1))).cuda().unsqueeze(0))
                    self.gt_list.append(self.transform(ToPILImage()(pseudo_colored_tensor[slice_idx+step_size, y_index-window_size//2:y_index+window_size//2, x_index-window_size//2:x_index+window_size//2, :].permute(2, 0, 1))).cuda().unsqueeze(0))
                    self.input1_list.append(self.transform(ToPILImage()(pseudo_colored_tensor[slice_idx+2*step_size, y_index-window_size//2:y_index+window_size//2, x_index-window_size//2:x_index+window_size//2, :].permute(2, 0, 1))).cuda().unsqueeze(0))
 
class c001_c(TripletTestSet):
    def __init__(self, db_dir):
        super(c001_c, self).__init__()
        
        slice_idx_start = 5200  # 첫 번째 축에서 선택할 슬라이스 시작 인덱스
        slice_idx_end = 5200  # 첫 번째 축에서 선택할 슬라이스 끝 인덱스
        step_size = 20  # 슬라이스 간격

        # 확대할 영역의 크기 정의
        window_size = 224

        # 확대할 영역의 중심 좌표 선택
        x_index_start = 200
        x_index_end = 800
        x_step_size = 50

        y_index_start = 200
        y_index_end = 500
        y_step_size = 50
        
        self.im_list = []
        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        
        pseudo_colored_tensor = torch.load(os.path.join(db_dir, 'false_color.pt'))

        for slice_idx in range(slice_idx_start, slice_idx_end + 1):
            for x_index in range(x_index_start, x_index_end + 1, x_step_size):
                for y_index in range(y_index_start, y_index_end + 1, y_step_size):
                    self.im_list.append(f"{slice_idx}_{x_index}_{y_index}_{step_size}")
                    self.input0_list.append(self.transform(ToPILImage()(pseudo_colored_tensor[slice_idx, y_index-window_size//2:y_index+window_size//2, x_index-window_size//2:x_index+window_size//2, :].permute(2, 0, 1))).cuda().unsqueeze(0))
                    self.gt_list.append(self.transform(ToPILImage()(pseudo_colored_tensor[slice_idx+step_size, y_index-window_size//2:y_index+window_size//2, x_index-window_size//2:x_index+window_size//2, :].permute(2, 0, 1))).cuda().unsqueeze(0))
                    self.input1_list.append(self.transform(ToPILImage()(pseudo_colored_tensor[slice_idx+2*step_size, y_index-window_size//2:y_index+window_size//2, x_index-window_size//2:x_index+window_size//2, :].permute(2, 0, 1))).cuda().unsqueeze(0))

class Middlebury_others(TripletTestSet):
    def __init__(self, db_dir):
        super(Middlebury_others, self).__init__()
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(self.transform(Image.open(join(db_dir, 'input', item , 'frame10.png'))).cuda().unsqueeze(0)) # [1,3,H,W] in [-1,1]
            self.input1_list.append(self.transform(Image.open(join(db_dir, 'input', item , 'frame11.png'))).cuda().unsqueeze(0)) # [1,3,H,W] in [-1,1]
            self.gt_list.append(self.transform(Image.open(join(db_dir, 'gt', item , 'frame10i11.png'))).cuda().unsqueeze(0))

class Davis(TripletTestSet):
    def __init__(self, db_dir):
        super(Davis, self).__init__()
        self.im_list = ['bike-trial', 'boxing', 'burnout', 'choreography', 'demolition', 'dive-in', 'dolphins', 'e-bike', 'grass-chopper', 'hurdles', 'inflatable', 'juggle', 'kart-turn', 'kids-turning', 'lions', 'mbike-santa', 'monkeys', 'ocean-birds', 'pole-vault', 'running', 'selfie', 'skydive', 'speed-skating', 'swing-boy', 'tackle', 'turtle', 'varanus-tree', 'vietnam', 'wings-turn']

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(self.transform(Image.open(join(db_dir, 'input', item , 'frame10.png'))).cuda().unsqueeze(0))
            self.input1_list.append(self.transform(Image.open(join(db_dir, 'input', item , 'frame11.png'))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, 'gt', item , 'frame10i11.png'))).cuda().unsqueeze(0))


class Ucf(TripletTestSet):
    def __init__(self, db_dir):
        super(Ucf, self).__init__()
        self.im_list = os.listdir(db_dir)

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(self.transform(Image.open(join(db_dir, item , 'frame_00.png'))).cuda().unsqueeze(0))
            self.input1_list.append(self.transform(Image.open(join(db_dir, item , 'frame_02.png'))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, item , 'frame_01_gt.png'))).cuda().unsqueeze(0))


class Snufilm(TripletTestSet):
    def __init__(self, db_dir, mode):
        super(Snufilm, self).__init__()     
        self.mode = mode
        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        with open(join(db_dir, 'test-{}.txt'.format(mode)), 'r') as f:
            triplet_list = f.read().splitlines()
        self.im_list = []
        for i, triplet in enumerate(triplet_list, 1):
            self.im_list.append('{}-{}'.format(mode, str(i).zfill(3)))
            lst = triplet.split(' ')
            self.input0_list.append(self.transform(Image.open(join(db_dir, lst[0]))).cuda().unsqueeze(0))
            self.input1_list.append(self.transform(Image.open(join(db_dir, lst[2]))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, lst[1]))).cuda().unsqueeze(0))


class Snufilm_easy(Snufilm):
    def __init__(self, db_dir):
        db_dir = db_dir[:-5]
        super(Snufilm_easy, self).__init__(db_dir, 'easy')

class Snufilm_medium(Snufilm):
    def __init__(self, db_dir):
        db_dir = db_dir[:-7]
        super(Snufilm_medium, self).__init__(db_dir, 'medium')

class Snufilm_hard(Snufilm):
    def __init__(self, db_dir):
        db_dir = db_dir[:-5]
        super(Snufilm_hard, self).__init__(db_dir, 'hard')

class Snufilm_extreme(Snufilm):
    def __init__(self, db_dir):
        db_dir = db_dir[:-8]
        super(Snufilm_extreme, self).__init__(db_dir, 'extreme')

class VFITex_triplet:
    def __init__(self, db_dir):
        self.seq_list = os.listdir(db_dir)
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #outptu tensor in [-1,1]


    def eval(self, model, sample_func, metrics=['PSNR', 'SSIM'], output_dir=None, output_name=None, resume=False):
        model.eval()
        results_dict = {k : [] for k in metrics}

        start_idx = 0
        if resume:
            # fill in results_dict with prev results and find where to start from
            assert os.path.exists(join(output_dir, 'results.txt')), 'no res file found to resume from!'
            with open(join(output_dir, 'results.txt'), 'r') as f:
                prev_lines = f.readlines()
                for line in prev_lines:
                    if len(line) < 2:
                        continue
                    cur_res = ast.literal_eval(line.strip().split('-- ')[1].split('time')[0]) #parse dict from string
                    for k in metrics:
                        results_dict[k].append(float(cur_res[k]))
                    start_idx += 1
        
        logfile = open(join(output_dir, 'results.txt'), 'a')

        for idx, seq in enumerate(self.seq_list):
            if resume and idx < start_idx:
                assert len(glob.glob(join(output_dir, seq, '*.png'))) > 0, f'skipping idx {idx} but output not found!'
                continue

            print(f'Evaluating {seq}')
            t0 = time.time()

            seqpath = join(self.db_dir, seq)
            if not exists(join(output_dir, seq)):
                os.makedirs(join(output_dir, seq))

            # interpolate between every 2 frames
            gt_list, out_list, inputs_list = [], [], []
            tmp_dict = {k : [] for k in metrics}
            num_frames = len([f for f in os.listdir(seqpath) if f.endswith('.png')])
            for t in range(1, num_frames-5, 2):
                im0 = Image.open(join(seqpath, str(t+2).zfill(3)+'.png'))
                im1 = Image.open(join(seqpath, str(t+3).zfill(3)+'.png'))
                im2 = Image.open(join(seqpath, str(t+4).zfill(3)+'.png'))
                # center crop if 4K
                if '4K' in seq:
                    w, h  = im0.size
                    im0 = TF.center_crop(im0, (h//2, w//2))
                    im1 = TF.center_crop(im1, (h//2, w//2))
                    im2 = TF.center_crop(im2, (h//2, w//2))
                im0 = self.transform(im0).cuda().unsqueeze(0)
                im1 = self.transform(im1).cuda().unsqueeze(0)
                im2 = self.transform(im2).cuda().unsqueeze(0)

                with torch.no_grad():
                    with model.ema_scope():
                        # form condition tensor and define shape of latent rep
                        xc = {'prev_frame': im0, 'next_frame': im2}
                        c, phi_prev_list, phi_next_list = model.get_learned_conditioning(xc)
                        shape = (model.channels, c.shape[2], c.shape[3])
                        # run sampling and get denoised latent rep
                        out = sample_func(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None)
                        if isinstance(out, tuple): # using ddim
                            out = out[0]
                        # reconstruct interpolated frame from latent
                        out = model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
                        out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]

                for metric in metrics:
                    score = getattr(utility, 'calc_{}'.format(metric.lower()))(im1, out, [im0, im2])[0].item()
                    tmp_dict[metric].append(score)

                imwrite(out, join(output_dir, seq, 'frame{}.png'.format(t+3)), value_range=(-1, 1), normalize=True)

            # compute sequence-level scores
            for metric in metrics:
                results_dict[metric].append(np.mean(tmp_dict[metric]))

            # log
            msg = '{:<15s} -- {}'.format(seq, {k: round(results_dict[k][-1], 3) for k in metrics}) + f'    time taken: {round(time.time()-t0,2)}' + '\n'
            print(msg, end='')
            logfile.write(msg)
        
        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]), 3) for k in metrics}) + '\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()




class Davis90_triplet:
    def __init__(self, db_dir):
        self.seq_list = sorted(os.listdir(db_dir))
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #outptu tensor in [-1,1]


    def eval(self, model, sample_func, metrics=['PSNR', 'SSIM'], output_dir=None, output_name=None, resume=False):
        model.eval()
        results_dict = {k : [] for k in metrics}

        start_idx = 0
        if resume:
            # fill in results_dict with prev results and find where to start from
            assert os.path.exists(join(output_dir, 'results.txt')), 'no res file found to resume from!'
            with open(join(output_dir, 'results.txt'), 'r') as f:
                prev_lines = f.readlines()
                for line in prev_lines:
                    if len(line) < 2:
                        continue
                    cur_res = ast.literal_eval(line.strip().split('-- ')[1].split('time')[0]) #parse dict from string
                    for k in metrics:
                        results_dict[k].append(float(cur_res[k]))
                    start_idx += 1
        
        logfile = open(join(output_dir, 'results.txt'), 'a')

        for idx, seq in enumerate(self.seq_list):
            if resume and idx < start_idx:
                assert len(glob.glob(join(output_dir, seq, '*.png'))) > 0, f'skipping idx {idx} but output not found!'
                continue

            print(f'Evaluating {seq}')
            t0 = time.time()

            seqpath = join(self.db_dir, seq)
            if not exists(join(output_dir, seq)):
                os.makedirs(join(output_dir, seq))

            # interpolate between every 2 frames
            gt_list, out_list, inputs_list = [], [], []
            tmp_dict = {k : [] for k in metrics}
            num_frames = len(os.listdir(seqpath))
            for t in range(0, num_frames-6, 2):
                im3 = Image.open(join(seqpath, str(t+2).zfill(5)+'.jpg'))
                im4 = Image.open(join(seqpath, str(t+3).zfill(5)+'.jpg'))
                im5 = Image.open(join(seqpath, str(t+4).zfill(5)+'.jpg'))

                im3 = self.transform(im3).cuda().unsqueeze(0)
                im4 = self.transform(im4).cuda().unsqueeze(0)
                im5 = self.transform(im5).cuda().unsqueeze(0)

                with torch.no_grad():
                    with model.ema_scope():
                        # form condition tensor and define shape of latent rep
                        xc = {'prev_frame': im3, 'next_frame': im5}
                        c, phi_prev_list, phi_next_list = model.get_learned_conditioning(xc)
                        shape = (model.channels, c.shape[2], c.shape[3])
                        # run sampling and get denoised latent rep
                        out = sample_func(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None)
                        if isinstance(out, tuple): # using ddim
                            out = out[0]
                        # reconstruct interpolated frame from latent
                        out = model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
                        out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]

                for metric in metrics:
                    score = getattr(utility, 'calc_{}'.format(metric.lower()))(im4, out, [im3, im5])[0].item()
                    tmp_dict[metric].append(score)

                imwrite(out, join(output_dir, seq, 'frame{}.png'.format(t+3)), value_range=(-1, 1), normalize=True)

            # compute sequence-level scores
            for metric in metrics:
                results_dict[metric].append(np.mean(tmp_dict[metric]))

            # log
            msg = '{:<15s} -- {}'.format(seq, {k: round(results_dict[k][-1], 3) for k in metrics}) + f'    time taken: {round(time.time()-t0,2)}' + '\n'
            print(msg, end='')
            logfile.write(msg)
        
        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]), 3) for k in metrics}) + '\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()



class Ucf101_triplet:
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #outptu tensor in [-1,1]

        self.im_list = os.listdir(db_dir)

        self.input3_list = []
        self.input5_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input3_list.append(self.transform(Image.open(join(db_dir, item , 'frame1.png'))).cuda().unsqueeze(0))
            self.input5_list.append(self.transform(Image.open(join(db_dir, item , 'frame2.png'))).cuda().unsqueeze(0))
            self.gt_list.append(self.transform(Image.open(join(db_dir, item , 'framet.png'))).cuda().unsqueeze(0))

    def eval(self, model, sample_func, metrics=['PSNR', 'SSIM'], output_dir=None, output_name='output.png', resume=False):
        model.eval()
        results_dict = {k : [] for k in metrics}

        start_idx = 0
        if resume:
            # fill in results_dict with prev results and find where to start from
            assert os.path.exists(join(output_dir, 'results.txt')), 'no res file found to resume from!'
            with open(join(output_dir, 'results.txt'), 'r') as f:
                prev_lines = f.readlines()
                for line in prev_lines:
                    if len(line) < 2:
                        continue
                    cur_res = ast.literal_eval(line.strip().split('-- ')[1].split('time')[0]) #parse dict from string
                    for k in metrics:
                        results_dict[k].append(float(cur_res[k]))
                    start_idx += 1
        
        logfile = open(join(output_dir, 'results.txt'), 'a')

        for idx in range(len(self.im_list)):
            if resume and idx < start_idx:
                assert os.path.exists(join(output_dir, self.im_list[idx], output_name)), f'skipping idx {idx} but output not found!'
                continue

            print(f'Evaluating {self.im_list[idx]}')
            t0 = time.time()

            if not exists(join(output_dir, self.im_list[idx])):
                os.makedirs(join(output_dir, self.im_list[idx]))

            with torch.no_grad():
                with model.ema_scope():
                    # form condition tensor and define shape of latent rep
                    xc = {'prev_frame': self.input3_list[idx], 'next_frame': self.input5_list[idx]}
                    c, phi_prev_list, phi_next_list = model.get_learned_conditioning(xc)
                    shape = (model.channels, c.shape[2], c.shape[3])
                    # run sampling and get denoised latent rep
                    out = sample_func(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None)
                    if isinstance(out, tuple): # using ddim
                        out = out[0]
                    # reconstruct interpolated frame from latent
                    out = model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
                    out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]

            gt = self.gt_list[idx]

            for metric in metrics:
                score = getattr(utility, 'calc_{}'.format(metric.lower()))(gt, out, [self.input3_list[idx], self.input5_list[idx]])[0].item()
                results_dict[metric].append(score)

            imwrite(out, join(output_dir, self.im_list[idx], output_name), value_range=(-1, 1), normalize=True)

            msg = '{:<15s} -- {}'.format(self.im_list[idx], {k: round(results_dict[k][-1],3) for k in metrics}) + f'    time taken: {round(time.time()-t0,2)}' + '\n'
            print(msg, end='')
            logfile.write(msg)

        msg = '{:<15s} -- {}'.format('Average', {k: round(np.mean(results_dict[k]),3) for k in metrics}) + '\n\n'
        print(msg, end='')
        logfile.write(msg)
        logfile.close()