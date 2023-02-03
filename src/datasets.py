from os import path, scandir
import json

import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader

class FFPP(Dataset):
    def __init__(self, root_dir, transform, types, compressions, split='test', num_frames=25):

        self.TYPE_DIRS = {
            'REAL': 'data/original_sequences/youtube/',
            'DF'  : 'data/manipulated_sequences/Deepfakes/',
            'FS'  : 'data/manipulated_sequences/FaceSwap/',
            'F2F' : 'data/manipulated_sequences/Face2Face/',
            'NT'  : 'data/manipulated_sequences/NeuralTextures/',
            'FSH' : 'data/manipulated_sequences/FaceShifter/',
        }
        self.root = path.expanduser(root_dir)
        self.types = types
        self.compressions = compressions
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        self._build_video_table()
        self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        From RealForensics paper, they use the first 110 frames of the testing video
        here we use the default num_frames per clip = 25
        and we drop the last clip (10 frames) to reduce variance
        therefore, we take the first 4 clips.
        '''
        n_clips = int(110 / self.num_frames)
        df_type, comp, idx = self.data_list[idx]
        reader = VideoReader(path.join(self.root, self.TYPE_DIRS[df_type], comp, 'cropped_faces', f'{idx}.avi'), "video")
        frames = []
        count = 0
        for frame in reader:
            count += 1
            if count > self.num_frames * n_clips:
                break
            frames.append(frame['data'])

        frames = self.transform(torch.stack(frames))
        frames = frames.view(n_clips, -1, *frames.shape[1:]).transpose(1, 2)
        return frames, 1 if df_type == 'REAL' else 0

    def _build_video_table(self):
        self.video_table = {}

        for df_type in self.types:
            self.video_table[df_type] = {}
            for comp in self.compressions:
                subdir = path.join(self.root, self.TYPE_DIRS[df_type], '')
                # video table
                videos = [f.name for f in scandir(path.join(subdir, f'{comp}/cropped_faces')) if '.avi' in f.name]

                self.video_table[df_type][comp] = videos
        
    def _build_data_list(self):
        self.data_list = []
        
        with open(path.join(self.root, 'splits', f'{self.split}.json')) as f:
            idxs = json.load(f)
            
        for df_type in self.types:
            for comp in self.compressions:
                adj_idxs = [i for inner in idxs for i in inner] if df_type == 'REAL' else ['_'.join(idx) for idx in idxs] + ['_'.join(reversed(idx)) for idx in idxs]

                for idx in adj_idxs:
                    if f'{idx}.avi' in self.video_table[df_type][comp]:
                        self.data_list.append((df_type, comp, idx))
                    else:
                        print(f'Warning: video {path.join(self.root, self.TYPE_DIRS[df_type], comp, "cropped_faces", idx)} does not present in the processed dataset.')

