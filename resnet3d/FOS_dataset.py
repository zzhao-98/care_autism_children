from torch.utils.data.dataset import Dataset
import os, glob
import torch
import os.path
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import torch.nn.functional as F

####################
# SlowFast transform
####################

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second
start_sec = 0
end_sec = start_sec + clip_duration
class FOS_set(Dataset):
    def __init__(self, df_dataset, list_caring_labels, transform=None):
        # self.paths_video = df_dataset['video_path'].tolist()
        # self.labels = df_dataset['labels'].tolist()
        self.df = df_dataset
        self.transform = transform
        self.list_caring_labels = list_caring_labels


    def __getitem__(self, index):
        # label processing
        onehot_labels = []
        for caring_label in self.list_caring_labels:
            onehot_labels.append(int(self.df.iloc[index][caring_label]))
        onehot_labels = torch.tensor(onehot_labels, dtype=torch.float32)
        # onehot_labels = F.one_hot(onehot_labels, 2)
        # onehot_labels = torch.reshape(onehot_labels, (-1,))
        # onehot_labels = onehot_labels.to(torch.float32)
        # video processings
        video_path = self.df.iloc[index]['video_path']
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        if self.transform:
            video_data = self.transform(video_data)
        inputs = video_data["video"]
        return (inputs, onehot_labels)

    def __len__(self):
        return len(self.df)

class FOS_dataset(Dataset):
    def __init__(self, df_dataset, transform=None):
        self.paths_video = df_dataset['video_path'].tolist()
        self.labels = df_dataset['label'].tolist()
        self.transform = transform
        self.label_dict = {'C+': 0, 'C-': 1, 'EA': 2, 'PN': 3}


    def __getitem__(self, index):
        label = self.labels[index]
        label = self.label_dict[label]
        video_path = self.paths_video[index]
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        if self.transform:
            video_data = self.transform(video_data)
        inputs = video_data["video"]
        return (inputs, label)

    def __len__(self):
        return len(self.labels)