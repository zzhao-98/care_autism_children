from torch.utils.data.dataset import Dataset
import os, glob
import os.path

from pytorchvideo.data.encoded_video import EncodedVideo


class FABO_body(Dataset):
    def __init__(self, root, transform=None):
        self.path_lists = []
        self.labels = []
        self.transform = transform

        for file in glob.glob(os.path.join(root, '*.avi')):
            if file.split('-')[2] == '2':
                pass
            else:
                self.path_lists.append(file)
                label = file.split('-')[-1].split('.')[0]
                label_idx = label_to_idx[label]
                self.labels.append(label_idx)

    def __getitem__(self, index):
        label = self.labels[index]
        video_path = self.path_lists[index]

        video = EncodedVideo.from_path(video_path)

        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        if self.transform:
            video_data = self.transform(video_data)

        inputs = video_data["video"]

        return (inputs, label, video_path)

    def __len__(self):
        return len(self.labels)