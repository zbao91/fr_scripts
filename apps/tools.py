import time
import torch

from torchvision import transforms as F
from PIL import Image
from torchvision.transforms import ToTensor

class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.

    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed
        return torch.tensor(elapsed)

class Logger(object):
    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class normalize(torch.nn.Module):
    def __init__(self):
        super(normalize, self).__init__()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

# 比较两张图片的空间距离
def CompareByEmbd(embeddings):
    dists = (embeddings[0] - embeddings[1]).norm().item()
    return dists

# 直接使用模型比较两张图片
def CompareByImage(model, device, imgs):
    aligned = torch.stack(imgs).to(device)
    embeddings = model(aligned).detach().cpu()
    dists = (embeddings[0] - embeddings[1]).norm().item()
    return dists

class Image_Processing():
    """
    图片处理
    """
    def __init__(self):
        pass

    def read_img(self, imgs_path):
        """
        read image
        """
        tmp_list = []
        for i in imgs_path:
            img = Image.open(i)
            tmp_list.append(self.img_convert(img))
        return tmp_list

    def img_convert(self, img):
        """
        convert img from other format to RGB format
        """
        if img.mode == 'RGBA':
            try:
                img.load()
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                background.save("sample_2.jpg", "JPEG", quality=100)
                img = Image.open("sample_2.jpg")
            except:
                raise Exception('error')
        img = ToTensor()(img)
        return img