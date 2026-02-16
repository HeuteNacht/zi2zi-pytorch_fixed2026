import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import io
import torch

class DatasetFromObj(data.Dataset):
    def __init__(self, obj_path, resize_to=256, start_from=0, input_nc=3, **kwargs):
        super(DatasetFromObj, self).__init__()
        self.obj_path = obj_path
        self.start_from = start_from
        self.resize_to = resize_to
        if not os.path.exists(obj_path):
            raise Exception(f"数据文件不存在: {obj_path}")
        with open(obj_path, 'rb') as f:
            self.dataset = pickle.load(f)
        if isinstance(self.dataset, dict):
            self.dataset = list(self.dataset.values())
        print(f"✅ 数据加载成功，总样本数: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        # 拆解 Label (防止嵌套 tuple)
        label = item[0]
        while isinstance(label, (tuple, list)): 
            label = label[0]
        # 拆解图片 Bytes
        img_bytes = item[1]
        while isinstance(img_bytes, (tuple, list)): 
            img_bytes = img_bytes[0]
        
        img_A, img_B = self.process(img_bytes)
        final_label = int(label) - self.start_from
        return final_label, img_A, img_B

    def process(self, img_bytes):
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(f"图片读取错误: {e}")
            dummy = torch.zeros(3, self.resize_to, self.resize_to)
            return dummy, dummy
        
        w, h = img.size
        img_A = img.crop((0, 0, w//2, h))
        img_B = img.crop((w//2, 0, w, h))
        # 适配 Pillow LANCZOS
        img_A = img_A.resize((self.resize_to, self.resize_to), Image.LANCZOS)
        img_B = img_B.resize((self.resize_to, self.resize_to), Image.LANCZOS)
        return self.transform(img_A), self.transform(img_B)

    def transform(self, img):
        img = np.array(img, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        img = (img / 127.5) - 1.0
        return torch.from_numpy(img)
