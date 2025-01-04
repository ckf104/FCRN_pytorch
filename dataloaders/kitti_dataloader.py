import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader


class KITTIDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(KITTIDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (228, 912)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # Scipy affine_transform produced RuntimeError when the depth map was
        # given as a 'numpy.ndarray'
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np

import torchvision.transforms as torchtrans
import torch.utils.data as data
import torch
from PIL import Image
import os

class MyKITTIDataset(data.Dataset):

    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        self.mode = type
        self.output_size = (228, 756)
        assert self.mode in ["train", "val"]

        if self.mode == "train":
            depth_root_path = os.path.join(root, "train")
            rgb_root_path = os.path.join(root, "rgb_train")
        else:
            depth_root_path = os.path.join(root, "val")
            rgb_root_path = os.path.join(root, "rgb_val")
        
        self.imgs, _ = self.collect_images(depth_root_path, rgb_root_path)
    
    def __len__(self):
        return len(self.imgs)

    def __getraw__(self, index):
        rgb_path, depth_path = self.imgs[index]
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = torchtrans.functional.pil_to_tensor(rgb)
        rgb = rgb.to(torch.float32) / 255

        # depth is stored in a png file as uint16
        depth = np.array(Image.open(depth_path), dtype=int)
        depth = torch.from_numpy(depth.astype(np.float32)).float() / 256.0
        depth = depth.unsqueeze(0)
        return rgb, depth
    
    def transform(self, rgb, depth):
        bilinear = torchtrans.functional.InterpolationMode.BILINEAR
        nearest = torchtrans.functional.InterpolationMode.NEAREST
        bilinear_resize = torchtrans.Resize(self.output_size, interpolation=bilinear)    
        nearest_resize = torchtrans.Resize(self.output_size, interpolation=nearest)

        rgb = bilinear_resize(rgb)
        depth = nearest_resize(depth)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        rgb, depth = self.transform(rgb, depth)
        return rgb, depth

    @staticmethod
    def collect_images(root_depth_path, root_rgb_path): 
        imgs = []
        folder_count = 0
        for file in os.listdir(root_depth_path):
           t = '_'.join(file.split('_')[:3])
           depth_path = os.path.join(root_depth_path, file, 'proj_depth', 'groundtruth')
           rgb_path = os.path.join(root_rgb_path, t, file)
           if not os.path.exists(depth_path):
               raise FileNotFoundError('Depth path not found: {}'.format(depth_path))
           if not os.path.exists(rgb_path):
               continue
           
           folder_count += 1
           for folder in os.listdir(depth_path):
               depth_folder = os.path.join(depth_path, folder)
               rgb_folder = os.path.join(rgb_path, folder, 'data')
               
               for depth_img in os.listdir(depth_folder):
                   if not depth_img.endswith('.png'):
                       continue
                   imgs.append((os.path.join(rgb_folder, depth_img), os.path.join(depth_folder, depth_img)))
                   if not os.path.exists(os.path.join(rgb_folder, depth_img)):
                       raise FileNotFoundError('RGB file not found: {}'.format(os.path.join(rgb_folder, depth_img)))

        return imgs, folder_count
    
    @staticmethod
    def unzip_rawdata(train_path, val_path, target_train_path, target_val_path):
        import zipfile
        for file in os.listdir(train_path):
            if not file.endswith('.zip'):
                continue
            p = os.path.join(train_path, file)
            with zipfile.ZipFile(p, 'r') as zip_ref:
                zip_ref.extractall(target_train_path)

        for file in os.listdir(val_path):
            if not file.endswith('.zip'):
                continue
            p = os.path.join(val_path, file)
            with zipfile.ZipFile(p, 'r') as zip_ref:
                zip_ref.extractall(target_val_path)
    
    @staticmethod
    def validation(train_depth_path, train_rgb_path, val_depth_path, val_rgb_path):
        for root_depth_path, root_rgb_path in zip([train_depth_path, val_depth_path], [train_rgb_path, val_rgb_path]):
            imgs, folder_count = MyKITTIDataset.collect_images(root_depth_path, root_rgb_path)
            img_count = len(imgs)
            print(f'depth path: {root_depth_path}, Folder count: {folder_count}, Image count: {img_count}')

if __name__ == "__main__":
    # train_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\train_rgb"
    # val_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\val_rgb"
    # target_train_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\rgb_train"
    # target_val_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\rgb_val"
    # MyKITTIDataset.unzip_rawdata(train_path, val_path, target_train_path, target_val_path)

    # train_depth_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\train"
    # train_rgb_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\rgb_train"
    # val_depth_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\val"
    # val_rgb_path = "C:\\Users\\24147\\Documents\\Dataset\\kitti\\rgb_val"
    # MyKITTIDataset.validation(train_depth_path, train_rgb_path, val_depth_path, val_rgb_path)

    # from network import FCRN
    # import torch
 
    # net = FCRN.ResNet(layers=50, output_size=(228, 756)).cuda()
    # rgb = torch.randn(3, 228, 756)
    # rgb_cuda = rgb.unsqueeze(0).cuda()
    # predict = net(rgb_cuda)
    # print(predict.shape)
    p = "C:\\Users\\24147\\Documents\\Dataset\\kitti"
    dataset = MyKITTIDataset(p, "train")
    dataset.__getraw__(0)