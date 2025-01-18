import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image

# Generated by Make3dDataset.shuffle_data
# test len = 107, train len = 318
make_3d_test_images = [
    "real9-p-46t0.jpg",
    "math13-p-77t0.jpg",
    "math9-p-46t0.jpg",
    "math3-p-108t0.jpg",
    "math7-p-15t0.jpg",
    "stats4-p-45t0.jpg",
    "gatesback2-p-46t0.jpg",
    "stats10-p-250t0.jpg",
    "stats6-p-251t0.jpg",
    "manmade6-p-251t0.jpg",
    "stats3-p-312t0.jpg",
    "stats9-p-219t0.jpg",
    "math5-p-169t0.jpg",
    "statroad4-p-189t0.jpg",
    "stats7-p-250t0.jpg",
    "gatesback1-p-139t0.jpg",
    "stats2-p-343t0.jpg",
    "math5-p-220t0.jpg",
    "math10-p-282t0.jpg",
    "math15-p-189t0.jpg",
    "math11-p-14t0.jpg",
    "math8-p-107t0.jpg",
    "real7-p-138t0.jpg",
    "manmade1-p-169t0.jpg",
    "stats9-p-170t0.jpg",
    "stats2-p-77t0.jpg",
    "math3-p-343t0.jpg",
    "stats9-p-312t0.jpg",
    "math1-p-77t0.jpg",
    "real6-p-15t0.jpg",
    "stats3-p-45t0.jpg",
    "stats7-p-344t0.jpg",
    "manmade6-p-107t0.jpg",
    "statroad4-p-108t0.jpg",
    "manmade7-p-107t0.jpg",
    "gatesback4-p-139t0.jpg",
    "real5-p-139t0.jpg",
    "stats8-p-220t0.jpg",
    "stats8-p-313t0.jpg",
    "real3-p-169t0.jpg",
    "stats3-p-139t0.jpg",
    "math5-p-343t0.jpg",
    "math4-p-251t0.jpg",
    "math16-p-344t0.jpg",
    "math11-p-169t0.jpg",
    "math13-p-139t0.jpg",
    "math12-p-169t0.jpg",
    "math7-p-343t0.jpg",
    "stats4-p-251t0.jpg",
    "manmade7-p-77t0.jpg",
    "math3-p-46t0.jpg",
    "statroad2-p-14t0.jpg",
    "math15-p-282t0.jpg",
    "gatesback3-p-139t0.jpg",
    "real5-p-343t0.jpg",
    "math1-p-282t0.jpg",
    "math9-p-250t0.jpg",
    "manmade5-p-14t0.jpg",
    "math3-p-77t0.jpg",
    "math4-p-344t0.jpg",
    "stats2-p-282t0.jpg",
    "statroad2-p-343t0.jpg",
    "gatesback2-p-313t0.jpg",
    "stats7-p-15t0.jpg",
    "combined1-p-108t0.jpg",
    "gatesback4-p-169t0.jpg",
    "math4-p-77t0.jpg",
    "stats11-p-281t0.jpg",
    "math7-p-46t0.jpg",
    "stats11-p-251t0.jpg",
    "math16-p-15t0.jpg",
    "math7-p-189t0.jpg",
    "real1-p-281t0.jpg",
    "math8-p-313t0.jpg",
    "stats6-p-46t0.jpg",
    "math9-p-313t0.jpg",
    "manmade6-p-169t0.jpg",
    "stats4-p-219t0.jpg",
    "math8-p-138t0.jpg",
    "math9-p-281t0.jpg",
    "real8-p-169t0.jpg",
    "combined1-p-170t0.jpg",
    "stats11-p-108t0.jpg",
    "combined1-p-220t0.jpg",
    "math12-p-76t0.jpg",
    "stats10-p-77t0.jpg",
    "math15-p-169t0.jpg",
    "math7-p-107t0.jpg",
    "stats11-p-313t0.jpg",
    "real5-p-169t0.jpg",
    "stats4-p-108t0.jpg",
    "real8-p-282t0.jpg",
    "manmade7-p-15t0.jpg",
    "gatesback1-p-169t0.jpg",
    "stats3-p-169t0.jpg",
    "manmade5-p-108t0.jpg",
    "stats4-p-282t0.jpg",
    "math16-p-189t0.jpg",
    "math14-p-220t0.jpg",
    "math16-p-282t0.jpg",
    "math12-p-251t0.jpg",
    "stats8-p-189t0.jpg",
    "stats11-p-139t0.jpg",
    "math6-p-313t0.jpg",
    "statroad2-p-313t0.jpg",
    "math12-p-282t0.jpg",
    "stats2-p-139t0.jpg",
]


class Make3dDataset(data.Dataset):
    # w: 1704, h: 2272
    file_suffix = ".jpg"
    total_number = 425
    random_seed = 121

    @staticmethod
    def shuffle_data(root_path: str):
        # split the data into train and test
        num_train = Make3dDataset.total_number * 400 // (400 + 134)
        num_test = Make3dDataset.total_number - num_train
        imgs = []
        # collect all the images in root_path
        for f in os.listdir(root_path):
            complete_path = os.path.join(root_path, f)
            if os.path.isdir(complete_path):
                continue
            if not f.endswith(Make3dDataset.file_suffix):
                continue
            base, _ = os.path.splitext(f)
            assert os.path.exists(
                os.path.join(root_path, f"{base}.dat")
            ), f"{base}.dat does not exist"
            imgs.append(f)

        assert len(imgs) == Make3dDataset.total_number, "The number of images is not correct"

        np.random.shuffle(imgs)
        test_imgs = imgs[:num_test]
        print(f"num_train: {num_train}, num_test: {num_test}")
        print(test_imgs)

    def __init__(self, root, type, sparsifier=None, modality="rgb"):
        torch.manual_seed(Make3dDataset.random_seed)
        self.imgs = []
        self.collect_images(root, type)
        self.mode = type

        self.output_size = (460, 345)
        self.net_input_size = 460 // 2, 345 // 2

    def collect_images(self, root: str, mode: str):
        assert mode in ["train", "val"]

        for f in os.listdir(root):
            complete_path = os.path.join(root, f)
            if os.path.isdir(complete_path):
                continue
            if not f.endswith(Make3dDataset.file_suffix):
                continue
            depth_map = os.path.join(root, f"{os.path.splitext(f)[0]}.dat")
            rgb_map = os.path.join(root, f)
            if f in make_3d_test_images and mode == "val":
                self.imgs.append((rgb_map, depth_map))
            elif f not in make_3d_test_images and mode == "train":
                self.imgs.append((rgb_map, depth_map))

    def train_transform(self, rgb, depth):
        # Get the parameters for the transformation
        scale = 1.0 + 0.2 * torch.rand(1).item()  # random scaling
        angle = transforms.RandomRotation.get_params((-5.0, 5.0))  # random rotation degrees
        do_flip = torch.rand(1) < 0.5  # random horizontal flip

        scaled_size = int(self.net_input_size[0] * scale), int(self.net_input_size[1] * scale)

        # perform rgb transformation
        rgb = transforms.Resize(self.net_input_size)(rgb)
        crop_params = transforms.RandomCrop.get_params(rgb, self.net_input_size)

        rgb = transforms.functional.rotate(rgb, angle)
        rgb = transforms.Resize(scaled_size)(rgb)
        rgb = transforms.functional.crop(rgb, *crop_params)
        rgb = transforms.functional.hflip(rgb) if do_flip else rgb
        rgb = transforms.ColorJitter(0.2, 0.2, 0.2)(rgb)

        # perform depth transformation
        depth_np = depth / scale
        depth_np = transforms.Resize(self.net_input_size)(depth_np)
        depth_np = transforms.functional.rotate(depth_np, angle)
        depth_np = transforms.Resize(scaled_size)(depth_np)
        depth_np = transforms.functional.crop(depth_np, *crop_params)
        depth_np = transforms.functional.hflip(depth_np) if do_flip else depth_np

        return rgb, depth_np

    def val_transform(self, rgb, depth):
        rgb = transforms.Resize(self.net_input_size)(rgb)
        rgb = rgb.to(torch.float32) / 255

        depth = transforms.Resize(self.output_size)(depth)

        return rgb, depth
    
    def __getraw__(self, rgb_path, depth_path):
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = transforms.functional.pil_to_tensor(rgb)
        depth = np.loadtxt(depth_path, dtype=np.float32)
        depth = torch.tensor(depth)
        depth = depth.unsqueeze(0)

        return rgb, depth

    def __getitem__(self, index):
        rgb_path, depth_path = self.imgs[index]
        rgb, depth = self.__getraw__(rgb_path, depth_path)

        if self.mode == "train":
            rgb, depth = self.train_transform(rgb, depth)
            rgb = rgb.to(torch.float32) / 255
            depth = transforms.Resize(self.output_size)(depth)
        else:
            rgb, depth = self.val_transform(rgb, depth)
        return rgb, depth
        # from Image, np.array to tensor

    def __len__(self):
        return len(self.imgs)


# Unlike Make3dDataset, Make3dDatasetV2 uses a fixed training dataset in the dataset,
# Default repeat numebr 50 will generate 15k images for training, keeping the same as the
# original paper.
class Make3dDatasetV2(Make3dDataset):

    def __init__(self, root, type, sparsifier=None, modality="rgb", repeat=50):
        self.repeat = repeat
        self.root = root
        super().__init__(root, type, sparsifier, modality)


    def generate_train_data(self):
        root_path = self.root
        train_path = os.path.join(root_path, "train")
        if os.path.exists(train_path):
            print(f"{train_path} already exists")
            return
        os.makedirs(train_path)
        for f in os.listdir(root_path):
            rgb_path = os.path.join(root_path, f)
            if os.path.isdir(rgb_path):
                continue
            if not f.endswith(Make3dDataset.file_suffix):
                continue    
            if f in make_3d_test_images:
                continue
            depth_path = os.path.join(root_path, f"{os.path.splitext(f)[0]}.dat")
            assert os.path.exists(depth_path), f"{depth_path} does not exist"

            rgb, depth = self.__getraw__(rgb_path, depth_path)

            for i in range(self.repeat):
                rgb_i, depth_i = self.train_transform(rgb, depth)
                depth_i = depth_i.squeeze(0)
                pil_image = transforms.functional.to_pil_image(rgb_i)
                pil_image.save(os.path.join(train_path, f"{os.path.splitext(f)[0]}_{i}{Make3dDataset.file_suffix}"))
                np.savetxt(os.path.join(train_path, f"{os.path.splitext(f)[0]}_{i}.dat"), depth_i.numpy())
    
    def collect_images(self, root, mode):
        if mode == "val":
            return super().collect_images(root, mode)
        
        train_path = os.path.join(root, "train")
        if not os.path.exists(train_path):
            return super().collect_images(root, mode)

        for f in os.listdir(train_path):
            complete_path = os.path.join(train_path, f)
            if os.path.isdir(complete_path):
                continue
            if not f.endswith(Make3dDataset.file_suffix):
                continue
            depth_map = os.path.join(train_path, f"{os.path.splitext(f)[0]}.dat")
            rgb_map = os.path.join(train_path, f)
            self.imgs.append((rgb_map, depth_map))
        
        assert len(self.imgs) == (Make3dDataset.total_number - len(make_3d_test_images)) * self.repeat, "The number of images is not correct"
    
    def __getitem__(self, index):
        if self.mode == "val":
            return super().__getitem__(index)
        rgb_path, depth_path = self.imgs[index]
        rgb, depth = self.__getraw__(rgb_path, depth_path)

        rgb = rgb.to(torch.float32) / 255
        depth = transforms.Resize(self.output_size)(depth)

        return rgb, depth


if __name__ == "__main__":
    from network import FCRN
    import sys

    Make3dDatasetV2(root=sys.argv[1], type="train").generate_train_data()

    # root = sys.argv[1]
    # myDataset = Make3dDataset(root, "train")
    # myTestDataset = Make3dDataset(root, "val")

    # net = FCRN.ResNet(layers=50, output_size=myDataset.output_size).cuda()
    # rgb, depth = myDataset[0]
    # rgb_cuda = rgb.unsqueeze(0).cuda()
    # predict = net(rgb_cuda)
    # print(predict.shape)

    # rgb, depth = myTestDataset[0]
    # rgb_cuda = rgb.unsqueeze(0).cuda()
    # predict = net(rgb_cuda)
    # print(predict.shape)
