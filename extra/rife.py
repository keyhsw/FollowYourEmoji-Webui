import cv2
import torch
from torch.nn import functional as F
import warnings
import os
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

class RIFE:
    _instance = None

    def __new__(cls, model_path='./ckpt_models/rife'):
        if cls._instance is None:
            cls._instance = super(RIFE, cls).__new__(cls)
            cls._instance.initialize(model_path)
        return cls._instance

    def initialize(self, model_path):
        try:
            try:
                from model.RIFE_HDv2 import Model
                self.model = Model()
                self.model.load_model(model_path, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                self.model = Model()
                self.model.load_model(model_path, -1)
                print("Loaded v3.x HD model.")
        except:
            try:
                from model.RIFE_HD import Model
                self.model = Model()
                self.model.load_model(model_path, -1)
                print("Loaded v1.x HD model")
            except:
                from model.RIFE import Model
                self.model = Model()
                self.model.load_model(model_path, -1)
                print("Loaded ArXiv-RIFE model")

        self.model.eval()
        self.model.device()

    def interpolate(self, img0, img1, exp=4, ratio=0, rthreshold=0.02, rmaxcycles=8):
        if isinstance(img0, str) and isinstance(img1, str):
            if img0.endswith('.exr') and img1.endswith('.exr'):
                img0 = cv2.imread(img0, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
                img1 = cv2.imread(img1, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
                img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
                img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)
            else:
                img0 = cv2.imread(img0, cv2.IMREAD_UNCHANGED)
                img1 = cv2.imread(img1, cv2.IMREAD_UNCHANGED)
                img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
                img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        elif isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor):
            img0 = img0.to(device)
            img1 = img1.to(device)
        else:
            raise ValueError("Input images must be either file paths or torch tensors")

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(rmaxcycles):
                    middle = self.model.inference(tmp_img0, tmp_img1)
                    middle_ratio = (img0_ratio + img1_ratio) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for i in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        return [img[:, :, :h, :w] for img in img_list]

    def unload(self):
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        print("RIFE model unloaded and CUDA cache cleared.")

    @classmethod
    def reset(cls):
        if cls._instance is not None:
            cls._instance.unload()
            cls._instance = None
        print("RIFE instance reset.")

def save_images(img_list, output_dir='output', img0_path='', img1_path=''):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(img_list):
        if img0_path.endswith('.exr') and img1_path.endswith('.exr'):
            cv2.imwrite(os.path.join(output_dir, f'img{i}.exr'), img[0].cpu().numpy().transpose(1, 2, 0), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        else:
            cv2.imwrite(os.path.join(output_dir, f'img{i}.png'), (img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--img', dest='img', nargs=2, required=True)
    parser.add_argument('--exp', default=4, type=int)
    parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
    parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
    parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
    parser.add_argument('--model', dest='modelDir', type=str, default='./ckpt_models/rife', help='directory with trained model files')
    args = parser.parse_args()

    rife = RIFE(args.modelDir)
    img_list = rife.interpolate(args.img[0], args.img[1], args.exp, args.ratio, args.rthreshold, args.rmaxcycles)
    save_images(img_list, img0_path=args.img[0], img1_path=args.img[1])
    rife.unload()
