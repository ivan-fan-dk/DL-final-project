# `test.py` and `test_se.py` are sister files.
# Known issues:
# Note gt is scattered and with real depth values, while our depth map is a graph with values between 0 and 1.

import torch

from imageio import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_float
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import SE_DispNetS
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    disp_net = SE_DispNetS().to(device)
    weights = torch.load(args.pretrained, map_location=device)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):

        # Load ground truth depth map
        file_prefix, file_suffix = file.split('/')[-1].split('sync_image')
        gt_path = "test_gt/" + file_prefix + "sync_groundtruth_depth" + file_suffix
        print("gt_path: ", gt_path)
        gt = imread(gt_path)

        img = img_as_float(imread(file))

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = resize(img, (args.img_height, args.img_width))
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        tensor_img = ((tensor_img - 0.5)/0.5).to(device)

        output = disp_net(tensor_img)[0]

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path.splitall()[1:])

        # if args.output_disp:
        #     disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
        #     imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))
        if args.output_depth:
            depth = 1./output

            # basically normalize depth
            depth = depth.detach().cpu()
            max_value = depth[depth < np.inf].max().item()
            norm_array = depth.squeeze().numpy()/max_value
            norm_array[norm_array == np.inf] = np.nan
            depth = norm_array

            # depth = (255*tensor2array(depth, max_value=None, colormap='rainbow')).astype(np.uint8)
            # depth = np.transpose((255*tensor2array(depth, max_value=None, colormap='rainbow')).astype(np.uint8), (1,2,0))
            img = np.transpose(img, (1,2,0))
            
            gt = resize(gt, (args.img_height, args.img_width))
            gt = (gt - gt.min())/(gt.max() - gt.min())
            print()
            print("image shape: ", img.shape, "image max: ", img.max(), "image min: ", img.min())
            print("depth shape: ", depth.shape, "depth max: ", depth.max(), "depth min: ", depth.min())
            print("gt shape: ", gt.shape, "gt max: ", gt.max(), "gt min: ", gt.min())
            print()
            # imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))
            show_results(output_dir, img, depth, gt, file_name)

def show_results(output_dir, img, depth, gt, filename):
    plt.figure(figsize=(9, 10))

    # Original image
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')

    # Depth map
    plt.subplot(3, 1, 2)
    plt.imshow(depth, cmap="rainbow")
    plt.title('Depth Prediction Map')
    plt.axis('off')

    # Disparity map
    plt.subplot(3, 1, 3)
    plt.imshow(depth - gt, cmap="gray")
    plt.title('Depth Error Map')
    plt.axis('off')

    plt.suptitle(f"{''.join(filename.split('_sync_image_'))}")
    plt.savefig(f"{output_dir}/{filename}.png")

if __name__ == '__main__':
    main()
