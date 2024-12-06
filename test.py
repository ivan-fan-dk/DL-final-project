# `test.py` and `test_se.py` are sister files.
import torch

from imageio.v3 import imread
from skimage.transform import resize
from skimage.util import img_as_float
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS
# from utils import tensor2array
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
# parser.add_argument("--img-height", default=128, type=int, help="Image height")
# parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--img-height", default=352, type=int, help="Image height")
parser.add_argument("--img-width", default=1216, type=int, help="Image width")
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

    disp_net = DispNetS().to(device)
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
            # max_value = depth[depth < np.inf].max().item()
            depth = depth.squeeze().numpy()
            norm_array = (depth - depth.min())/(depth.max() - depth.min())
            norm_array[norm_array == np.inf] = np.nan
            depth = norm_array

            # depth = (255*tensor2array(depth, max_value=None, colormap='rainbow')).astype(np.uint8)
            # depth = np.transpose((255*tensor2array(depth, max_value=None, colormap='rainbow')).astype(np.uint8), (1,2,0))
            img = np.transpose(img, (1,2,0))

            gt = resize(gt, (args.img_height, args.img_width))

            # Normalize gt (map minimal nonzero value to zero and maximal value to 1)
            gt_min = gt[gt != 0.].min()
            gt = (gt - gt_min)/(gt.max() - gt_min)
            gt[gt < 0.] = 0.

            plt.hist(depth[gt != 0.].flatten(), bins=100, label='depth')
            plt.hist(gt[gt != 0.].flatten(), bins=100, label="gt")
            
            plt.legend()
            plt.savefig(f"hist/hist_{file_name}.png")
            plt.clf()

            print()
            print("image shape: ", img.shape, "image max: ", img.max(), "image min: ", img.min())
            print("depth shape: ", depth.shape, "depth max: ", depth.max(), "depth min: ", depth.min())
            print("gt shape: ", gt.shape, "gt max: ", gt.max(), "gt min: ", gt.min())
            print()
            # imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))
            show_results(output_dir, img, depth, gt, file_name)

def show_results(output_dir, img, depth, gt, filename):
    # Create figure and subplots
    fig = plt.figure(figsize=(12, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.2])  # Adjust height for table

    # Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Input Image', fontsize=14)
    ax1.axis('off')
    ax1.set_aspect('equal')  # Force equal aspect ratio

    # Depth Prediction Map
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(depth, cmap="rainbow", aspect='equal')  # Force equal aspect ratio
    ax2.set_title('Depth Prediction Map', fontsize=14)
    ax2.axis('off')

    # Depth Error Map
    ax3 = fig.add_subplot(gs[2, 0])
    error_map = np.where(gt == 0., 0., np.abs(depth - gt))
    rainbow_cmap = plt.cm.rainbow(np.linspace(0, 1, 256))  # Get rainbow colormap
    custom_colors = np.vstack(([0, 0, 0, 1], rainbow_cmap))  # Add black for zero
    custom_cmap = ListedColormap(custom_colors)
    im = ax3.imshow(error_map, cmap=custom_cmap, aspect='equal')  # Force equal aspect ratio
    ax3.set_title('Depth Error Map', fontsize=14)
    ax3.axis('off')

    # Add colorbar for the error map
    fig.colorbar(im, ax=ax3, orientation="horizontal", pad=0.1)

    # Metrics Table
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axis('off')  # Turn off the axes for the table
    depth = depth[gt != 0.]
    gt = gt[gt != 0.]
    metrics = [
        ['RMSE', 'SI-Log', 'sqErrorRel', 'absErrorRel'],
        [RMSE(depth, gt), SILog(depth, gt), sqErrorRel(depth, gt), absErrorRel(depth, gt)]
    ]
    table = ax4.table(cellText=metrics, cellLoc='center', loc='center', colWidths=[0.2] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust table scaling

    # Add a main title and save
    plt.suptitle(f"{''.join(filename.split('_sync_image_'))}", fontsize=16)#, y=0.98)
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.close(fig)

def SILog(x, y):
    """
    Scale-Invariant Logarithmic Error
    x: predicted
    y: ground truth
    """
    mask = np.where((x > 0.) & (y > 0.))
    x, y = x[mask], y[mask]
    d = np.log(x) - np.log(y)
    return "{:.2f}".format(np.mean(np.square(d)) - np.square(np.mean(d)))

def sqErrorRel(x, y):
    """
    Squared relative error
    x: predicted
    y: ground truth
    """
    return "{:.2f}".format(np.mean(np.abs(x - y) / y))

def absErrorRel(x, y):
    """
    Absolute relative error
    x: predicted
    y: ground truth
    """
    return "{:.2f}".format(np.mean(np.square(x - y) / y))

def RMSE(x, y):
    """
    Root Mean Squared Error
    x: predicted
    y: ground truth
    """
    return "{:.2f}".format(np.sqrt(np.mean(np.square(x - y))))

if __name__ == '__main__':
    main()
