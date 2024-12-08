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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=352, type=int, help="Image height")
parser.add_argument("--img-width", default=1216, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--output-hist-dir", default='hist', type=str, help="Output directory for histograms")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value!')
        return

    # Validate input dimensions
    if args.img_height <= 0 or args.img_width <= 0:
        print('Image dimensions must be positive!')
        return

    # Create necessary directories
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    hist_dir = Path(args.output_hist_dir)
    hist_dir.makedirs_p()

    disp_net = DispNetS().to(device)
    weights = torch.load(args.pretrained, map_location=device)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):
        try:
            # Load ground truth depth map
            file_prefix, file_suffix = file.split('/')[-1].split('sync_image')
            gt_path = "test_gt/" + file_prefix + "sync_groundtruth_depth" + file_suffix
            print("gt_path: ", gt_path)
            
            # Handle file reading with error checking
            try:
                gt = imread(gt_path)
                img = img_as_float(imread(file))
            except Exception as e:
                print(f"Error reading files: {str(e)}")
                continue

            h,w,_ = img.shape
            if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                img = resize(img, (args.img_height, args.img_width))
            img = np.transpose(img, (2, 0, 1))

            tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
            tensor_img = ((tensor_img - 0.5)/0.5).to(device)

            output = disp_net(tensor_img)[0]

            file_path, file_ext = file.relpath(args.dataset_dir).splitext()
            file_name = '-'.join(file_path.splitall()[1:])

            if args.output_depth:
                depth = 1./output
                depth = depth.detach().cpu()
                depth = depth.squeeze().numpy()
                norm_array = (depth - depth.min())/(depth.max() - depth.min())
                norm_array[norm_array == np.inf] = np.nan
                depth = norm_array

                img = np.transpose(img, (1,2,0))
                gt = resize(gt, (args.img_height, args.img_width))

                # Normalize gt (map minimal nonzero value to zero and maximal value to 1)
                gt_min = gt[gt != 0.].min()
                gt = (gt - gt_min)/(gt.max() - gt_min)
                gt[gt < 0.] = 0.

                plt.hist(depth[gt != 0.].flatten(), bins=100, label='depth')
                plt.hist(gt[gt != 0.].flatten(), bins=100, label="gt")
                
                plt.legend()
                plt.savefig(f"{hist_dir}/hist_{file_name}.png")
                plt.clf()

                print()
                print("image shape: ", img.shape, "image max: ", img.max(), "image min: ", img.min())
                print("depth shape: ", depth.shape, "depth max: ", depth.max(), "depth min: ", depth.min())
                print("gt shape: ", gt.shape, "gt max: ", gt.max(), "gt min: ", gt.min())
                print()
                show_results(output_dir, img, depth, gt, file_name)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

def show_results(output_dir, img, depth, gt, filename):
    fig = plt.figure(figsize=(12, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Input Image', fontsize=14)
    ax1.axis('off')
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(depth, cmap="rainbow", aspect='equal')
    ax2.set_title('Depth Prediction Map', fontsize=14)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[2, 0])
    error_map = np.where(gt == 0., 0., np.abs(depth - gt))
    rainbow_cmap = plt.cm.rainbow(np.linspace(0, 1, 256))
    custom_colors = np.vstack(([0, 0, 0, 1], rainbow_cmap))
    custom_cmap = ListedColormap(custom_colors)
    im = ax3.imshow(error_map, cmap=custom_cmap, aspect='equal')
    ax3.set_title('Depth Error Map', fontsize=14)
    ax3.axis('off')

    fig.colorbar(im, ax=ax3, orientation="horizontal", pad=0.1)

    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axis('off')
    
    # Extract non-zero values for metric computation
    valid_mask = gt != 0
    depth_valid = depth[valid_mask]
    gt_valid = gt[valid_mask]
    
    metrics = [
        ['RMSE', 'SI-Log', 'Squared Rel', 'Abs Rel'],
        [RMSE(depth_valid, gt_valid), 
         SILog(depth_valid, gt_valid), 
         sqErrorRel(depth_valid, gt_valid), 
         absErrorRel(depth_valid, gt_valid)]
    ]
    table = ax4.table(cellText=metrics, cellLoc='center', loc='center', colWidths=[0.2] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    plt.suptitle(f"{''.join(filename.split('_sync_image_'))}", fontsize=16)
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.close(fig)

def SILog(pred, gt):
    """
    Scale-Invariant Logarithmic Error
    pred: predicted depth
    gt: ground truth depth
    """
    d = np.log(pred) - np.log(gt)
    return "{:.3f}".format(np.sqrt(np.mean(np.square(d)) - np.square(np.mean(d))))

def sqErrorRel(pred, gt):
    """
    Squared relative error
    pred: predicted depth
    gt: ground truth depth
    """
    return "{:.3f}".format(np.mean(np.square(pred - gt) / gt))

def absErrorRel(pred, gt):
    """
    Absolute relative error
    pred: predicted depth
    gt: ground truth depth
    """
    return "{:.3f}".format(np.mean(np.abs(pred - gt) / gt))

def RMSE(pred, gt):
    """
    Root Mean Squared Error
    pred: predicted depth
    gt: ground truth depth
    """
    return "{:.3f}".format(np.sqrt(np.mean(np.square(pred - gt))))

if __name__ == '__main__':
    main()
