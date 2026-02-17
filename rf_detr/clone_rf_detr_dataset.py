import os
import shutil
import random
import argparse


def copy_subset(src_dir, dest_dir, percentage):
    os.makedirs(dest_dir, exist_ok=True)
    # Copy JSON file
    json_file = os.path.join(src_dir, '_annotations.coco.json')
    if os.path.exists(json_file):
        shutil.copy(json_file, dest_dir)
    # Copy subset of images
    images_src = os.path.join(src_dir, 'images')
    images_dest = os.path.join(dest_dir, 'images')
    os.makedirs(images_dest, exist_ok=True)
    images = [f for f in os.listdir(images_src) if os.path.isfile(os.path.join(images_src, f))]
    n = max(1, int(len(images) * percentage / 100))
    sampled_images = random.sample(images, n)
    for img in sampled_images:
        shutil.copy(os.path.join(images_src, img), images_dest)


def main():
    parser = argparse.ArgumentParser(description='Clone rf_detr_dataset with a subset of images.')
    parser.add_argument('--src', type=str, default='rf_detr_dataset', help='Source dataset directory')
    parser.add_argument('--dest', type=str, default='rf_detr_dataset_subset', help='Destination directory for subset')
    parser.add_argument('--percent', type=float, default=10.0, help='Percentage of images to copy from each split')
    args = parser.parse_args()

    for split in ['train', 'valid', 'test']:
        src_split = os.path.join(args.src, split)
        dest_split = os.path.join(args.dest, split)
        if os.path.exists(src_split):
            copy_subset(src_split, dest_split, args.percent)

    print(f"Subset dataset created at {args.dest} with {args.percent}% images from each split.")


if __name__ == '__main__':
    main()
