import argparse
import ast
import csv
import json
from pathlib import Path
import zipfile

import cv2
from tqdm import tqdm


SPLITS = ("train", "valid", "test")


def load_splits(splits_path):
    with open(splits_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    split_map = {}
    for split_key, out_name in (("train_set", "train"), ("val_set", "valid"), ("test_set", "test")):
        for video_name in data.get(split_key, []):
            split_map[video_name] = out_name
    return split_map


def load_categories(species_csv):
    categories = []
    with open(species_csv, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for line in lines:
        if line.lower().startswith('id'):
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                category_id = int(parts[0])
                name = parts[1].strip()
                categories.append({'id': category_id, 'name': name, 'supercategory': 'bird'})
            except ValueError:
                continue
    return categories


def ensure_dirs(base_out):
    for split in SPLITS:
        (base_out / split / 'images').mkdir(parents=True, exist_ok=True)


def make_coco_dict(categories):
    return {
        split: {'images': [], 'annotations': [], 'categories': list(categories)}
        for split in SPLITS
    }


def init_stats():
    return {
        'rows_read': 0,
        'rows_bad_columns': 0,
        'rows_species_parse_fail': 0,
        'rows_frame_parse_fail': 0,
        'rows_bbox_parse_fail': 0,
        'rows_empty_bbox': 0,
        'rows_with_species': 0,
        'rows_valid_bbox_field': 0,
        'rows_positive': 0,
        'rows_after_downsample': 0,
        'bbox_candidates': 0,
        'invalid_bbox_count': 0,
        'invalid_species_count': 0,
        'missing_video_count': 0,
        'frame_decode_fail_count': 0,
        'fallback_split_frames': 0,
        'videos_opened': 0,
        'frames_indexed': 0,
        'frames_extracted': 0,
        'images_written': 0,
        'images_reused': 0,
        'annotations_written': 0,
        'annotations_dropped_no_image': 0,
    }


def resolve_split(video_name, split_map, stats):
    split = split_map.get(video_name)
    if split is None:
        stats['fallback_split_frames'] += 1
        return 'train'
    return split


def parse_index(args, split_map, valid_category_ids, stats):
    # {video_name: {frame_idx: {'split': split, 'boxes': [(species_id, xmin, ymin, xmax, ymax), ...]}}}
    frame_index = {}

    with open(args.bounding_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in tqdm(reader, desc='index rows'):
            stats['rows_read'] += 1
            if len(row) < 5:
                stats['rows_bad_columns'] += 1
                continue

            try:
                species_id = int(row[0])
                stats['rows_with_species'] += 1
            except Exception:
                stats['rows_species_parse_fail'] += 1
                continue

            if species_id not in valid_category_ids:
                stats['invalid_species_count'] += 1
                continue

            video_name = row[2]
            try:
                frame_idx = int(row[3])
            except Exception:
                stats['rows_frame_parse_fail'] += 1
                continue

            if frame_idx % args.downsample != 0:
                continue
            stats['rows_after_downsample'] += 1

            bbox_field = row[4]
            if bbox_field.strip() in ['', '[]']:
                stats['rows_empty_bbox'] += 1
                continue

            try:
                bboxes = ast.literal_eval(bbox_field)
            except Exception:
                stats['rows_bbox_parse_fail'] += 1
                continue
            stats['rows_valid_bbox_field'] += 1

            if not bboxes:
                stats['rows_empty_bbox'] += 1
                continue
            stats['rows_positive'] += 1

            split = resolve_split(video_name, split_map, stats)
            video_entry = frame_index.setdefault(video_name, {})
            frame_entry = video_entry.setdefault(frame_idx, {'split': split, 'boxes': []})

            for bb in bboxes:
                stats['bbox_candidates'] += 1
                try:
                    xmin = float(bb[0])
                    ymin = float(bb[1])
                    xmax = float(bb[2])
                    ymax = float(bb[3])
                except Exception:
                    stats['invalid_bbox_count'] += 1
                    continue

                if xmax <= xmin or ymax <= ymin:
                    stats['invalid_bbox_count'] += 1
                    continue

                frame_entry['boxes'].append((species_id, xmin, ymin, xmax, ymax))

    for frames in frame_index.values():
        for frame_entry in frames.values():
            if frame_entry['boxes']:
                stats['frames_indexed'] += 1

    return frame_index


def extract_images_and_build_coco(args, frame_index, categories, stats):
    out_dir = Path(args.output_dir)
    ensure_dirs(out_dir)
    coco = make_coco_dict(categories)
    image_id = 1
    ann_id = 1

    for video_name in tqdm(sorted(frame_index.keys()), desc='videos'):
        video_path = Path(args.videos_dir) / f'{video_name}.mp4'
        if not video_path.exists():
            stats['missing_video_count'] += 1
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            stats['missing_video_count'] += 1
            continue
        stats['videos_opened'] += 1

        frame_items = sorted(frame_index[video_name].items(), key=lambda item: item[0])
        for frame_idx, frame_entry in frame_items:
            if not frame_entry['boxes']:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = cap.read()
            if not ok or img is None:
                stats['frame_decode_fail_count'] += 1
                stats['annotations_dropped_no_image'] += len(frame_entry['boxes'])
                continue

            split = frame_entry['split']
            height, width = img.shape[:2]
            img_name = f'{video_name}_{frame_idx:06d}.{args.image_ext}'
            img_out = out_dir / split / 'images' / img_name

            if args.overwrite or not img_out.exists():
                write_ok = cv2.imwrite(str(img_out), img)
                if write_ok:
                    stats['images_written'] += 1
                else:
                    stats['frame_decode_fail_count'] += 1
                    stats['annotations_dropped_no_image'] += len(frame_entry['boxes'])
                    continue
            else:
                stats['images_reused'] += 1

            coco[split]['images'].append({
                'id': image_id,
                'file_name': img_name,
                'width': width,
                'height': height,
            })

            for species_id, xmin, ymin, xmax, ymax in frame_entry['boxes']:
                box_w = xmax - xmin
                box_h = ymax - ymin
                ann = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': species_id,
                    'bbox': [round(xmin, 2), round(ymin, 2), round(box_w, 2), round(box_h, 2)],
                    'area': round(box_w * box_h, 2),
                    'iscrowd': 0,
                }
                coco[split]['annotations'].append(ann)
                ann_id += 1
                stats['annotations_written'] += 1

            image_id += 1
            stats['frames_extracted'] += 1

        cap.release()

    return coco


def write_coco_files(out_dir, coco):
    for split_name in SPLITS:
        out_file = out_dir / split_name / '_annotations.coco.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(coco[split_name], f, ensure_ascii=False, indent=2)


def validate_coco(coco):
    for split in SPLITS:
        image_ids = {img['id'] for img in coco[split]['images']}
        category_ids = {cat['id'] for cat in coco[split]['categories']}
        file_names = [img['file_name'] for img in coco[split]['images']]

        if len(file_names) != len(set(file_names)):
            raise ValueError(f"Duplicate image file_name detected in split '{split}'")

        for ann in coco[split]['annotations']:
            if ann['image_id'] not in image_ids:
                raise ValueError(
                    f"Annotation {ann['id']} in split '{split}' references missing image_id {ann['image_id']}"
                )
            if ann['category_id'] not in category_ids:
                raise ValueError(
                    f"Annotation {ann['id']} in split '{split}' references unknown category_id {ann['category_id']}"
                )


def write_zip_archives(out_dir):
    zip_sizes = {}
    for split in SPLITS:
        split_dir = out_dir / split
        zip_path = out_dir / f'{split}.zip'
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for p in split_dir.rglob('*'):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(split_dir))
        zip_sizes[split] = zip_path.stat().st_size
    return zip_sizes


def print_summary(stats, coco, zip_sizes):
    print('\n=== Dataset Build Summary ===')
    print(f"rows_read: {stats['rows_read']}")
    print(f"rows_bad_columns: {stats['rows_bad_columns']}")
    print(f"rows_species_parse_fail: {stats['rows_species_parse_fail']}")
    print(f"rows_frame_parse_fail: {stats['rows_frame_parse_fail']}")
    print(f"rows_bbox_parse_fail: {stats['rows_bbox_parse_fail']}")
    print(f"rows_empty_bbox: {stats['rows_empty_bbox']}")
    print(f"rows_with_species: {stats['rows_with_species']}")
    print(f"rows_after_downsample: {stats['rows_after_downsample']}")
    print(f"rows_valid_bbox_field: {stats['rows_valid_bbox_field']}")
    print(f"rows_positive: {stats['rows_positive']}")
    print(f"bbox_candidates: {stats['bbox_candidates']}")
    print(f"invalid_bbox_count: {stats['invalid_bbox_count']}")
    print(f"invalid_species_count: {stats['invalid_species_count']}")
    print(f"missing_video_count: {stats['missing_video_count']}")
    print(f"frame_decode_fail_count: {stats['frame_decode_fail_count']}")
    print(f"fallback_split_frames: {stats['fallback_split_frames']}")
    print(f"videos_opened: {stats['videos_opened']}")
    print(f"frames_indexed: {stats['frames_indexed']}")
    print(f"frames_extracted: {stats['frames_extracted']}")
    print(f"images_written: {stats['images_written']}")
    print(f"images_reused: {stats['images_reused']}")
    print(f"annotations_written: {stats['annotations_written']}")
    print(f"annotations_dropped_no_image: {stats['annotations_dropped_no_image']}")

    print('\n--- Per Split ---')
    for split in SPLITS:
        print(
            f"{split}: images={len(coco[split]['images'])}, "
            f"annotations={len(coco[split]['annotations'])}, "
            f"categories={len(coco[split]['categories'])}"
        )

    if zip_sizes:
        print('\n--- Zip Sizes (bytes) ---')
        for split in SPLITS:
            print(f"{split}.zip: {zip_sizes.get(split, 0)}")


def main(args):
    if args.downsample <= 0:
        raise ValueError('--downsample must be > 0')

    out_dir = Path(args.output_dir)
    split_map = load_splits(args.splits_json)
    categories = load_categories(args.species_csv)
    valid_category_ids = {cat['id'] for cat in categories}

    stats = init_stats()
    frame_index = parse_index(args, split_map, valid_category_ids, stats)
    coco = extract_images_and_build_coco(args, frame_index, categories, stats)
    validate_coco(coco)
    write_coco_files(out_dir, coco)

    zip_sizes = {}
    if args.zip:
        zip_sizes = write_zip_archives(out_dir)

    print_summary(stats, coco, zip_sizes)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bounding_csv', default='bounding_boxes.csv')
    p.add_argument('--videos_dir', default='videos')
    p.add_argument('--splits_json', default='splits.json')
    p.add_argument('--species_csv', default='species_ID.csv')
    p.add_argument('--output_dir', default='rf_detr_dataset')
    p.add_argument('--downsample', type=int, default=10, help='keep 1 every N frames')
    p.add_argument('--zip', action='store_true', help='create zip files for each split')
    p.add_argument('--image_ext', default='jpg', help='output image extension')
    p.add_argument('--overwrite', action='store_true', help='overwrite existing image files')
    args = p.parse_args()
    main(args)
