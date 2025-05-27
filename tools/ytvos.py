import os
import numpy as np
from PIL import Image
from pathlib import Path
import glob
from collections import defaultdict
from tqdm import tqdm
import json
import shutil

def get_unique_colors(image_path):
    """Extract unique colors (object IDs) from annotation image, excluding background (0)."""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # For YouTube-VOS, objects are represented by unique pixel values
    # Background is typically 0, so we exclude it
    unique_colors = np.unique(img_array)
    unique_colors = unique_colors[unique_colors != 0]  # Remove background
    
    return set(unique_colors.tolist())

def analyze_video_annotations(video_path):
    """Analyze a single video's annotations to find when all objects appear."""
    # Get all annotation frames for this video
    annotation_files = sorted(glob.glob(os.path.join(video_path, '*.png')))
    
    if not annotation_files:
        return None
    
    # Track all unique objects that appear in the entire video
    all_objects = set()
    frame_objects = []
    
    # First pass: collect all objects and per-frame objects
    for frame_path in annotation_files:
        objects_in_frame = get_unique_colors(frame_path)
        frame_objects.append(objects_in_frame)
        all_objects.update(objects_in_frame)
    
    if not all_objects:
        return None
    
    # Second pass: find first frame where all objects are present
    first_complete_frame = None
    for idx, objects in enumerate(frame_objects):
        if objects == all_objects:
            first_complete_frame = idx
            break
    
    return {
        'total_frames': len(annotation_files),
        'total_objects': len(all_objects),
        'first_complete_frame': first_complete_frame,
        'frames_to_skip': first_complete_frame if first_complete_frame is not None else len(annotation_files),
        'all_objects_present': first_complete_frame is not None,
        'object_appearances': [(idx, len(objs)) for idx, objs in enumerate(frame_objects[:10])]  # First 10 frames for debugging
    }

def filter_video_frames(video_id, first_complete_frame, annotations_path, jpeg_path):
    """Remove frames before first_complete_frame and renumber remaining frames."""
    ann_video_path = os.path.join(annotations_path, video_id)
    jpeg_video_path = os.path.join(jpeg_path, video_id)
    
    # Get all annotation frames
    ann_files = sorted(glob.glob(os.path.join(ann_video_path, '*.png')))
    
    # Get all JPEG frames
    jpeg_files = sorted(glob.glob(os.path.join(jpeg_video_path, '*.jpg')))
    if not jpeg_files:  # Try .jpeg extension
        jpeg_files = sorted(glob.glob(os.path.join(jpeg_video_path, '*.jpeg')))
    
    if len(ann_files) != len(jpeg_files):
        print(f"  WARNING: Mismatch in frame count for {video_id}: {len(ann_files)} annotations vs {len(jpeg_files)} jpegs")
        return False
    
    # Delete frames before first_complete_frame
    frames_deleted = 0
    for i in range(first_complete_frame):
        if i < len(ann_files):
            os.remove(ann_files[i])
            frames_deleted += 1
        if i < len(jpeg_files):
            os.remove(jpeg_files[i])
    
    # Rename remaining frames to start from 00000
    remaining_ann_files = sorted(glob.glob(os.path.join(ann_video_path, '*.png')))
    remaining_jpeg_files = sorted(glob.glob(os.path.join(jpeg_video_path, '*.jpg')))
    if not remaining_jpeg_files:
        remaining_jpeg_files = sorted(glob.glob(os.path.join(jpeg_video_path, '*.jpeg')))
    
    # Determine file extension for JPEGs
    jpeg_ext = '.jpg' if remaining_jpeg_files and remaining_jpeg_files[0].endswith('.jpg') else '.jpeg'
    
    # Rename files
    for idx, (ann_file, jpeg_file) in enumerate(zip(remaining_ann_files, remaining_jpeg_files)):
        new_name = f"{idx:05d}"
        new_ann_path = os.path.join(ann_video_path, f"{new_name}.png")
        new_jpeg_path = os.path.join(jpeg_video_path, f"{new_name}{jpeg_ext}")
        
        os.rename(ann_file, new_ann_path)
        os.rename(jpeg_file, new_jpeg_path)
    
    return True

def analyze_dataset(annotations_parent_path, apply_filter=False):
    """Analyze entire YouTube-VOS dataset."""
    annotations_path = os.path.join(annotations_parent_path, 'Annotations')
    jpeg_path = os.path.join(annotations_parent_path, 'JPEGImages')
    
    if not os.path.exists(annotations_path):
        print(f"Annotations path not found: {annotations_path}")
        print("Expected structure: annotations_parent_path/Annotations/video_id/*.png")
        return None, None
    
    if apply_filter and not os.path.exists(jpeg_path):
        print(f"JPEGImages path not found: {jpeg_path}")
        print("Cannot apply filter without JPEGImages folder")
        return None, None
    
    video_dirs = [d for d in os.listdir(annotations_path) if os.path.isdir(os.path.join(annotations_path, d))]
    
    print(f"Found {len(video_dirs)} videos in Annotations folder")
    if apply_filter:
        print("FILTER MODE: Will modify files in-place!")
    
    results = []
    videos_incomplete = []
    videos_filtered = []
    videos_removed = []
    
    for video_id in tqdm(video_dirs, desc="Analyzing videos" if not apply_filter else "Filtering videos"):
        video_path = os.path.join(annotations_path, video_id)
        result = analyze_video_annotations(video_path)
        
        if result:
            result['video_id'] = video_id
            results.append(result)
            
            if not result['all_objects_present']:
                videos_incomplete.append(video_id)
                
                if apply_filter:
                    # Remove entire video if objects never appear together
                    ann_video_path = os.path.join(annotations_path, video_id)
                    jpeg_video_path = os.path.join(jpeg_path, video_id)
                    
                    shutil.rmtree(ann_video_path, ignore_errors=True)
                    shutil.rmtree(jpeg_video_path, ignore_errors=True)
                    videos_removed.append(video_id)
            
            elif apply_filter and result['first_complete_frame'] > 0:
                # Filter frames for this video
                success = filter_video_frames(video_id, result['first_complete_frame'], 
                                            annotations_path, jpeg_path)
                if success:
                    videos_filtered.append((video_id, result['first_complete_frame']))
    
    if apply_filter:
        print(f"\nFiltering complete:")
        print(f"  Videos filtered: {len(videos_filtered)}")
        print(f"  Videos removed entirely: {len(videos_removed)}")
        print(f"  Total frames removed: {sum(f[1] for f in videos_filtered)}")
    
    return results, videos_incomplete

def print_statistics(results, videos_incomplete):
    """Print comprehensive statistics about frame skipping."""
    if not results:
        print("No results to analyze!")
        return
    
    # Calculate statistics
    frames_to_skip = [r['frames_to_skip'] for r in results if r['all_objects_present']]
    total_frames = [r['total_frames'] for r in results]
    num_objects = [r['total_objects'] for r in results]
    
    print("\n" + "="*60)
    print("YouTube-VOS Frame Skip Statistics")
    print("="*60)
    
    print(f"\nTotal videos analyzed: {len(results)}")
    print(f"Videos where all objects appear together: {len(frames_to_skip)}")
    print(f"Videos where objects NEVER all appear together: {len(videos_incomplete)} ({len(videos_incomplete)/len(results)*100:.1f}%)")
    
    if frames_to_skip:
        print(f"\nFor videos where all objects appear together:")
        print(f"  Average frames to skip: {np.mean(frames_to_skip):.2f}")
        print(f"  Median frames to skip: {np.median(frames_to_skip):.0f}")
        print(f"  Max frames to skip: {max(frames_to_skip)}")
        print(f"  Min frames to skip: {min(frames_to_skip)}")
        print(f"  Std dev: {np.std(frames_to_skip):.2f}")
        
        # Percentiles
        print(f"\nPercentiles of frames to skip:")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"  {p}th percentile: {np.percentile(frames_to_skip, p):.0f}")
        
        # Percentage of frames lost
        frames_lost_pct = [(r['frames_to_skip'] / r['total_frames'] * 100) 
                          for r in results if r['all_objects_present']]
        print(f"\nPercentage of frames lost (skipped):")
        print(f"  Average: {np.mean(frames_lost_pct):.1f}%")
        print(f"  Median: {np.median(frames_lost_pct):.1f}%")
        print(f"  Max: {max(frames_lost_pct):.1f}%")
    
    print(f"\nVideo statistics:")
    print(f"  Total frames:")
    print(f"    Average: {np.mean(total_frames):.1f}")
    print(f"    Min: {min(total_frames)}")
    print(f"    Max: {max(total_frames)}")
    print(f"    Median: {np.median(total_frames):.0f}")
    
    print(f"\n  Objects per video:")
    print(f"    Average: {np.mean(num_objects):.1f}")
    print(f"    Min: {min(num_objects)}")
    print(f"    Max: {max(num_objects)}")
    print(f"    Median: {np.median(num_objects):.0f}")
    
    # Object distribution
    print(f"\n  Object count distribution:")
    unique_obj_counts = sorted(set(num_objects))
    for obj_count in unique_obj_counts:
        count = num_objects.count(obj_count)
        pct = count / len(num_objects) * 100
        print(f"    {obj_count} objects: {count:4d} videos ({pct:5.1f}%)")
    
    # Distribution of frames to skip
    if frames_to_skip:
        print("\nDistribution of frames to skip:")
        max_skip = max(frames_to_skip)
        # Create bins that make sense for the data range
        base_bins = [0, 5, 10, 20, 30, 50, 100, 200]
        bins = [b for b in base_bins if b <= max_skip] + [max_skip + 1]
        
        hist, _ = np.histogram(frames_to_skip, bins=bins)
        for i in range(len(bins)-1):
            count = hist[i]
            pct = count / len(frames_to_skip) * 100
            if bins[i+1] - 1 == bins[i]:
                print(f"  {bins[i]:3d} frames: {count:4d} videos ({pct:5.1f}%)")
            else:
                print(f"  {bins[i]:3d} - {bins[i+1]-1:3d} frames: {count:4d} videos ({pct:5.1f}%)")
    
    if videos_incomplete:
        print(f"\nVideos where objects never appear together:")
        incomplete_details = [(r['video_id'], r['total_objects'], r['total_frames']) 
                             for r in results if not r['all_objects_present']]
        for i, (vid, objs, frames) in enumerate(incomplete_details[:10]):
            print(f"  - {vid}: {objs} objects, {frames} frames")
        if len(videos_incomplete) > 10:
            print(f"  ... and {len(videos_incomplete) - 10} more")
    
    # Additional interesting statistics
    print(f"\n" + "="*60)
    print("Additional Analysis")
    print("="*60)
    
    # Find videos with most objects
    sorted_by_objects = sorted(results, key=lambda x: x['total_objects'], reverse=True)
    print(f"\nVideos with most objects:")
    for i in range(min(5, len(sorted_by_objects))):
        r = sorted_by_objects[i]
        print(f"  {r['video_id']}: {r['total_objects']} objects, {r['total_frames']} frames")
    
    # Find videos requiring most frame skips
    if frames_to_skip:
        sorted_by_skip = sorted([r for r in results if r['all_objects_present']], 
                               key=lambda x: x['frames_to_skip'], reverse=True)
        print(f"\nVideos requiring most frame skips:")
        for i in range(min(5, len(sorted_by_skip))):
            r = sorted_by_skip[i]
            skip_pct = r['frames_to_skip'] / r['total_frames'] * 100
            print(f"  {r['video_id']}: skip {r['frames_to_skip']} frames ({skip_pct:.1f}%), {r['total_objects']} objects")
    
    # Correlation analysis
    if len(results) > 10 and frames_to_skip:
        objects_complete = [r['total_objects'] for r in results if r['all_objects_present']]
        if len(objects_complete) > 0:
            correlation = np.corrcoef(objects_complete, frames_to_skip)[0, 1]
            print(f"\nCorrelation between number of objects and frames to skip: {correlation:.3f}")
    
    # Summary for training implications
    print(f"\n" + "="*60)
    print("Training Implications Summary")
    print("="*60)
    
    if frames_to_skip:
        total_original_frames = sum([r['total_frames'] for r in results if r['all_objects_present']])
        total_usable_frames = sum([r['total_frames'] - r['frames_to_skip'] 
                                  for r in results if r['all_objects_present']])
        overall_loss_pct = (1 - total_usable_frames / total_original_frames) * 100
        
        print(f"\nIf requiring all objects to be present:")
        print(f"  - Completely unusable videos: {len(videos_incomplete)} ({len(videos_incomplete)/len(results)*100:.1f}%)")
        print(f"  - Usable videos: {len(frames_to_skip)}")
        print(f"  - Total frames in usable videos: {total_original_frames:,}")
        print(f"  - Frames after skipping: {total_usable_frames:,}")
        print(f"  - Overall frame loss: {overall_loss_pct:.1f}%")
        print(f"  - Median skip per video: {int(np.median(frames_to_skip))} frames")
    else:
        print("\nNo videos found where all objects appear together!")

def save_detailed_results(results, output_path='youtube_vos_analysis.json'):
    """Save detailed results to JSON for further analysis."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze YouTube-VOS annotations for frame skip statistics')
    parser.add_argument('annotations_parent', type=str, 
                        help='Path to parent directory containing Annotations folder')
    parser.add_argument('--output', type=str, default='youtube_vos_analysis.json',
                        help='Output JSON file path (default: youtube_vos_analysis.json)')
    parser.add_argument('--filter', action='store_true',
                        help='Apply filtering to remove frames before all objects appear (modifies files in-place!)')
    
    args = parser.parse_args()
    
    print("YouTube-VOS 2019 Frame Skip Analysis")
    print("This script analyzes how many frames need to be skipped")
    print("before all objects appear together in each video.\n")
    
    if args.filter:
        print("WARNING: Filter mode is enabled!")
        print("This will DELETE frames from your dataset in-place.")
        print("Make sure you have a backup of your data!")
        response = input("\nContinue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    # Check if path exists
    if not os.path.exists(args.annotations_parent):
        print(f"ERROR: Path does not exist: {args.annotations_parent}")
        return
    
    # Analyze dataset
    print(f"\nAnalyzing dataset at: {args.annotations_parent}")
    results, videos_incomplete = analyze_dataset(args.annotations_parent, apply_filter=args.filter)
    
    if results and not args.filter:
        print_statistics(results, videos_incomplete)
        save_detailed_results(results, args.output)
    elif not results:
        print("No results found. Please check your dataset path.")

if __name__ == "__main__":
    main()