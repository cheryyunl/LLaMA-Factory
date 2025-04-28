import os
import math
import random
import pathlib
import lz4.frame
import argparse
import threading
import ujson as json
import multiprocessing as mp
import numpy as np
import base64
from io import BytesIO
from glob import glob
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict

def clean_object_name(text, dataset_type=None):
    """
    Clean and normalize object names based on dataset conventions:
    - 3D-FUTURE: Uses slashes (/) to separate different names
    - ShapeNet: Uses commas (,) to separate synonyms
    - ABO: Keep full product names
    - Objaverse: Keep the original object name
    - Default: Handles both formats
    """
    # If text starts with a comma, skip this entry
    if text.startswith(','):
        return None
    text = text.lower()
    # Replace escaped slashes with normal slashes
    text = text.replace('\\/', '/')
    
    # Process names based on dataset type
    if dataset_type == "3D-FUTURE":
        # Only split by slash for 3D-FUTURE
        names = [name.strip() for name in text.split('/')]
    elif dataset_type == "ShapeNet":
        # Only split by comma for ShapeNet
        names = [name.strip() for name in text.split(',')]
    elif dataset_type == "ABO":
        # For ABO, keep the full product name as is
        # Just clean up any weird characters and extra whitespace
        text = text.replace('\\u00a0', ' ').replace('\\u2013', '-').replace('\\u2014', '-')
        text = text.replace('\\"', '"').replace('\\\'', "'")
        names = [text.strip()]
    elif dataset_type == "Objaverse":
        # For Objaverse, keep the original name
        names = [text.strip()]
    else:
        # Default: handle both formats by splitting first by slash, then by comma
        slash_parts = [part.strip() for part in text.split('/')]
        names = []
        for part in slash_parts:
            comma_parts = [name.strip() for name in part.split(',')]
            names.extend(comma_parts)
    
    # Filter out empty names
    names = [name for name in names if name]
    
    return names if names else None

def points_to_base64(points: np.ndarray) -> str:
    """Convert numpy array to base64 encoded string"""
    points = points.astype(np.float32)
    return base64.b64encode(points.tobytes()).decode('utf-8')

def process_npy_file(file_path, queue, dataset_type=None) -> None:
    """Process a single NPY file and extract point cloud data with object names"""
    try:
        # Load npy file with allow_pickle=True
        data = np.load(file_path, allow_pickle=True)
        
        # Get the data dictionary
        data_dict = data.item()
        
        # For Objaverse, apply additional filtering
        if dataset_type == "Objaverse":
            # Import the filtering function
            from clean_objaverse import apply_filtering
            
            # Apply filtering - skip this file if it fails the filter
            if not apply_filtering(data_dict):
                print(f"Skipping file {file_path} - failed quality filter")
                queue.put("END_OF_FILE")
                return
        
        # Extract xyz coordinates and rgb colors
        xyz = data_dict['xyz']  # shape: (10000, 3)
        rgb = data_dict['rgb']  # shape: (10000, 3)
        
        # Get object identifiers from the first text label
        raw_text = data_dict['text'][0]
        
        # Apply cleaning function to get normalized object names (may return multiple)
        object_names = clean_object_name(raw_text, dataset_type=dataset_type)
        
        # Skip this file if we couldn't determine proper object names
        if object_names is None:
            print(f"Skipping file {file_path} - could not determine object name from '{raw_text}'")
            queue.put("END_OF_FILE")
            return
            
        # Create the 6D point cloud and encode it once
        point_cloud = np.concatenate([xyz, rgb], axis=1)  # shape: (10000, 6)
        encoded_point_cloud = points_to_base64(point_cloud)
        file_id = os.path.basename(file_path).replace('.npy', '')
        
        # Create a separate data entry for each object name
        for object_name in object_names:
            point_cloud_content = {
                "type": "image_url",
                "image_url": {
                    "url": encoded_point_cloud
                }
            }
            
            content = [
                point_cloud_content,
                {"type": "text", "text": object_name}
            ]
            
            cur_stat_counter = defaultdict(int)
            cur_stat_counter["n_objects_processed"] = 1
            
            # Put processed data into the queue
            queue.put(({
                "role": "assistant",
                "content": content
            }, cur_stat_counter))
            
            # Update object category counter
            queue.put(("OBJECT_NAME", (object_name, file_id)))
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    # Indicate file processing is complete
    queue.put("END_OF_FILE")

def get_data_instances(file_paths, queue, dataset_type=None) -> None:
    """Process multiple NPY files"""
    print(f"Process {mp.current_process().name} started for processing {len(file_paths)} files.")
    
    # Process each npy file
    for file_path in file_paths:
        process_npy_file(file_path, queue, dataset_type)
    
    print(f"Process {mp.current_process().name} finished.")

def main():
    parser = argparse.ArgumentParser(description='Process 3D point cloud data')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing .npy files')
    parser.add_argument('--output-filepath', type=str, required=True,
                       help='Output file pattern (e.g., data/processed/3dfuture/objects.shard_{shard_id:03d}.jsonl.lz4)')
    parser.add_argument('--n-workers', type=int, default=50)
    parser.add_argument('--n-output-shards', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-type', type=str, default=None, 
                       choices=['3D-FUTURE', 'ShapeNet', 'ABO', 'Objaverse', None],
                       help='Type of dataset being processed (affects name splitting)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    pathlib.Path(args.output_filepath).parent.mkdir(parents=True, exist_ok=True)

    # Get all npy file paths - recursively search subdirectories
    file_paths = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.npy'):
                file_paths.append(os.path.join(root, file))
    
    file_paths = sorted(file_paths)
    n_files_per_shard = math.ceil(len(file_paths) / args.n_output_shards)
    
    print(f"Processing {len(file_paths)} .npy files")
    print(f"Writing to {args.output_filepath} with {args.n_output_shards} shards ({n_files_per_shard} files per shard)")

    # Statistics counters
    stats_counter = defaultdict(int)
    object_category_counter = defaultdict(int)
    object_file_mapping = defaultdict(list)  # Store file IDs for each object category
    stats_counter["n_instances"] = 0
    
    # Progress bar
    pbar = tqdm(total=len(file_paths))
    for shard_id in range(args.n_output_shards):
        start = shard_id * n_files_per_shard
        end = min((shard_id + 1) * n_files_per_shard, len(file_paths))
        cur_shard_filepaths = file_paths[start:end]

        queue = mp.Queue(maxsize=args.n_workers * 1024)
        processes = []

        # Create worker processes
        for i in range(args.n_workers):
            cur_worker_paths = cur_shard_filepaths[i::args.n_workers]
            process = mp.Process(target=get_data_instances, args=(cur_worker_paths, queue, args.dataset_type))
            process.start()
            processes.append(process)

        # Writer thread
        def writer(queue, fout):
            print("Starting writer thread")
            while True:
                content = queue.get()
                if content is None:
                    break
                elif content == "END_OF_FILE":
                    pbar.update(1)
                    continue
                elif isinstance(content, tuple) and content[0] == "OBJECT_NAME":
                    obj_name, file_id = content[1]
                    object_category_counter[obj_name] += 1
                    object_file_mapping[obj_name].append(file_id)
                    continue

                instance, cur_stats_counter = content
                fout.write(json.dumps(instance) + "\n")
                for k, v in cur_stats_counter.items():
                    stats_counter[k] += v
                pbar.set_postfix(stats_counter)
                stats_counter["n_instances"] += 1

        with lz4.frame.open(args.output_filepath.format(shard_id=shard_id), "wt") as fout:
            writer_thread = threading.Thread(target=writer, args=(queue, fout))
            writer_thread.start()

            for process in processes:
                process.join()
                
            queue.put(None)
            writer_thread.join()

    pbar.close()

    # Output final object category statistics
    print("\nObject category statistics:")
    sorted_categories = sorted(object_category_counter.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        print(f"{category}: {count}")

    # Save all statistics
    stats_counter["object_categories"] = dict(object_category_counter)
    stats_counter["object_file_mapping"] = object_file_mapping
    with open(args.output_filepath.replace(".shard_{shard_id:03d}.jsonl.lz4", ".stats.json"), "w") as f:
        json.dump(stats_counter, f, indent=4)

if __name__ == "__main__":
    main()
