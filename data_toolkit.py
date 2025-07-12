


# upload_data.py
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import hashlib
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import argparse
import sys
from insightface_func.face_detect_crop_single import Face_detect_crop

class DatasetUpdater:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(exist_ok=True)
        
        # Initialize face detection
        self.face_app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.face_app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640), mode='None')
        
        # Statistics
        self.stats = {
            'total_people': 0,
            'total_images': 0,
            'duplicates_removed': 0,
            'invalid_images': 0,
            'last_updated': None
        }
        
    def add_person(self, person_name, image_folder):
        """Add a new person to the dataset"""
        print(f"Adding person: {person_name}")
        
        # Create person directory
        person_dir = self.dataset_path / person_name
        person_dir.mkdir(exist_ok=True)
        
        # Process all images in the folder
        valid_images = 0
        image_files = list(Path(image_folder).glob('*.jpg')) + \
                     list(Path(image_folder).glob('*.jpeg')) + \
                     list(Path(image_folder).glob('*.png'))
        
        for img_path in image_files:
            if self.process_and_add_image(img_path, person_dir):
                valid_images += 1
        
        print(f"Added {valid_images} valid images for {person_name}")
        return valid_images
    
    def process_and_add_image(self, image_path, person_dir):
        """Process and add a single image"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Cannot read image: {image_path}")
                return False
            
            # Check if face is detected
            try:
                faces, _ = self.face_app.get(img, 224)
                if len(faces) == 0:
                    print(f"No face detected in: {image_path}")
                    return False
                elif len(faces) > 1:
                    print(f"Multiple faces detected in: {image_path}")
                    return False
            except:
                print(f"Face detection failed for: {image_path}")
                return False
            
            # Check for duplicates
            if self.is_duplicate(image_path, person_dir):
                print(f"Duplicate image: {image_path}")
                self.stats['duplicates_removed'] += 1
                return False
            
            # Generate new filename
            img_hash = self.get_image_hash(image_path)
            new_filename = f"{img_hash}.jpg"
            new_path = person_dir / new_filename
            
            # Copy image
            shutil.copy2(image_path, new_path)
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            self.stats['invalid_images'] += 1
            return False
    
    def is_duplicate(self, new_image_path, person_dir):
        """Check if image is duplicate using hash comparison"""
        new_hash = self.get_image_hash(new_image_path)
        
        # Check against existing images
        for existing_img in person_dir.glob('*.jpg'):
            if existing_img.stem == new_hash:
                return True
        
        return False
    
    def get_image_hash(self, image_path):
        """Generate hash for image"""
        with open(image_path, 'rb') as f:
            img_data = f.read()
            return hashlib.md5(img_data).hexdigest()
    
    def remove_duplicates(self):
        """Remove duplicate images from entire dataset"""
        print("Removing duplicates from dataset...")
        
        for person_dir in self.dataset_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            print(f"Processing: {person_dir.name}")
            
            # Group images by hash
            hash_groups = defaultdict(list)
            for img_path in person_dir.glob('*.jpg'):
                img_hash = self.get_image_hash(img_path)
                hash_groups[img_hash].append(img_path)
            
            # Remove duplicates
            for img_hash, img_list in hash_groups.items():
                if len(img_list) > 1:
                    # Keep the first one, remove others
                    for img_path in img_list[1:]:
                        print(f"Removing duplicate: {img_path}")
                        img_path.unlink()
                        self.stats['duplicates_removed'] += 1
    
    def validate_dataset(self):
        """Validate entire dataset"""
        print("Validating dataset...")
        
        total_people = 0
        total_images = 0
        
        for person_dir in self.dataset_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            total_people += 1
            person_images = len(list(person_dir.glob('*.jpg')))
            total_images += person_images
            
            print(f"{person_dir.name}: {person_images} images")
            
            # Check for minimum images per person
            if person_images < 3:
                print(f"Warning: {person_dir.name} has only {person_images} images")
        
        self.stats['total_people'] = total_people
        self.stats['total_images'] = total_images
        self.stats['last_updated'] = datetime.now().isoformat()
        
        print(f"\nDataset Statistics:")
        print(f"Total People: {total_people}")
        print(f"Total Images: {total_images}")
        print(f"Duplicates Removed: {self.stats['duplicates_removed']}")
        print(f"Invalid Images: {self.stats['invalid_images']}")
    
    def merge_datasets(self, source_dataset_path):
        """Merge another dataset into current one"""
        print(f"Merging dataset from: {source_dataset_path}")
        
        source_path = Path(source_dataset_path)
        if not source_path.exists():
            print(f"Source dataset not found: {source_dataset_path}")
            return
        
        for person_dir in source_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            print(f"Merging person: {person_dir.name}")
            self.add_person(person_dir.name, person_dir)
    
    def save_stats(self):
        """Save dataset statistics"""
        stats_file = self.dataset_path / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Statistics saved to: {stats_file}")
    
    def list_people(self):
        """List all people in the dataset"""
        print(f"People in dataset: {self.dataset_path}")
        print("-" * 50)
        
        for person_dir in sorted(self.dataset_path.iterdir()):
            if not person_dir.is_dir():
                continue
                
            image_count = len(list(person_dir.glob('*.jpg')))
            print(f"{person_dir.name}: {image_count} images")
    
    def delete_person(self, person_name):
        """Delete a person from the dataset"""
        person_dir = self.dataset_path / person_name
        if not person_dir.exists():
            print(f"Person '{person_name}' not found in dataset")
            return False
        
        # Ask for confirmation
        response = input(f"Are you sure you want to delete '{person_name}' and all their images? (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled")
            return False
        
        shutil.rmtree(person_dir)
        print(f"Deleted person: {person_name}")
        return True

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Face Swap Dataset Updater",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a new person
  python upload_data.py add --person john_doe --images ./john_images --dataset ./training_data
  
  # Merge two datasets
  python upload_data.py merge --source ./vgg_dataset --dataset ./training_data
  
  # Validate dataset
  python upload_data.py validate --dataset ./training_data
  
  # Remove duplicates
  python upload_data.py clean --dataset ./training_data
  
  # List all people
  python upload_data.py list --dataset ./training_data
  
  # Delete a person
  python upload_data.py delete --person john_doe --dataset ./training_data
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add person command
    add_parser = subparsers.add_parser('add', help='Add a new person to dataset')
    add_parser.add_argument('--person', '-p', type=str, required=True,
                           help='Person name')
    add_parser.add_argument('--images', '-i', type=str, required=True,
                           help='Path to folder containing person images')
    add_parser.add_argument('--dataset', '-d', type=str, default='./training_data',
                           help='Path to dataset directory (default: ./training_data)')
    
    # Merge datasets command
    merge_parser = subparsers.add_parser('merge', help='Merge another dataset')
    merge_parser.add_argument('--source', '-s', type=str, required=True,
                             help='Path to source dataset to merge')
    merge_parser.add_argument('--dataset', '-d', type=str, default='./training_data',
                             help='Path to dataset directory (default: ./training_data)')
    
    # Validate dataset command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--dataset', '-d', type=str, default='./training_data',
                                help='Path to dataset directory (default: ./training_data)')
    
    # Clean dataset command
    clean_parser = subparsers.add_parser('clean', help='Remove duplicates from dataset')
    clean_parser.add_argument('--dataset', '-d', type=str, default='./training_data',
                             help='Path to dataset directory (default: ./training_data)')
    
    # List people command
    list_parser = subparsers.add_parser('list', help='List all people in dataset')
    list_parser.add_argument('--dataset', '-d', type=str, default='./training_data',
                            help='Path to dataset directory (default: ./training_data)')
    
    # Delete person command
    delete_parser = subparsers.add_parser('delete', help='Delete a person from dataset')
    delete_parser.add_argument('--person', '-p', type=str, required=True,
                              help='Person name to delete')
    delete_parser.add_argument('--dataset', '-d', type=str, default='./training_data',
                              help='Path to dataset directory (default: ./training_data)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--dataset', '-d', type=str, default='./training_data',
                             help='Path to dataset directory (default: ./training_data)')
    
    return parser

def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize dataset updater
    try:
        updater = DatasetUpdater(args.dataset)
        print(f"Dataset path: {args.dataset}")
        print("")
    except Exception as e:
        print(f"Error initializing dataset updater: {e}")
        return
    
    # Execute command
    try:
        if args.command == 'add':
            if not Path(args.images).exists():
                print(f"Error: Image folder '{args.images}' not found")
                return
            
            valid_images = updater.add_person(args.person, args.images)
            print(f"\nSummary: Added {valid_images} valid images for {args.person}")
            
        elif args.command == 'merge':
            if not Path(args.source).exists():
                print(f"Error: Source dataset '{args.source}' not found")
                return
            
            updater.merge_datasets(args.source)
            print("\nMerge completed")
            
        elif args.command == 'validate':
            updater.validate_dataset()
            
        elif args.command == 'clean':
            updater.remove_duplicates()
            print("\nDuplicate removal completed")
            
        elif args.command == 'list':
            updater.list_people()
            
        elif args.command == 'delete':
            updater.delete_person(args.person)
            
        elif args.command == 'stats':
            updater.validate_dataset()
            updater.save_stats()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()