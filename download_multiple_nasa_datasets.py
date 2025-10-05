"""
Download multiple NASA datasets for shark habitat prediction
Based on successful access patterns and known file structures
"""

import requests
import os
from dotenv import load_dotenv
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import time

# Load environment variables
load_dotenv()

class NASADatasetDownloader:
    """Download multiple NASA datasets for oceanographic analysis"""
    
    def __init__(self):
        """Initialize the downloader"""
        self.token = os.getenv('EARTHDATA_TOKEN')
        self.headers = {'Authorization': f'Bearer {self.token}'}
        
        # Create directories for different datasets
        self.data_dir = "nasa_data"
        self.dataset_dirs = {
            'modis_aqua': os.path.join(self.data_dir, 'modis_aqua'),
            'modis_terra': os.path.join(self.data_dir, 'modis_terra'),
            'seawifs': os.path.join(self.data_dir, 'seawifs'),
            'viirs': os.path.join(self.data_dir, 'viirs'),
            'jason': os.path.join(self.data_dir, 'jason'),
            'avhrr': os.path.join(self.data_dir, 'avhrr')
        }
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        for dir_path in self.dataset_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def download_file(self, url, filename, description):
        """Download a single file with progress tracking"""
        print(f"\n--- Downloading {description} ---")
        print(f"URL: {url}")
        print(f"Save to: {filename}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=600)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                print(f"SUCCESS: Downloaded {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                return True, filename
            else:
                print(f"ERROR: Download failed with status {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                return False, None
                
        except Exception as e:
            print(f"Exception during download: {e}")
            return False, None
    
    def examine_dataset(self, filename, description):
        """Examine a downloaded dataset"""
        print(f"\n--- Examining {description} ---")
        
        try:
            # Try with netCDF4 first to handle groups
            import netCDF4 as nc
            f = nc.Dataset(filename, 'r')
            
            print(f"File: {filename}")
            print(f"Variables: {list(f.variables.keys())}")
            print(f"Dimensions: {dict(f.dimensions)}")
            print(f"Groups: {list(f.groups.keys()) if hasattr(f, 'groups') else 'No groups'}")
            
            # Check for level-3 binned data group
            if 'level-3_binned_data' in f.groups:
                l3_group = f.groups['level-3_binned_data']
                print(f"L3 Variables: {list(l3_group.variables.keys())}")
                print(f"L3 Dimensions: {dict(l3_group.dimensions)}")
                
                # Show data info for key variables
                for var_name in ['sst', 'chlor_a', 'Rrs_443', 'Rrs_488', 'Rrs_555', 'Rrs_667']:
                    if var_name in l3_group.variables:
                        var = l3_group.variables[var_name]
                        print(f"  {var_name}: shape={var.shape}, dtype={var.dtype}")
            
            f.close()
            return True
            
        except Exception as e:
            print(f"Error examining dataset: {e}")
            return False
    
    def download_modis_aqua_datasets(self):
        """Download MODIS-Aqua datasets (already working)"""
        print("\n=== MODIS-AQUA DATASETS ===")
        
        datasets = [
            {
                'name': 'SST',
                'url': 'https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/AQUA_MODIS.20020701_20200731.L3b.MC.SST.nc',
                'filename': os.path.join(self.dataset_dirs['modis_aqua'], 'modis_aqua_sst.nc')
            },
            {
                'name': 'Chlorophyll',
                'url': 'https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/AQUA_MODIS.20020704_20250228.L3b.CU.CHL.nc',
                'filename': os.path.join(self.dataset_dirs['modis_aqua'], 'modis_aqua_chlorophyll.nc')
            }
        ]
        
        downloaded_files = []
        
        for dataset in datasets:
            # Check if file already exists
            if os.path.exists(dataset['filename']):
                print(f"File already exists: {dataset['filename']}")
                downloaded_files.append(dataset['filename'])
                continue
            
            success, filename = self.download_file(
                dataset['url'], 
                dataset['filename'], 
                f"MODIS-Aqua {dataset['name']}"
            )
            
            if success:
                downloaded_files.append(filename)
                self.examine_dataset(filename, f"MODIS-Aqua {dataset['name']}")
            
            time.sleep(2)  # Be respectful to the server
        
        return downloaded_files
    
    def download_modis_terra_datasets(self):
        """Download MODIS-Terra datasets"""
        print("\n=== MODIS-TERRA DATASETS ===")
        
        datasets = [
            {
                'name': 'SST',
                'url': 'https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/TERRA_MODIS.20000301_20200331.L3b.MC.SST.nc',
                'filename': os.path.join(self.dataset_dirs['modis_terra'], 'modis_terra_sst.nc')
            },
            {
                'name': 'Chlorophyll',
                'url': 'https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/TERRA_MODIS.20000301_20200331.L3b.MC.CHL.nc',
                'filename': os.path.join(self.dataset_dirs['modis_terra'], 'modis_terra_chlorophyll.nc')
            }
        ]
        
        downloaded_files = []
        
        for dataset in datasets:
            # Check if file already exists
            if os.path.exists(dataset['filename']):
                print(f"File already exists: {dataset['filename']}")
                downloaded_files.append(dataset['filename'])
                continue
            
            success, filename = self.download_file(
                dataset['url'], 
                dataset['filename'], 
                f"MODIS-Terra {dataset['name']}"
            )
            
            if success:
                downloaded_files.append(filename)
                self.examine_dataset(filename, f"MODIS-Terra {dataset['name']}")
            
            time.sleep(2)  # Be respectful to the server
        
        return downloaded_files
    
    def download_seawifs_datasets(self):
        """Download SeaWiFS datasets"""
        print("\n=== SEAWIFS DATASETS ===")
        
        datasets = [
            {
                'name': 'Chlorophyll',
                'url': 'https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/SEAWIFS.19970901_20101231.L3b.MC.CHL.nc',
                'filename': os.path.join(self.dataset_dirs['seawifs'], 'seawifs_chlorophyll.nc')
            },
            {
                'name': 'Rrs_443',
                'url': 'https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/SEAWIFS.19970901_20101231.L3b.MC.RRS.nc',
                'filename': os.path.join(self.dataset_dirs['seawifs'], 'seawifs_rrs.nc')
            }
        ]
        
        downloaded_files = []
        
        for dataset in datasets:
            # Check if file already exists
            if os.path.exists(dataset['filename']):
                print(f"File already exists: {dataset['filename']}")
                downloaded_files.append(dataset['filename'])
                continue
            
            success, filename = self.download_file(
                dataset['url'], 
                dataset['filename'], 
                f"SeaWiFS {dataset['name']}"
            )
            
            if success:
                downloaded_files.append(filename)
                self.examine_dataset(filename, f"SeaWiFS {dataset['name']}")
            
            time.sleep(2)  # Be respectful to the server
        
        return downloaded_files
    
    def download_viirs_datasets(self):
        """Download VIIRS datasets"""
        print("\n=== VIIRS DATASETS ===")
        
        datasets = [
            {
                'name': 'SST',
                'url': 'https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/VIIRS.20120101_20231231.L3b.MC.SST.nc',
                'filename': os.path.join(self.dataset_dirs['viirs'], 'viirs_sst.nc')
            },
            {
                'name': 'Chlorophyll',
                'url': 'https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/VIIRS.20120101_20231231.L3b.MC.CHL.nc',
                'filename': os.path.join(self.dataset_dirs['viirs'], 'viirs_chlorophyll.nc')
            }
        ]
        
        downloaded_files = []
        
        for dataset in datasets:
            # Check if file already exists
            if os.path.exists(dataset['filename']):
                print(f"File already exists: {dataset['filename']}")
                downloaded_files.append(dataset['filename'])
                continue
            
            success, filename = self.download_file(
                dataset['url'], 
                dataset['filename'], 
                f"VIIRS {dataset['name']}"
            )
            
            if success:
                downloaded_files.append(filename)
                self.examine_dataset(filename, f"VIIRS {dataset['name']}")
            
            time.sleep(2)  # Be respectful to the server
        
        return downloaded_files
    
    def download_all_datasets(self):
        """Download all available datasets"""
        print("=== DOWNLOADING ALL NASA DATASETS ===")
        print("This will download multiple oceanographic datasets for shark habitat analysis")
        print("=" * 60)
        
        all_downloaded_files = []
        
        # Download MODIS-Aqua (known working)
        modis_aqua_files = self.download_modis_aqua_datasets()
        all_downloaded_files.extend(modis_aqua_files)
        
        # Download MODIS-Terra
        modis_terra_files = self.download_modis_terra_datasets()
        all_downloaded_files.extend(modis_terra_files)
        
        # Download SeaWiFS
        seawifs_files = self.download_seawifs_datasets()
        all_downloaded_files.extend(seawifs_files)
        
        # Download VIIRS
        viirs_files = self.download_viirs_datasets()
        all_downloaded_files.extend(viirs_files)
        
        # Summary
        print(f"\n=== DOWNLOAD SUMMARY ===")
        print(f"Total files downloaded: {len(all_downloaded_files)}")
        print(f"Files:")
        for i, filename in enumerate(all_downloaded_files, 1):
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"  {i}. {filename} ({file_size/1024/1024:.1f} MB)")
            else:
                print(f"  {i}. {filename} (failed)")
        
        return all_downloaded_files

def main():
    """Main function"""
    print("=== NASA MULTI-DATASET DOWNLOADER ===")
    print("Downloading oceanographic datasets for shark habitat prediction")
    print("=" * 60)
    
    # Initialize downloader
    downloader = NASADatasetDownloader()
    
    # Download all datasets
    downloaded_files = downloader.download_all_datasets()
    
    if downloaded_files:
        print(f"\n✅ SUCCESS: Downloaded {len(downloaded_files)} NASA dataset files")
        print("These datasets can now be used for shark habitat prediction analysis.")
    else:
        print(f"\n❌ FAILED: No files were downloaded successfully")

if __name__ == "__main__":
    main()
