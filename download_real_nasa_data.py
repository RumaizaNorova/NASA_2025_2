"""
Download real NASA MODIS-Aqua data for 2020
"""

import requests
import os
from dotenv import load_dotenv
import xarray as xr
import numpy as np

# Load environment variables
load_dotenv()

def download_real_nasa_data():
    """Download real NASA MODIS-Aqua data"""
    print("=== DOWNLOADING REAL NASA DATA ===")
    
    token = os.getenv('EARTHDATA_TOKEN')
    headers = {'Authorization': f'Bearer {token}'}
    
    # Create directory for downloads
    os.makedirs("real_nasa_data", exist_ok=True)
    
    # The download URLs we found
    download_urls = {
        'sst': "https://oceandata.sci.gsfc.nasa.gov/cmr/getfile/AQUA_MODIS.20020701_20200731.L3b.MC.SST.nc",
        'chlorophyll': "https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/AQUA_MODIS.20020704_20250228.L3b.CU.CHL.nc"
    }
    
    downloaded_files = {}
    
    for data_type, url in download_urls.items():
        print(f"\n--- Downloading {data_type.upper()} data ---")
        print(f"URL: {url}")
        
        try:
            # Download the file
            response = requests.get(url, headers=headers, timeout=600)  # 10 min timeout
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Save the file
                filename = f"real_nasa_data/modis_aqua_{data_type}.nc"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                print(f"SUCCESS: Downloaded {len(response.content)} bytes")
                print(f"Saved to: {filename}")
                
                downloaded_files[data_type] = filename
                
                # Try to open and examine the data
                print(f"Examining the data...")
                try:
                    ds = xr.open_dataset(filename)
                    print(f"Dataset dimensions: {dict(ds.dims)}")
                    print(f"Dataset variables: {list(ds.data_vars.keys())}")
                    print(f"Dataset coordinates: {list(ds.coords.keys())}")
                    
                    # Print some data values
                    for var in ds.data_vars:
                        print(f"\nVariable: {var}")
                        print(f"  Shape: {ds[var].shape}")
                        print(f"  Data type: {ds[var].dtype}")
                        if hasattr(ds[var], 'values'):
                            values = ds[var].values
                            if values.size > 0:
                                print(f"  Min: {np.nanmin(values):.3f}")
                                print(f"  Max: {np.nanmax(values):.3f}")
                                print(f"  Mean: {np.nanmean(values):.3f}")
                    
                    # Check time dimension
                    if 'time' in ds.coords:
                        print(f"\nTime range:")
                        print(f"  Start: {ds.time.min().values}")
                        print(f"  End: {ds.time.max().values}")
                        print(f"  Number of time steps: {len(ds.time)}")
                    
                    # Check spatial dimensions
                    if 'latitude' in ds.coords:
                        print(f"\nLatitude range: {ds.latitude.min().values:.3f} to {ds.latitude.max().values:.3f}")
                    if 'longitude' in ds.coords:
                        print(f"Longitude range: {ds.longitude.min().values:.3f} to {ds.longitude.max().values:.3f}")
                    
                except Exception as e:
                    print(f"Error examining data: {e}")
                    
            else:
                print(f"ERROR: Download failed with status {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                
        except Exception as e:
            print(f"Exception during download: {e}")
    
    return downloaded_files

if __name__ == "__main__":
    files = download_real_nasa_data()
    
    if files:
        print(f"\nSUCCESS: Downloaded {len(files)} files:")
        for data_type, filename in files.items():
            print(f"  {data_type}: {filename}")
    else:
        print(f"\nFAILED: No files downloaded")

