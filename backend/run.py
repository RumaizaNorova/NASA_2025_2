"""
Run the FastAPI backend server
"""

import uvicorn
import os
import sys

# Add the parent directory to the path so we can access the model files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

