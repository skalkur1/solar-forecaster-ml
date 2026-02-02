from pvdaq_access import *
import pandas as pd
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent.parent       

system_id_list = ["4","10","50","51"]
for system_id in system_id_list:
    directory_name = PROJECT_ROOT / "data" / "pvdaq_raw" / ("system" + (system_id))
    os.mkdir(directory_name)
    downloadData(system_id, directory_name)#takes all data from that pvdaq system across all years