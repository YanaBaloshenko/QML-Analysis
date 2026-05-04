from iqm import qiskit_iqm
from iqm.qiskit_iqm import IQMProvider


import numpy as np
import pandas as pd
import time

import os
from dotenv import load_dotenv


### Initializing QPU connection ###
load_dotenv()
token = os.getenv("IQM_TOKEN")
iqm_link = "https://resonance.meetiqm.com/"
qpu = "sirius" # sirius, garnet or emerald
provider = IQMProvider(iqm_link, qpu, token)
backend = provider.get_backend()


