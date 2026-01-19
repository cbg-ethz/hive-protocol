# Demo: Run `pixi run ruff check --fix messy_code.py && pixi run ruff format messy_code.py`
# This file is intentionally messy for the Ruff demo

import numpy as np
import os,sys
from typing import List,Dict,Optional
import json
def bad_func(x,y,z):
    result=x+y
    unused_var = 10
    if result>0:
        return result
    else:
        return 0

class messyClass:
    def __init__(self,value):
        self.value=value
    def get_value(self):return self.value
