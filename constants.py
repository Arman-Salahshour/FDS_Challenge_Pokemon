# ================================
# Standard Library Imports
# ================================
import os
import json
import time
import warnings
from typing import List, Dict, Any, Callable, Optional, Type

# ================================
# Third-Party Imports
# ================================

# Data Handling
import pandas as pd
import numpy as np

# Progress Bars
from tqdm.notebook import tqdm

# Kaggle (dataset handling)
import kagglehub

# Google Colab utilities
from google.colab import files

# Scikit-Learn
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Plotting
import matplotlib.pyplot as plt

# ================================
# Settings
# ================================
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# ================================
# Global Constants
# ================================
SEED = 42
TARGET = "player_won"
BATTLE_ID = "battle_id"
