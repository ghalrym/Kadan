"""Host test compatibility import for the shared API CPU policy.

Operators run from outside the repository; add the source root explicitly. The
API package initializer is empty and the shared module has only stdlib imports.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from api.inference.cpu_thermal import ABORT_C, WARNING_C, POLICY_ID, IDENTITY, REQUIRED, read_identity, read_cpu_sensors, evaluate, CpuMonitor
