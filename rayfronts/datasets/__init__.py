import logging
logger = logging.getLogger(__name__)

from rayfronts.datasets.base import PosedRgbdDataset, SemSegDataset

failed_to_import = list()
try:
  from rayfronts.datasets.replica import NiceReplicaDataset, SemanticNerfReplicaDataset
except:
  failed_to_import.append("NiceReplicaDataset/SemanticNerfReplicaDataset")

try:
  from rayfronts.datasets.ros import RosnpyDataset, Ros2Subscriber
except:
  failed_to_import.append("RosnpyDataset/Ros2Subscriber")

try:
  from rayfronts.datasets.scannet import ScanNetDataset
except:
  failed_to_import.append("ScanNetDataset")

try:
  from rayfronts.datasets.tartanair import TartanAirDataset
except:
  failed_to_import.append("TartanAirDataset")

try:
  from rayfronts.datasets.dummy import DummyDataset
except:
  failed_to_import.append("DummyDataset")

try:
  from rayfronts.datasets.airsim import AirSimDataset
except:
  failed_to_import.append("AirSimDataset")

try:
  from rayfronts.datasets.mcap import McapRos2Dataset
except:
  failed_to_import.append("McapRos2Dataset")

if len(failed_to_import) > 0:
  logger.info(
    "Could not import %s. Make sure you have their extra dependencies "
    "installed if you want to use them.", ", ".join(failed_to_import))
