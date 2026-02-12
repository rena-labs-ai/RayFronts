"""Visualizer exports.

Some visualizers depend on optional runtime packages.
"""

import logging

from rayfronts.visualizers.base import Mapping3DVisualizer

logger = logging.getLogger(__name__)

failed_to_import = list()

try:
  from rayfronts.visualizers.rerun import RerunVis
except ModuleNotFoundError:
  failed_to_import.append("RerunVis")

try:
  from rayfronts.visualizers.ros import Ros2Vis
except ModuleNotFoundError:
  failed_to_import.append("Ros2Vis")

if len(failed_to_import) > 0:
  logger.info("Could not import %s. Install optional visualizer dependencies "
              "to enable them.", ", ".join(failed_to_import))
