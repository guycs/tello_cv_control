import logging
import sys
from droneapp.models.drone_manager import DroneManager
import droneapp.controllers.server


logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout)


if __name__ == '__main__':
    DroneManager()
    droneapp.controllers.server.run()