# Autonomous Parking System Development and Evaluation Using CARLA

This project implements an **Autonomous Parking System** using the [CARLA](https://github.com/carla-simulator/carla) simulator.  
It focuses on detecting available parking spaces, planning safe maneuvers, and executing autonomous parking actions in a simulated urban environment.

The system integrates:
- **Deep learning-based object detection** (YOLOv8)
- **Perception and control modules** built on the CARLA Python API
- **ZeroMQ** for efficient inter-process communication between detectors and vehicle control
- **Comprehensive test scripts** for evaluating various parking scenarios

---

## Features

- **Parallel Parking** and **Behind Vehicle Parking**
- **Parking in Crowded Environments**
- **YOLOv8 Integration** for Real-time Car Detection
- **Autonomous Path Planning and Maneuver Execution**
- **Multiple Camera Views** (front, rear, left, right, bird’s-eye semantic view)
- **Crowded Scenario Stress Testing**
- **Recorded Test Videos**

