## Dynamic Pathfinding Agent — Dependencies
## All dependencies are part of Python's standard library.
## No external packages need to be installed via pip.
#
# Required Python version: 3.6+
#
# Standard library modules used:
#   - tkinter     (GUI framework)
#   - heapq       (priority queue for A* and GBFS)
#   - random      (maze generation, dynamic obstacles)
#   - time        (execution time measurement)
#   - math        (Euclidean heuristic)
#   - collections (defaultdict for g-scores)
#
# ──────────────────────────────────────────────
# INSTALLATION INSTRUCTIONS
# ──────────────────────────────────────────────
#
# Windows & macOS:
#   tkinter is bundled with the official Python installer.
#   No extra steps needed. Just run:
#       python pathfinding_agent.py
#
# Linux (Ubuntu / Debian):
#   tkinter is sometimes not included by default. Install it with:
#       sudo apt install python3-tk
#   Then run:
#       python3 pathfinding_agent.py
#
# Linux (Fedora / RHEL):
#       sudo dnf install python3-tkinter
#
# Linux (Arch):
#       sudo pacman -S tk
