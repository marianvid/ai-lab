# The macOS menu (AI-Lab Menu)

On the Mac, AI-Lab runs under a small menu bar app, `AI-Lab Menu`
(`system/macos/AI-LabMenu.swift`, installed by `scripts/install-macos-service.sh`).
The app starts at login and starts AI-Lab.

## States

| State | Icon | Meaning |
|---|---|---|
| Stopped | grey outline | AI-Lab is not running and is not wanted (Stop was chosen). |
| Running | green | AI-Lab runs and answers `GET /api/health`. |
| Warning | yellow | Waiting for the disk with the code, starting, restarting after an exit, or not answering `/api/health`. The second menu line says which. |
| Error | red | AI-Lab could not be started; the second menu line gives the reason, and a macOS notification is posted. |

## Waiting for the disk

The code and the configuration may live on an external disk that macOS mounts
after login. Before starting, the menu checks every 5 s that it can read
`ai_lab/main.py` in the project directory and the configuration, and that the
Python is executable. When they are not there within `diskWaitSeconds`
(in the app's settings file, which the installer writes; 300 s when absent), the menu shows Error.

## Restarts

When AI-Lab ends while it is wanted, the menu starts it again after 5 s, at
most 3 times in 10 minutes; the next exit is an Error. Stop, Restart and Start
from the menu are not counted; Start clears the count.

## Logs

`~/Library/Logs/AI-Lab/manager.log` holds AI-Lab's output;
`~/Library/Logs/AI-Lab/menu-events.log` holds one line per menu event
(waiting, started, exited, error).
