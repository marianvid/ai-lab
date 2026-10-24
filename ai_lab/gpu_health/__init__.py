"""Long-term health of the NVIDIA card: readings, references, trends, alarms.

Read-only: the monitor asks NVML for values and never changes a setting.
See docs/gpu-health.md for what it watches and how to read the page. The
manager starts it through `service.GpuHealth`; its settings are
`settings.GpuHealthSettings`.
"""
