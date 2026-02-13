# Pipeline Configuration

## Global
- sample_interval_sec: 0.4
- clock_start: 13:45:00
- cpu_core_count: 12
- l1_cache_kb: 64
- l2_cache_kb: 1024
- l3_cache_mb: 36
- gpu_count: 3
- gpu_vram_totals_mb: 24576,16384,12288

## Stages
{ id: router, name: Router, start: 0.0, end: 1.2, color: #8B5CF6, label: Step 1: Router, hw: CPU only, cpu: on, mem: on, cpu_level: 0.6, mem_level: 0.5 }
{ id: extract, name: Extract, start: 1.2, duration: 3.0, color: #3B82F6, label: Step 2: Extract, hw: CPU + RAM + SSD, cpu: gradual_increase, mem: gradual_increase, disk: gradual_increase, cpu_level: 0.9, mem_level: 0.8, disk_level: 0.7 }
{ id: inference, name: Inference, start: 4.0, end: 10.5, color: #10B981, label: Step 3: Inference, hw: GPU + CPU, cpu: on, gpu: gradual_increase, mem: on, cpu_level: 0.7, gpu_level: 1.0, mem_level: 0.7, cpu_load_mode: custom, cpu_core_loads: 1,1,1,1,1,1,2,2,2,2,3,3, gpu_load_mode: custom, gpu_device_loads: 1,2,3 }
{ id: assembly, name: Assembly, start: 10.5, duration: 3.0, color: #F59E0B, label: Step 4: Assembly, hw: CPU + SSD, cpu: gradual_decrease, mem: gradual_decrease, disk: gradual_increase, cpu_level: 0.6, mem_level: 0.5, disk_level: 0.9 }

## Events
{ event_id: E001, name: OCR Burst, start: 1.3, end: 2.4, resource_target: cpu, cpu: gradual_increase, disk: gradual_increase, cpu_level: 1.0, disk_level: 0.8 }
{ event_id: E002, name: Retrieval, start: 2.0, duration: 1.4, resource_target: none, network: gradual_increase, disk: on, network_level: 0.9, disk_level: 0.5 }
{ event_id: E003, name: GPU Warmup, start: 4.2, duration: 1.2, resource_target: gpu, gpu: gradual_increase, mem: on, gpu_level: 0.8, mem_level: 0.4 }
{ event_id: E004, name: Batch Compute A, start: 5.1, duration: 2.4, resource_target: cpu_gpu, gpu: on, cpu: on, gpu_level: 1.0, cpu_level: 0.6, cpu_load_mode: custom, cpu_core_loads: 0,0,0,0,1,1,2,2,3,3,4,4, gpu_load_mode: custom, gpu_device_loads: 0,1,4 }
{ event_id: E005, name: Batch Compute B, start: 5.8, duration: 2.0, resource_target: cpu_gpu, gpu: on, cpu: gradual_increase, gpu_level: 0.9, cpu_level: 0.7, cpu_load_mode: uniform, gpu_load_mode: uniform }
{ event_id: E006, name: Postprocess Upload, start: 10.9, duration: 1.8, resource_target: cpu, network: gradual_increase, cpu: on, network_level: 1.0, cpu_level: 0.5 }
