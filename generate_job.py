#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import click
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config_file(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise click.FileError(path, hint="Config file not found.")
    

def load_template() -> Dict[str, Any]:
    """Load the base job template"""
    return {
        'apiVersion': 'batch/v1',
        'kind': 'Job',
        'metadata': {
            'name': '${k8s_user_name}-${job_name}',
            'namespace': 'yn-gpu-workload',
            'labels': {
                'k8s-user': '${k8s_user_name}'
            }
        },
        'spec': {
            'backoffLimit': 0,
            'ttlSecondsAfterFinished': 60,
            'template': {
                'spec': {
                    'nodeSelector': {
                        'nvidia.com/gpu.present': 'true'
                        # gpu.product will be added conditionally
                    },
                    'securityContext': {
                        'runAsUser': '${k8s_user_id}',
                        'runAsGroup': '${k8s_user_group}',
                        'fsGroup': '${k8s_user_group}'
                    },
                    'containers': [{
                        'name': 'main',
                        'image': '${image}',
                        'env': [
                            {'name': 'HOME', 'value': '/home'},
                            {'name': 'HF_LOCAL_STORAGE', 'value': '${hf_local_storage}'},
                            {'name': 'HF_TOKEN', 'value': '${hf_token}'},
                        ],
                        'volumeMounts': [
                            {'name': 'shm', 'mountPath': '/dev/shm'},
                            {'name': 'home', 'mountPath': '/home'}
                        ],
                        'resources': {
                            'requests': {
                                'cpu': '${cpu_low}',
                                'memory': '${memory_low}',
                                'nvidia.com/gpu': '${gpu_low}'
                            },
                            'limits': {
                                'cpu': '${cpu_high}',
                                'memory': '${memory_high}',
                                'nvidia.com/gpu': '${gpu_high}'
                            }
                        },
                        'imagePullPolicy': 'IfNotPresent'
                    }],
                    'volumes': [
                        {
                            'name': 'shm',
                            'emptyDir': {
                                'medium': 'Memory',
                                'sizeLimit': '1200Gi'
                            }
                        },
                        {
                            'name': 'home',
                            'persistentVolumeClaim': {
                                'claimName': '${k8s_gpu_pvc}'
                            }
                        }
                    ],
                    'restartPolicy': 'Never'
                }
            }
        }
    }

def substitute_variables(template: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
    """Substitute variables in the template with actual values"""
    # Add gpu.product to nodeSelector if gpu_type is specified
    if variables.get('gpu_type'):
        template['spec']['template']['spec']['nodeSelector']['nvidia.com/gpu.product'] = variables['gpu_type']
    
    # Add command to template
    if variables.get('command'):
        template['spec']['template']['spec']['containers'][0]['command'] = variables['command']
    
    yaml_str = yaml.dump(template)
    for key, value in variables.items():
        if value is not None:  # Only substitute if value is not None
            yaml_str = yaml_str.replace(f'${{{key}}}', str(value))
    return yaml.safe_load(yaml_str)

@click.command()
@click.option('--config', type=str, default=os.getenv('CONFIG_PATH'), help='YAML config file')
@click.option('--image', type=str, default=None, help='Image')
@click.option('--command', type=str, default=None, help='Command')
@click.option('--hf-token', type=str, default=os.getenv('HF_TOKEN'), help='HF token')
@click.option('--output', type=str, default=None, help='Output YAML file path')
@click.option('--run', type=str, is_flag=True, help='run the generated job')
@click.option('--k8s-user-name', type=str, default=None, help='K8s user name')
@click.option('--k8s-user-id', type=str, default=None, help='K8s user id')
@click.option('--k8s-user-group', type=str, default=None, help='K8s user group')
@click.option('--k8s-user-home', type=str, default=None, help='The directory of your home on HPC, this will be mounted to the container and any data reading from your personal data and permament data saving will be done in this directory')
@click.option('--hf-local-storage', type=str, default=None, help='The directory of your huggingface cache on HPC, relative to your home directory')
@click.option('--k8s-gpu-pvc', type=str, default=None, help='K8s gpu pvc')
@click.option('--gpu-type', type=str, default=None, help='GPU type')
@click.option('--gpu-low', type=int, default=None, help='the minimum number of GPUs to request')
@click.option('--gpu-high', type=int, default=None, help='the maximum number of GPUs to request')
@click.option('--cpu-low', type=int, default=None, help='the minimum number of CPUs to request')
@click.option('--cpu-high', type=int, default=None, help='the maximum number of CPUs to request')
@click.option('--memory-low', type=int, default=None, help='the minimum amount of memory to request')
@click.option('--memory-high', type=int, default=None, help='the maximum amount of memory to request')


def main(config: str, **kwargs):
    config_data = load_config_file(config)
    for key, value in kwargs.items():
        if value is not None:
            config_data[key] = value

    # check HF_TOKEN
    if 'hf_token' not in config_data:
        raise click.UsageError("HF_TOKEN is required. Either set it as an environment variable or pass it as a command line argument.")

    # Load and process the template
    template = load_template()
    final_yaml = substitute_variables(template, config_data)
    
    # Write the output
    output_path = Path(config_data['output'])
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)

    click.echo(f"Generated Kubernetes job YAML at: {output_path}")

    if kwargs.get('run'):
        click.echo(f"Running the generated job at: {output_path}")
        subprocess.run(['kubectl', 'apply', '-f', output_path])

if __name__ == '__main__':
    main()  # Click will automatically handle command line arguments