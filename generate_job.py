#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import Dict, Any

import click
import yaml
from dotenv import load_dotenv
from click.testing import CliRunner

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
            'name': 'sft',
            'namespace': 'yn-gpu-workload',
            'labels': {
                'k8s-user': '${K8S_USER_NAME}'
            }
        },
        'spec': {
            'backoffLimit': 0,
            'ttlSecondsAfterFinished': 60,
            'template': {
                'spec': {
                    'nodeSelector': {
                        'nvidia.com/gpu.present': 'true',
                        'nvidia.com/gpu.product': '${GPU_TYPE}'
                    },
                    'securityContext': {
                        'runAsUser': '${K8S_USER_ID}',
                        'runAsGroup': '${K8S_USER_GROUP}',
                        'fsGroup': '${K8S_USER_GROUP}'
                    },
                    'containers': [{
                        'name': 'main',
                        'image': '${IMAGE}',
                        'env': [
                            {'name': 'HOME', 'value': '${K8S_USER_HOME}'},
                            {'name': 'MODEL', 'value': '${MODEL}'},
                            {'name': 'DATASET', 'value': '${DATASET}'},
                            {'name': 'HF_TOKEN', 'value': '${HF_TOKEN}'},
                        ],
                        'volumeMounts': [
                            {'name': 'shm', 'mountPath': '/dev/shm'},
                            {'name': 'home', 'mountPath': '${K8S_USER_HOME}'}
                        ],
                        'resources': {
                            'limits': {
                                'cpu': '48',
                                'memory': '1200Gi',
                                'nvidia.com/gpu': '8'
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
                                'claimName': '${K8S_GPU_PVC}'
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
    yaml_str = yaml.dump(template)
    for key, value in variables.items():
        yaml_str = yaml_str.replace(f'${{{key}}}', str(value))
    return yaml.safe_load(yaml_str)

@click.command()
@click.option('--config', type=click.Path(exists=True), default=None, help='YAML config file')
@click.option('--k8s-user-name', type=str, default=None, help='K8s user name')
@click.option('--k8s-user-id', type=str, default=None, help='K8s user id')
@click.option('--k8s-user-group', type=str, default=None, help='K8s user group')
@click.option('--k8s-user-home', type=str, default=None, help='K8s user home')
@click.option('--k8s-cpu-pvc', type=str, default=None, help='K8s cpu pvc')
@click.option('--k8s-gpu-pvc', type=str, default=None, help='K8s gpu pvc')
@click.option('--k8s-gpu-shared-pvc', type=str, default=None, help='K8s gpu shared pvc')
@click.option('--image', type=str, default=None, help='Image')
@click.option('--gpu-type', type=str, default=None, help='GPU type')
@click.option('--hf-token', type=str, default=os.getenv('HF_TOKEN'), help='HF token')
@click.option('--output', type=click.Path(exists=False), required=True, help='Output YAML file path')
def main(config: str, image: str, output: str,
          gpu_type: str,
          hf_token: str,
          k8s_user_name: str, 
          k8s_user_id: str, k8s_user_group: str, 
          k8s_user_home: str, k8s_cpu_pvc: str, 
          k8s_gpu_pvc: str, k8s_gpu_shared_pvc: str):
    """Generate Kubernetes job YAML file from environment variables and CLI arguments."""
    if not output:
        raise click.UsageError("Both --config and --output parameters are required")
    if not hf_token:
        raise click.UsageError("HF token is required. Either set it as an environment variable or pass it as a command line argument.")

    config_data = {}
    if config:
        config_data = load_config_file(config)

    variables = {
        'K8S_USER_NAME': k8s_user_name or config_data['K8S_USER_NAME'],
        'K8S_USER_ID': k8s_user_id or config_data['K8S_USER_ID'],
        'K8S_USER_GROUP': k8s_user_group or config_data['K8S_USER_GROUP'],
        'K8S_USER_HOME': k8s_user_home or config_data['K8S_USER_HOME'],
        'K8S_CPU_PVC': k8s_cpu_pvc or config_data['K8S_CPU_PVC'],
        'K8S_GPU_PVC': k8s_gpu_pvc or config_data['K8S_GPU_PVC'],
        'K8S_GPU_SHARED_PVC': k8s_gpu_shared_pvc or config_data['K8S_GPU_SHARED_PVC'],
        'IMAGE': image or config_data['IMAGE'],
        'GPU_TYPE': gpu_type or config_data['GPU_TYPE'],
    }

    # Load and process the template
    template = load_template()
    final_yaml = substitute_variables(template, variables)
    
    # Write the output
    output_path = Path(output)
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)

    click.echo(f"Generated Kubernetes job YAML at: {output_path}")

if __name__ == '__main__':
    main()  # Click will automatically handle command line arguments