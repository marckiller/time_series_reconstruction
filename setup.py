import os
import subprocess
import sys
import platform

def create_virtual_environment(env_name='venv'):
    if not os.path.exists(env_name):
        subprocess.check_call([sys.executable, '-m', 'venv', env_name])
        print(f"Virtual environment '{env_name}' created.")
    else:
        print(f"Virtual environment '{env_name}' already exists.")

def get_activate_command(env_name='venv'):
    system = platform.system()
    if system == 'Windows':
        activate_cmd = f"{env_name}\\Scripts\\activate.bat"
        activate_msg = f"Run:\n{activate_cmd}"
    elif system == 'Linux' or system == 'Darwin':
        activate_cmd = f"source {env_name}/bin/activate"
        activate_msg = f"Run:\n{activate_cmd}"
    else:
        activate_msg = f"Please activate your virtual environment manually. Location: {env_name}"
    return activate_cmd, activate_msg

def install_requirements(env_name='venv'):
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        system = platform.system()
        if system == 'Windows':
            pip_executable = os.path.join(env_name, 'Scripts', 'pip.exe')
        else:
            pip_executable = os.path.join(env_name, 'bin', 'pip')
        try:
            subprocess.check_call([pip_executable, 'install', '-r', requirements_file])
            print(f"Installed requirements from {requirements_file}.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
    else:
        print(f"No {requirements_file} file found.")

if __name__ == '__main__':
    venv_name = 'venv'
    create_virtual_environment(venv_name)
    activate_cmd, activate_msg = get_activate_command(venv_name)
    install_requirements(venv_name)
    print("\nSetup completed.")
    print(f"To activate virtual environment:\n{activate_msg}")