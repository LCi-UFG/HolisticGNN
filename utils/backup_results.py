"""
Simple module for backing up experiment results.
"""
import os
import shutil
import datetime
import glob

def create_backup_directory(base_directory, experiment_name):
    """Creates backup directory with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{experiment_name}_backup_results_{timestamp}"
    destination_directory = os.path.join(base_directory, full_name)
    
    # Create folders
    os.makedirs(destination_directory, exist_ok=True)
    os.makedirs(os.path.join(destination_directory, "embeddings"), exist_ok=True)
    
    print(f"Creating backup of results in: {destination_directory}")
    return destination_directory

def move_embeddings(destination_directory):
    """Moves embeddings to backup folder."""
    source_embeddings = "../output/embeddings"
    destination_embeddings = os.path.join(destination_directory, "embeddings")
    
    if os.path.exists(source_embeddings):
        embedding_files = os.listdir(source_embeddings)
        for file in embedding_files:
            source_path = os.path.join(source_embeddings, file)
            destination_path = os.path.join(destination_embeddings, file)
            shutil.move(source_path, destination_path)
        print(f"✓ {len(embedding_files)} embedding files moved successfully")
    else:
        print("✗ Embeddings folder not found at source")

def move_figures(destination_directory):
    """Moves figures to backup folder."""
    source_figures = "../output/figures"
    destination_figures = os.path.join(destination_directory, "figures")
    
    if os.path.exists(source_figures):
        # Create figures subdirectory in backup
        os.makedirs(destination_figures, exist_ok=True)
        
        figure_files = os.listdir(source_figures)
        for file in figure_files:
            source_path = os.path.join(source_figures, file)
            destination_path = os.path.join(destination_figures, file)
            shutil.move(source_path, destination_path)
        print(f"✓ {len(figure_files)} figure files moved successfully")
    else:
        print("✗ Figures folder not found at source")

def move_specific_files(destination_directory, fine_tuning=False):
    """Moves/copies specific model files with conditional filenames for fine-tuning."""
    # Determine filenames based on mode
    model_file = "new_model.pth" if fine_tuning else "model.pth"
    params_file = "new_model.json" if fine_tuning else "model.json"

    files_to_process = [
        {
            "path": "../output/models/model.pth",
            "destination": os.path.join(destination_directory, model_file),
            "description": model_file,
            "action": "move"
        },
        {
            "path": "../output/optimization/optuna_study.db",
            "destination": os.path.join(destination_directory, "optuna_study.db"),
            "description": "optuna_study.db",
            "action": "copy"  # Copy because it may be in use
        },
        {
            "path": "../output/params/model.json",
            "destination": os.path.join(destination_directory, params_file),
            "description": params_file,
            "action": "move"
        },
        {
            "path": "../output/calibration/thresholds.json",
            "destination": os.path.join(destination_directory, "thresholds.json"),
            "description": "thresholds.json",
            "action": "move"
        }
    ]
    
    for item in files_to_process:
        if os.path.exists(item["path"]):
            if item["action"] == "copy":
                shutil.copy2(item["path"], item["destination"])
                print(f"✓ File {item['description']} copied successfully  - SQLite file cannot be moved due to an open connection to optuna. Delete manually after restarting the Kernel.")
            else:
                shutil.move(item["path"], item["destination"])
                print(f"✓ File {item['description']} moved successfully")
        else:
            print(f"✗ File not found: {item['path']}")

def capture_optimization(destination_directory):
    """Captures and saves optimization output."""
    study_output = ""
    
    try:
        ip = get_ipython()
        
        if ip is not None and hasattr(ip, 'user_ns') and 'best_params' in ip.user_ns:
            best_params = ip.user_ns['best_params']
            study_output = f"Best hyperparameters: {str(best_params)}\n"
            
            if 'study' in ip.user_ns:
                study = ip.user_ns['study']
                study_output += f"\nBest value: {study.best_value}\n"
                study_output += f"Best trial: {study.best_trial.number}\n"
                
                study_output += "\nCompleted trials:\n"
                for trial in study.trials:
                    if trial.state.name == 'COMPLETE':
                        study_output += f"Trial {trial.number}: value={trial.value}, params={trial.params}\n"
            
            print("✓ Optimization output captured successfully")
        else:
            study_output = "# Optimization output not found\n"
            study_output += "# Please manually add the optimization results here\n"
    except:
        study_output = "# Error capturing optimization output\n"
    
    # Save file
    output_file = os.path.join(destination_directory, "optimizations.txt")
    with open(output_file, "w") as f:
        f.write(study_output)
    
    print(f"✓ Optimization file created at {output_file}")

def generate_summary(destination_directory, fine_tuning=False):
    """Generates backup summary."""
    # Get correct filenames based on mode
    model_file = "new_model.pth" if fine_tuning else "model.pth"
    params_file = "new_model.json" if fine_tuning else "model.json"

    print("\nBackup summary:")
    print(f"Backup location: {os.path.abspath(destination_directory)}")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count files in each subdirectory
    embeddings_count = 0
    figures_count = 0
    
    embeddings_dir = os.path.join(destination_directory, "embeddings")
    if os.path.exists(embeddings_dir):
        embeddings_count = len([f for f in os.listdir(embeddings_dir) if os.path.isfile(os.path.join(embeddings_dir, f))])
    
    figures_dir = os.path.join(destination_directory, "figures")
    if os.path.exists(figures_dir):
        figures_count = len([f for f in os.listdir(figures_dir) if os.path.isfile(os.path.join(figures_dir, f))])
    
    # Count total files
    total_files = sum(1 for _ in glob.glob(f"{destination_directory}/**/*", recursive=True) if os.path.isfile(_))
    
    print(f"Files moved:")
    print(f"  - Embeddings: {embeddings_count} files")
    print(f"  - Figures: {figures_count} files")
    print(f"  - Model files: {model_file}, {params_file}, thresholds.json, optuna_study.db")
    print(f"Total files in backup: {total_files}")

def execute_backup(base_directory, experiment_name, fine_tuning=False):
    """
    Executes complete backup.
    
    Args:
        base_directory: Directory where to create the backup
        experiment_name: Experiment name 
        fine_tuning: Whether this is a fine-tuning run (default: False)
    
    Returns:
        Path of the created backup directory
    """
    # Create directory
    destination_directory = create_backup_directory(base_directory, experiment_name)
    
    # Move files
    move_embeddings(destination_directory)
    move_specific_files(destination_directory, fine_tuning)
    move_figures(destination_directory)
    # Capture optimization
    capture_optimization(destination_directory)
    
    # Summary
    generate_summary(destination_directory, fine_tuning)
    
    return destination_directory