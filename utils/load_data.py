import numpy as np
import pandas as pd
from tabulate import tabulate
from rdkit import Chem


def load_data(file_path):
    df = pd.read_csv(
        file_path, 
        sep=None, 
        engine="python"
    )
    smiles = df.iloc[:, 1].values
    targets = df.iloc[:, 2:].values
    
    return smiles, targets


def get_task_names(file_path):
    """
    Extracts task names from the CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List with task names
    """
    # Read just the header with the SAME parameters as load_data()
    df_header = pd.read_csv(
        file_path, 
        sep=None, 
        engine="python",
        nrows=0
    )
    # Assuming tasks start at column 2 (index 2)
    task_names = df_header.columns[2:].tolist()
    return task_names

def check_empty_samples(smiles, y_data, dataset_name=""):
    """
    Checks for samples without any valid value (only NaN).
    
    Args:
        smiles: Array of SMILES
        y_data: Array of targets
        dataset_name: Name of the dataset (train/val/test) for identification
        
    Returns:
        List of tuples (index, SMILES, dataset_name) of empty samples
    """
    empty_samples = []
    
    for i in range(len(smiles)):
        # Check if all values are NaN
        if np.all(np.isnan(y_data[i])):
            empty_samples.append((i, smiles[i], dataset_name))
    
    if empty_samples:
        print(f"\n⚠️ WARNING: {len(empty_samples)} SMILES without valid values found in {dataset_name}!")
        print(f"SMILES with only NaN values in {dataset_name}:")
        for idx, smile, _ in empty_samples[:5]:  # Show only the first 5
            print(f"  - ID {idx}: {smile}")
        if len(empty_samples) > 5:
            print(f"  ... and {len(empty_samples) - 5} more samples")
    else:
        print(f"\n✓ All SMILES have at least one valid value in {dataset_name}")
    
    return empty_samples

def check_empty_samples_all_datasets(train_data, val_data, test_data):
    """
    Checks for empty samples across all datasets and provides a summary.
    
    Args:
        train_data: Tuple (smiles, y_data) for training set
        val_data: Tuple (smiles, y_data) for validation set
        test_data: Tuple (smiles, y_data) for test set
        
    Returns:
        Dictionary with empty samples from all datasets
    """
    train_smiles, y_train = train_data
    val_smiles, y_val = val_data
    test_smiles, y_test = test_data
    
    # Check each dataset
    empty_train = check_empty_samples(train_smiles, y_train, "train")
    empty_val = check_empty_samples(val_smiles, y_val, "validation")
    empty_test = check_empty_samples(test_smiles, y_test, "test")
    
    # Combine all empty samples
    all_empty = empty_train + empty_val + empty_test
    
    if all_empty:
        print("\n" + "="*70)
        print("SUMMARY OF SMILES WITH ONLY NaN VALUES")
        print("="*70)
        
        # Group by dataset
        empty_by_dataset = {
            "train": len(empty_train),
            "validation": len(empty_val),
            "test": len(empty_test)
        }
        
        print("\nCount by dataset:")
        for dataset, count in empty_by_dataset.items():
            if count > 0:
                print(f"  - {dataset}: {count} SMILES")
        
        print(f"\nTotal SMILES with only NaN values: {len(all_empty)}")
        
        # Show some examples
        print("\nExamples:")
        for idx, smile, dataset in all_empty[:10]:
            print(f"  - {dataset} (ID {idx}): {smile}")
        
    else:
        print("\n✅ Great! No SMILES with only NaN values found in any dataset!")
    
    return {
        "train": empty_train,
        "validation": empty_val,
        "test": empty_test,
        "total": all_empty
    }

def check_invalid_smiles(smiles):
    """
    Checks for invalid SMILES using RDKit.
    
    Args:
        smiles: Array of SMILES
        
    Returns:
        List of tuples (index, SMILES) of invalid SMILES
    """
    invalid_smiles = []
    
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            invalid_smiles.append((i, smile))
    
    print(f"\nInvalid SMILES found: {len(invalid_smiles)}")
    if invalid_smiles and len(invalid_smiles) <= 10:
        for idx, smile in invalid_smiles:
            print(f"  - Index {idx}: {smile}")
    
    return invalid_smiles

def analyze_dataset(smiles, y_data, task_names, set_name):
    """
    Analyzes class distribution in a dataset.
    
    Args:
        smiles: Array of SMILES
        y_data: Array of targets
        task_names: List with task names
        set_name: Name of the set (train, validation, test)
        
    Returns:
        Dictionary with statistics per task
    """
    total_samples = len(smiles)
    print(f"\n{'-'*50}")
    print(f"Analysis of {set_name} set:")
    print(f"{'-'*50}")
    print(f"Total samples: {total_samples}")
    
    # Prepare data for table
    table_data = []
    stats = {}
    
    # For each task in the dataset
    for task_idx in range(y_data.shape[1]):
        task_name = task_names[task_idx] if task_idx < len(task_names) else f"Task_{task_idx}"
        
        # Remove NaN values for analysis
        valid_indices = ~np.isnan(y_data[:, task_idx])
        y_task = y_data[valid_indices, task_idx]
        
        active_samples = np.sum(y_task == 1)
        inactive_samples = np.sum(y_task == 0)
        total_valid = len(y_task)
        
        # Calculate inactive/active ratio
        if active_samples > 0:
            ratio = f"{inactive_samples/active_samples:.2f}:1"
        else:
            ratio = "∞ (no active samples)"
        
        # Store statistics
        stats[task_name] = {
            'total_valid': total_valid,
            'active': active_samples,
            'inactive': inactive_samples,
            'has_no_active': active_samples == 0,
            'has_no_inactive': inactive_samples == 0
        }
        
        table_data.append([
            task_idx + 1,
            task_name,
            total_valid,
            f"{active_samples} ({(active_samples/total_valid)*100:.2f}%)" if total_valid > 0 else "0 (0.00%)",
            f"{inactive_samples} ({(inactive_samples/total_valid)*100:.2f}%)" if total_valid > 0 else "0 (0.00%)",
            ratio
        ])
    
    # Create table
    headers = ["ID", "Task", "Valid Samples", "Active Samples (1)", "Inactive Samples (0)", "Inactive/Active Ratio"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    
    return stats

def analyze_all_datasets(train_data, val_data, test_data, task_names):
    """
    Analyzes all datasets and generates a summary.
    
    Args:
        train_data: Tuple (smiles, y_data) of training set
        val_data: Tuple (smiles, y_data) of validation set
        test_data: Tuple (smiles, y_data) of test set
        task_names: List with task names
    """
    train_smiles, y_train = train_data
    val_smiles, y_val = val_data
    test_smiles, y_test = test_data
    
    # Analyze each set
    train_stats = analyze_dataset(train_smiles, y_train, task_names, "train")
    val_stats = analyze_dataset(val_smiles, y_val, task_names, "validation")
    test_stats = analyze_dataset(test_smiles, y_test, task_names, "test")
    
    # Collect problematic tasks
    tasks_without_active = []
    tasks_without_inactive = []
    
    for task_name in task_names:
        if train_stats[task_name]['has_no_active']:
            tasks_without_active.append((task_name, "train"))
        if val_stats[task_name]['has_no_active']:
            tasks_without_active.append((task_name, "validation"))
        if test_stats[task_name]['has_no_active']:
            tasks_without_active.append((task_name, "test"))
            
        if train_stats[task_name]['has_no_inactive']:
            tasks_without_inactive.append((task_name, "train"))
        if val_stats[task_name]['has_no_inactive']:
            tasks_without_inactive.append((task_name, "validation"))
        if test_stats[task_name]['has_no_inactive']:
            tasks_without_inactive.append((task_name, "test"))
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY:")
    
    if tasks_without_active:
        print("\nTasks without ACTIVE samples (class 1):")
        without_active_data = [[task, set_name] for task, set_name in tasks_without_active]
        print(tabulate(without_active_data, headers=["Task", "Set"], tablefmt="pretty"))
    else:
        print("\nAll tasks have at least one ACTIVE sample (class 1) in all sets.")
    
    if tasks_without_inactive:
        print("\nTasks without INACTIVE samples (class 0):")
        without_inactive_data = [[task, set_name] for task, set_name in tasks_without_inactive]
        print(tabulate(without_inactive_data, headers=["Task", "Set"], tablefmt="pretty"))
    else:
        print("\nAll tasks have at least one INACTIVE sample (class 0) in all sets.")
    
    if not tasks_without_active and not tasks_without_inactive:
        print("\nAll tasks have samples from both classes in all sets!")
    
    return


def to_float_array(arr):
    def convert(val):
        if isinstance(val, float):
            return val
        elif isinstance(val, str):
            val = val.replace(',', '.')
            try:
                return float(val)
            except ValueError:
                return np.nan
        else:
            return np.nan
    return np.vectorize(convert)(arr)