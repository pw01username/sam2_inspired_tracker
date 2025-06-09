import os
import csv
import re
import matplotlib.pyplot as plt
from pathlib import Path

# Base path
base_path = "/cluster/home/patricwu/ondemand/w/sam2.1/sam2/outputs/"
base_name = "checkpoints_checkpoint"
#base_name = "checkpoints_checkpoint"

# Function to extract checkpoint number
def get_checkpoint_number(folder_name, base_name):
    if folder_name == base_name:
        return 0
    match = re.match(fr"{base_name}_(\d+)", folder_name)
    if match:
        return int(match.group(1))
    return None

# Find all checkpoint folders and extract J&F values
checkpoint_data = []

# Scan the outputs directory
for folder in os.listdir(base_path):
    # Check if it matches the checkpoint pattern
    if folder == base_name or re.match(fr"{base_name}_\d+", folder):
        checkpoint_num = get_checkpoint_number(folder, base_name)
        if checkpoint_num is not None:
            # Construct path to results.csv
            results_path = os.path.join(base_path, folder, "results.csv")
            
            # Check if results.csv exists
            if os.path.exists(results_path):
                try:
                    # Read the CSV file
                    with open(results_path, 'r') as f:
                        reader = csv.reader(f)
                        # Skip header
                        next(reader)
                        # Get the first data row (Global score)
                        first_row = next(reader)
                        
                        # Extract J&F value (index 2)
                        if first_row[0].strip() == "Global score":
                            jf_value = float(first_row[2])
                            checkpoint_data.append((checkpoint_num, jf_value))
                            print(f"Checkpoint {checkpoint_num}: J&F = {jf_value}")
                except Exception as e:
                    print(f"Error reading {results_path}: {e}")
            else:
                print(f"No results.csv found in {folder}")

# Sort by checkpoint number
checkpoint_data.sort(key=lambda x: x[0])

if checkpoint_data:
    # Extract data for plotting
    checkpoint_nums = [x[0] for x in checkpoint_data]
    jf_values = [x[1] for x in checkpoint_data]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoint_nums, jf_values, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Checkpoint Number', fontsize=12)
    plt.ylabel('J&F Global Score', fontsize=12)
    plt.title('J&F Global Summary Values Across Checkpoints', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(checkpoint_nums, jf_values)):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=9)
    
    # Set y-axis limits with some padding
    y_min, y_max = min(jf_values), max(jf_values)
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Set x-axis to show integer checkpoint numbers
    plt.xticks(checkpoint_nums)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('jf_global_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('jf_global_summary_plot.pdf', bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total checkpoints found: {len(checkpoint_data)}")
    print(f"J&F range: {min(jf_values):.2f} - {max(jf_values):.2f}")
    print(f"Average J&F: {sum(jf_values)/len(jf_values):.2f}")
else:
    print("No checkpoint data found!")