import csv
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

def visualize_weights(csv_path='linear_model_weights.csv', output_img='weights_plot.png'):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [(row[0], float(row[1])) for row in reader if row and row[0] != 'BIAS']
            
    # Separate and sort by integer suffix
    key_fn = lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0
    bp_data = sorted([x for x in data if x[0].startswith('bp_')], key=key_fn)
    rp_data = sorted([x for x in data if x[0].startswith('rp_')], key=key_fn)
    other_data = [x for x in data if not (x[0].startswith('bp_') or x[0].startswith('rp_'))]
    
    all_data = bp_data + rp_data + other_data
    all_features = [x[0] for x in all_data]
    all_weights = [x[1] for x in all_data]
    bp_len = len(bp_data)
    rp_len = len(rp_data)
    
    plt.figure(figsize=(14, 6))
    x = np.arange(len(all_features))
    
    # Plot BP, RP, and others
    plt.bar(x[:bp_len], all_weights[:bp_len], label='BP weights', color='blue', alpha=0.7)
    plt.bar(x[bp_len:bp_len+rp_len], all_weights[bp_len:bp_len+rp_len], label='RP weights', color='red', alpha=0.7)
    if other_data:
        plt.bar(x[bp_len+rp_len:], all_weights[bp_len+rp_len:], label='Other weights', color='green', alpha=0.7)
    
    # Formatting
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    title_suffix = ' (with EBV)' if 'ebv' in all_features else ''
    plt.title(f'Linear Model Weights for Features{title_suffix}', fontsize=14)
    
    # Set x-ticks (show every 5th label)
    plt.xticks(x[::5], [all_features[i] for i in range(0, len(all_features), 5)], rotation=45)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Plot saved to {output_img}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-ebv', action='store_true', help="Visualize weights including 'ebv' feature.")
    args = parser.parse_args()
    
    csv_file = 'linear_model_weights_ebv.csv' if args.use_ebv else 'linear_model_weights.csv'
    img_file = 'weights_plot_ebv.png' if args.use_ebv else 'weights_plot.png'
    
    visualize_weights(csv_file, img_file)
