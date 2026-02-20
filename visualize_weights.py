import csv
import matplotlib.pyplot as plt
import numpy as np
import re

def visualize_weights(csv_path='linear_model_weights.csv', output_img='weights_plot.png'):
    features = []
    weights = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row: continue
            feat, w = row[0], float(row[1])
            if feat == 'BIAS':
                continue
            features.append(feat)
            weights.append(w)
            
    # Separate and sort
    bp_data = [(f, w, int(re.search(r'\d+', f).group())) for f, w in zip(features, weights) if f.startswith('bp_')]
    rp_data = [(f, w, int(re.search(r'\d+', f).group())) for f, w in zip(features, weights) if f.startswith('rp_')]
    
    bp_data.sort(key=lambda x: x[2])
    rp_data.sort(key=lambda x: x[2])
    
    bp_features = [x[0] for x in bp_data]
    bp_weights = [x[1] for x in bp_data]
    
    rp_features = [x[0] for x in rp_data]
    rp_weights = [x[1] for x in rp_data]
    
    all_features = bp_features + rp_features
    all_weights = bp_weights + rp_weights
    
    plt.figure(figsize=(14, 6))
    x = np.arange(len(all_features))
    
    # Plot BP
    plt.bar(x[:len(bp_data)], bp_weights, label='BP weights', color='blue', alpha=0.7)
    # Plot RP
    plt.bar(x[len(bp_data):], rp_weights, label='RP weights', color='red', alpha=0.7)
    
    # Formatting
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('Features (BP 1-55, RP 1-55)', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Linear Model Weights for BP and RP Features', fontsize=14)
    
    # Set x-ticks (show every 5th label)
    plt.xticks(x[::5], [all_features[i] for i in range(0, len(all_features), 5)], rotation=45)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Plot saved to {output_img}")

if __name__ == '__main__':
    visualize_weights()
