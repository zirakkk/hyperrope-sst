from scipy.io import loadmat
import numpy as np
import scipy.io as sio
import random
import math
import os

def load_data(data_sign, data_path_prefix):
    """Load hyperspectral data and ground truth labels"""
    if data_sign == "WHHH":
        data = sio.loadmat(f'{data_path_prefix}/WHHH/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        labels = sio.loadmat(f'{data_path_prefix}/WHHH/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
    elif data_sign == "KSC":
        data = sio.loadmat(f'{data_path_prefix}/KSC/KSC.mat')['KSC']
        labels = sio.loadmat(f'{data_path_prefix}/KSC/KSC_gt.mat')['KSC_gt']
    elif data_sign == "Salinas":
        data = sio.loadmat(f'{data_path_prefix}/Salinas/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat(f'{data_path_prefix}/Salinas/Salinas_gt.mat')['salinas_gt']
    else:
        raise ValueError(f"Dataset {data_sign} not supported")
    
    return data, labels

def generate_train_test_split(data_sign, train_num_per_class, data_path_prefix, max_percent=0.5):
    """Generate train/test split with specified samples per class"""
    data, labels = load_data(data_sign, data_path_prefix)
    h, w, c = data.shape
    class_num = labels.max()
    
    # Group pixels by class
    class2data = {}
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                if labels[i, j] in class2data:
                    class2data[labels[i, j]].append([i, j])
                else:
                    class2data[labels[i, j]] = [[i, j]]
    
    # Initialize train/test matrices
    TR = np.zeros_like(labels)
    TE = np.zeros_like(labels)
    
    # Sample training pixels for each class
    for cl in range(class_num):
        class_index = cl + 1
        if class_index not in class2data:
            continue
            
        ll = class2data[class_index]
        all_index = list(range(len(ll)))
        real_train_num = min(train_num_per_class, int(len(all_index) * max_percent))
        
        if len(all_index) > 0:
            select_train_index = set(random.sample(all_index, real_train_num))
            for index in select_train_index:
                item = ll[index]
                TR[item[0], item[1]] = class_index
    
    # Test set = all labeled pixels - training pixels
    TE = labels - TR
    
    # Statistics
    ntr = TR[TR > 0].shape[0]
    nte = TE[TE > 0].shape[0]
    print(f'{data_sign}: train={ntr}, test={nte}, classes={class_num}')
    
    return {'TR': TR, 'TE': TE, 'input': data}

def main():
    """Main function to generate splits for WHHH, KSC, and Salinas datasets"""
    # Configuration
    datasets = ['Salinas']
    data_path_prefix = 'data\dataset'  
    train_samples_per_class = [10, 20, 30]  # Different training sizes
    random_seed = 42
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory
    output_dir = 'data\dataset\splits'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate splits for each dataset and configuration
    for data_sign in datasets:
        print(f"\nProcessing {data_sign} dataset...")
        
        for train_num in train_samples_per_class:
            try:
                # Generate train/test split
                result = generate_train_test_split(
                    data_sign=data_sign,
                    train_num_per_class=train_num,
                    data_path_prefix=data_path_prefix
                )
                
                # Convert data to single precision (fp32)
                result['input'] = result['input'].astype(np.float32)
                
                # Save to .mat file
                save_path = f'{output_dir}/{data_sign}_{train_num}_split.mat'
                sio.savemat(save_path, result, do_compression=False, format='5')
                print(f' Saved: {save_path}')
                
            except Exception as e:
                print(f'Error processing {data_sign} with {train_num} samples: {e}')

if __name__ == "__main__":
    main()