import os
import json
import glob

def analyze_results(save_dir):
    results = []
    
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist.")
        return

    for run_dir in os.listdir(save_dir):
        full_run_dir = os.path.join(save_dir, run_dir)
        if not os.path.isdir(full_run_dir):
            continue
            
        # Parse M, L, k from directory name if possible
        # Format: run_TIMESTAMP_M{M}_L{L}_k{k}
        try:
            parts = run_dir.split('_')
            # parts example: ['run', '13', '59', '54', 'M50', 'L1', 'k10']
            # We need to find the parts starting with M, L, k
            m_part = next((p for p in parts if p.startswith('M')), None)
            l_part = next((p for p in parts if p.startswith('L')), None)
            k_part = next((p for p in parts if p.startswith('k')), None)
            
            M = int(m_part[1:]) if m_part else None
            L = int(l_part[1:]) if l_part else None
            k = int(k_part[1:]) if k_part else None
        except Exception as e:
            print(f"Skipping {run_dir}: could not parse parameters. {e}")
            continue

        # Find score files
        val_files = glob.glob(os.path.join(full_run_dir, "val_scores_*.json"))
        test_files = glob.glob(os.path.join(full_run_dir, "test_scores_*.json"))
        
        if not val_files or not test_files:
            continue
            
        # Use the latest file if multiple exist (though there should be one per run dir)
        val_file = sorted(val_files)[-1]
        test_file = sorted(test_files)[-1]
        
        try:
            with open(val_file, 'r') as f:
                # The file format has "Class scores:" then JSON then "Overall scores:" then JSON
                # Wait, my previous edit used json.dump twice in the same file with text in between.
                # That is NOT valid JSON. I need to handle that.
                content = f.read()
                
            # Extract Overall scores JSON block
            # It starts after "Overall scores:\n"
            overall_marker = "Overall scores:\n"
            if overall_marker in content:
                overall_json_str = content.split(overall_marker)[1]
                val_scores = json.loads(overall_json_str)
            else:
                # Fallback if format is different
                print(f"Could not parse val scores in {val_file}")
                continue

            with open(test_file, 'r') as f:
                content = f.read()
                
            if overall_marker in content:
                overall_json_str = content.split(overall_marker)[1]
                test_scores = json.loads(overall_json_str)
            else:
                print(f"Could not parse test scores in {test_file}")
                continue
                
            results.append({
                "M": M,
                "L": L,
                "k": k,
                "val_acc": val_scores.get("accuracy", 0),
                "test_acc": test_scores.get("accuracy", 0),
                "val_f1": val_scores.get("weighted_f1", 0),
                "test_f1": test_scores.get("weighted_f1", 0),
                "dir": run_dir
            })
            
        except Exception as e:
            print(f"Error reading files in {run_dir}: {e}")
            continue

    # Sort by Validation Accuracy
    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS BY VALIDATION ACCURACY")
    print("="*60)
    print(f"{'M':<5} {'L':<5} {'k':<5} | {'Val Acc':<10} {'Test Acc':<10} | {'Val F1':<10} {'Test F1':<10}")
    print("-" * 60)
    
    sorted_by_val = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    for r in sorted_by_val[:5]:
        print(f"{r['M']:<5} {r['L']:<5} {r['k']:<5} | {r['val_acc']:.4f}     {r['test_acc']:.4f}     | {r['val_f1']:.4f}     {r['test_f1']:.4f}")

    # Sort by Test Accuracy
    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS BY TEST ACCURACY")
    print("="*60)
    print(f"{'M':<5} {'L':<5} {'k':<5} | {'Val Acc':<10} {'Test Acc':<10} | {'Val F1':<10} {'Test F1':<10}")
    print("-" * 60)
    
    sorted_by_test = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    for r in sorted_by_test[:5]:
        print(f"{r['M']:<5} {r['L']:<5} {r['k']:<5} | {r['val_acc']:.4f}     {r['test_acc']:.4f}     | {r['val_f1']:.4f}     {r['test_f1']:.4f}")

if __name__ == "__main__":
    analyze_results("/Users/emirhankoc/Desktop/ComputerVision/Spm/save_dataset1")
