import pandas as pd

def check_splits():
    try:
        train_df = pd.read_parquet('data/splits/proper_train.parquet')
        calib_df = pd.read_parquet('data/splits/calibration.parquet')
        test_df = pd.read_parquet('data/splits/test.parquet')
        
        train_urls = set(train_df['url'])
        calib_urls = set(calib_df['url'])
        test_urls = set(test_df['url'])
        
        report = []
        report.append("# Phase 12 — Split Integrity\n")
        report.append("## Sizes")
        report.append(f"- proper_train: {len(train_urls)}")
        report.append(f"- calibration: {len(calib_urls)}")
        report.append(f"- test: {len(test_urls)}\n")
        
        report.append("## Disjointness Checks")
        report.append(f"- TRAIN ∩ CALIBRATION = {len(train_urls.intersection(calib_urls))}")
        report.append(f"- CALIBRATION ∩ TEST = {len(calib_urls.intersection(test_urls))}")
        report.append(f"- TRAIN ∩ TEST = {len(train_urls.intersection(test_urls))}\n")
        
        if len(train_urls) != len(train_df):
            report.append(f"WARNING: proper_train has duplicates! ({len(train_df)} rows, {len(train_urls)} unique URLs)")
        if len(calib_urls) != len(calib_df):
            report.append(f"WARNING: calibration has duplicates! ({len(calib_df)} rows, {len(calib_urls)} unique URLs)")
        if len(test_urls) != len(test_df):
            report.append(f"WARNING: test has duplicates! ({len(test_df)} rows, {len(test_urls)} unique URLs)")
            
        report.append("## Conclusion")
        if len(train_urls.intersection(calib_urls)) == 0 and len(calib_urls.intersection(test_urls)) == 0 and len(train_urls.intersection(test_urls)) == 0:
            report.append("All partitions are disjoint. We will use `data/splits/calibration.parquet` as the canonical ICP calibration set.")
        else:
            report.append("There is an intersection. This needs to be resolved.")
            
        with open('reports/phase12_split_integrity.md', 'w') as f:
            f.write("\n".join(report) + "\n")
            
        print("Split integrity check complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    check_splits()
