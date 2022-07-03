import json
import numpy as np
# Read Key to Logits JSON
with open("structured-framework/key_to_logits.json") as f:
    key_to_logits = json.loads(f.read())


dg_overlap = []
rd_overlap = []
og_drops = []
rd_drops = []
gt_drops = []

for instance_idx, inst_val in key_to_logits.items():
    for key, key_val in inst_val.items():
        gt_dg_overlap = key_val["gt_dg_overlap"]
        gt_rd_overlap = key_val["gt_rd_overlap"]
        gt_pixels = key_val["gt_pixels"]
        if gt_pixels != 0:
            dg_overlap.append(gt_dg_overlap / gt_pixels)
            rd_overlap.append(gt_rd_overlap / gt_pixels)
            og_drops.append(key_val["doublegrad_logits"] - key_val["original_logits"])
            rd_drops.append(key_val["random_drop_logits"] - key_val["original_logits"])
            gt_drops.append(key_val["ground_truth_logits"] - key_val["original_logits"])



# Calculate mean and std of overlap
dg_overlap_mean = np.mean(dg_overlap)
dg_overlap_std = np.std(dg_overlap)
rd_overlap_mean = np.mean(rd_overlap)
rd_overlap_std = np.std(rd_overlap)

# Calculate mean and std of drops
og_drops_mean = np.mean(og_drops)
og_drops_std = np.std(og_drops)
rd_drops_mean = np.mean(rd_drops)
rd_drops_std = np.std(rd_drops)
gt_drops_mean = np.mean(gt_drops)
gt_drops_std = np.std(gt_drops)

print("DG Overlap Mean: {}".format(dg_overlap_mean))
print("DG Overlap Std: {}".format(dg_overlap_std))
print("RD Overlap Mean: {}".format(rd_overlap_mean))
print("RD Overlap Std: {}".format(rd_overlap_std))

print("DG Drops Mean: {}".format(og_drops_mean))
print("DG Drops Std: {}".format(og_drops_std))
print("RD Drops Mean: {}".format(rd_drops_mean))
print("RD Drops Std: {}".format(rd_drops_std))
print("GT Drops Mean: {}".format(gt_drops_mean))
print("GT Drops Std: {}".format(gt_drops_std))