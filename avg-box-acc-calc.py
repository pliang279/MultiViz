import json
import numpy as np
# Read Key to Logits JSON
with open("structured-framework/key_to_logits-box-acc.json") as f:
    key_to_logits = json.loads(f.read())


dg_acc = []
rd_acc = []
og_drops = []
rd_drops = []
gt_drops = []

for instance_idx, inst_val in key_to_logits.items():
    for key, key_val in inst_val.items():
        num_dg_matching_boxes = key_val["num_dg_matching_boxes"]
        num_random_matching_boxes = key_val["num_random_matching_boxes"]
        num_gt_boxes = key_val["num_gt_boxes"]
        if num_gt_boxes != 0:
            dg_acc.append(num_dg_matching_boxes / num_gt_boxes)
            rd_acc.append(num_random_matching_boxes / num_gt_boxes)
            og_drops.append(key_val["doublegrad_box_logits"] - key_val["original_logits"])
            rd_drops.append(key_val["random_box_logits"] - key_val["original_logits"])
            gt_drops.append(key_val["ground_truth_logits"] - key_val["original_logits"])



# Calculate mean and std of acc
dg_acc_mean = np.mean(dg_acc)
dg_acc_std = np.std(dg_acc)
rd_acc_mean = np.mean(rd_acc)
rd_acc_std = np.std(rd_acc)


# Calculate mean and std of drops
og_drops_mean = np.mean(og_drops)
og_drops_std = np.std(og_drops)
rd_drops_mean = np.mean(rd_drops)
rd_drops_std = np.std(rd_drops)
gt_drops_mean = np.mean(gt_drops)
gt_drops_std = np.std(gt_drops)

print("DG Acc Mean: {}".format(dg_acc_mean))
print("DG Acc Std: {}".format(dg_acc_std))
print("RD Acc Mean: {}".format(rd_acc_mean))
print("RD Acc Std: {}".format(rd_acc_std))

print("DG Drops Mean: {}".format(og_drops_mean))
print("DG Drops Std: {}".format(og_drops_std))
print("RD Drops Mean: {}".format(rd_drops_mean))
print("RD Drops Std: {}".format(rd_drops_std))
print("GT Drops Mean: {}".format(gt_drops_mean))
print("GT Drops Std: {}".format(gt_drops_std))