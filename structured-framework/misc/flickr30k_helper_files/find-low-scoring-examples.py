import json

# open the json file
with open("structured-framework/all-logit-scores.json", "r") as f:
    all_logit_scores = json.load(f)

# convert json to a map
all_logit_scores = {int(k): v["original_logits"] for k, v in all_logit_scores.items()}

# sort the map
sorted_all_logit_scores = sorted(all_logit_scores.items(), key=lambda x: x[1])

# take top 10 examples
top_15_logit_scores = sorted_all_logit_scores[:15]

print(top_15_logit_scores)
for i in range(15):
    print(top_15_logit_scores[i][0])