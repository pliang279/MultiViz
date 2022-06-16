import os
import sys
import PIL.Image as Image
sys.path.insert(1, os.getcwd())

#from models.clevr_cnnlstmsa import CLEVRCNNLSTMSA
from models.clevr_mdetr import CLEVRMDETR
from datasets.clevr import CLEVRDataset
from visualizations.visualizegradient import *

feature_idxs_1 = [15, 26, 65, 75, 88, 90, 129, 159, 189, 210, 221, 246, 301, 350, 407, 424, 432, 438, 466, 483]
feature_idxs_2 = [i*2+i%2 for i in feature_idxs_1]
print(feature_idxs_2)
feature_idxs = feature_idxs_2
data_idxs = [i for i in range(10000)]

for feat in feature_idxs:
    if not os.path.isdir('private_test_scripts/clevrmdetr_simexp/feat' + str(feat)):
        os.mkdir('private_test_scripts/clevrmdetr_simexp/feat' + str(feat))

def generate_all(dataset, model):
    data_instances = [dataset.getdata(i) for i in data_idxs]
    forwards = [model.forward(data_instance) for data_instance in data_instances]
    prelinears = [model.getprelinear(resobj) for resobj in forwards]

    for feat in feature_idxs:
        print(feat)
        sorted_prelinear_idxs = sorted(range(len(prelinears)), key=lambda x: prelinears[x][feat])
        max3 = sorted_prelinear_idxs[len(prelinears)-3:]
        min3 = sorted_prelinear_idxs[:3]
        outdir = 'private_test_scripts/clevrmdetr_simexp/feat' + str(feat) + '/'
        os.system('rm ' + outdir + 'grad*')

        for i in max3:
            data_instance = data_instances[i]
            data_idx = data_idxs[i]
            imgfile = data_instance[0]
            question = data_instance[1]
            img = Image.open(imgfile)
            _, grad, _ = model.getgrad(data_instance, feat, prelinear=True)
            res, parsed, _, _ = model.getgradtext(data_instance, feat, prelinear=True)

            img.save(outdir + 'max_orig_' + str(data_idx) + '.png')
            with open(outdir + 'max_question_' + str(data_idx) + '.txt', 'w') as outfile:
                outfile.write(question)

            grads = torch.sum(torch.abs(grad).squeeze(), dim=0)
            t = normalize255(grads)
            heatmap2d(t, outdir + 'max_grad_' + str(data_idx) + '.png', imgfile)  
            textmap(parsed, torch.abs(res), outdir + 'max_textgrad_' + str(data_idx) + '.png')


        for i in min3:
            data_instance = data_instances[i]
            data_idx = data_idxs[i]
            imgfile = data_instance[0]
            question = data_instance[1]
            img = Image.open(imgfile)
            _, grad, _ = model.getgrad(data_instance, feat, prelinear=True)
            res, parsed, _, _ = model.getgradtext(data_instance, feat, prelinear=True)

            img.save(outdir + 'min_orig_' + str(data_idx) + '.png')
            with open(outdir + 'min_question_' + str(data_idx) + '.txt', 'w') as outfile:
                outfile.write(question)        

            grads = torch.sum(torch.abs(grad).squeeze(), dim=0)
            t = normalize255(grads)
            heatmap2d(t, outdir + 'min_grad_' + str(data_idx) + '.png', imgfile)   
            textmap(parsed, torch.abs(res), outdir + 'min_textgrad_' + str(data_idx) + '.png')

if __name__ == '__main__':
    dataset = CLEVRDataset()
    #model = CLEVRCNNLSTMSA()
    model = CLEVRMDETR()
    generate_all(dataset, model)