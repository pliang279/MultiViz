import matplotlib.pyplot as plt
import torch
from skimage.segmentation import mark_boundaries
def visualizelime(explanation,modalitytype,label,savedir = 'visuals/lime.png',show=False,num_features = 10, hide_rest = False,positive_only=False):
    plt.clf()
    if modalitytype == 'image':
        temp, mask = explanation.get_image_and_mask(label, positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)
        img_boundry2 = mark_boundaries(temp/255.0, mask)
        plt.imshow(img_boundry2)
    elif modalitytype == 'text' or modalitytype == 'tabular':
        text_fig = explanation.as_pyplot_figure(label=label)
        #plt.imshow(text_fig)
    elif modalitytype == 'timeseries':
        explanatio = explanation[str(label)]
        exps = explanatio[0]
        framelength = explanatio[1]
        locs = []
        vals = []
        colors = []
        for a,b in exps[1]:
            locs.append(a*framelength)
            vals.append(b)
            if b > 0:
                colors.append('green')
            else:
                colors.append('red')
        _ = plt.bar(locs,vals,color=colors,width=framelength,align='edge')

    elif modalitytype == 'timeseriesC':
        exps = explanation[str(label)]
        locs = []
        vals = []
        colors = []
        for a,b in exps[1]:
            locs.append(a)
            vals.append(b)
            if b > 0:
                colors.append('green')
            else:
                colors.append('red')
        _ = plt.bar(locs,vals,color=colors,width=0.7)
    else:
        raise ValueError
    if show:
        plt.show()
    plt.savefig(savedir)
    plt.clf()

        



