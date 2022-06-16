import torch
from tqdm import tqdm
device='cuda:0'
saved = torch.load('prepval3.pt')
def vallxmert(lxmert):
    totals=0
    corrects=0
    bluetotals=0
    bluecorrects=0
    bluelist =torch.load('bluelist.pt') 
    for id,normalized_boxes,features,inputs,correct in tqdm(saved):
        if correct is None:
            continue
        output = lxmert(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            visual_feats=features.to(device),
            visual_pos=normalized_boxes.to(device),
            token_type_ids=inputs.token_type_ids.to(device),
            output_attentions=False,
        )
        pred = torch.argmax(output['question_answering_score'][0])
        if id in bluelist:
            bluetotals += 1
            if pred.item() == correct:
                bluecorrects += 1
        if pred.item() == correct:
            corrects += 1
        totals += 1
    return corrects,totals,bluecorrects,bluetotals



if __name__ == '__main__':


    from transformers import LxmertForQuestionAnswering

    #model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to(device)
    model = torch.load('debugging0.pt').to(device)


    print(vallxmert(model))


