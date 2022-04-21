import json

outfile=open('vqa554.json','w+')
ids=[
        [6115,  554, 4272],
        [6788, 9383, 3321],
        [1162, 5238, 9702],
        [8217, 1031, 8996],
        [578, 3979, 3920]]
weights=[0.1620,0.0701,0.2832,0.1663,0.0]

metadata = {'dataset':'VQA','split':'val','id':554}
instance = {'image':'vqa-lxmert-554-origimage.png','text':'What is the table made of?','correct-answer':'glass','correct-answer-id':2353,'pred-answer':'glass','pred-id':2353}

predlimes = {'image':'vqa-lxmert-554-image-lime-pred.png','text':'vqa-lxmert-554-text-lime-pred.png'}
preddimes = {'image-uni':'vqa-lxmert-554-image-dime-pred-uni.png','text-uni':'vqa-lxmert-554-text-dime-pred-uni.png','image-multi':'vqa-lxmert-554-image-dime-pred-multi.png','text-multi':'vqa-lxmert-554-text-dime-pred-multi.png'}
overviews = {'LIME':predlimes,'DIME':preddimes}

features = []
featids=[83,569,1134,1278,1535]
for i in range(5):
    ii = featids[i]
    examples=[]
    for j in range(3):
        example={'id':ids[i][j],'orig':'vqa-val-'+str(ids[i][j])+'.png','image':'vqa-lxmert-sparse-554-sampled-image-lime-feat'+str(ii)+'-'+str(j)+'.png','text':'vqa-lxmert-sparse-554-sampled-text-lime-feat'+str(ii)+'-'+str(j)+'.png'}
        examples.append(example)
    forwards={'image':'vqa-lxmert-sparse-554--image-lime-feat'+str(ii)+'.png','text':'vqa-lxmert-sparse-554--text-lime-feat'+str(ii)+'.png'}
    feat={'id':i,'weight':weights[i],'forward':forwards,'backward':examples}
    features.append(feat)

label2353 = {'overviews':overviews,'features':features}

alls={'metadata':metadata,'instance':instance,'labels':{2353:label2353}}
json.dump(alls,outfile,indent=4)
