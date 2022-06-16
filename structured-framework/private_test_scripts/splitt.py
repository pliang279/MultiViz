import json
import os

def cpjson(id):
    os.system("mkdir simexp/vqa"+str(id))
    def cp(strr):
        os.system('cp visuals/simexp/'+strr+' simexp/vqa'+str(id)+'/')
    fis = open("jsons/vqa"+str(id)+".json",'r').read()
    jis = json.loads(fis)
    cp(jis['instance']['image'])
    for lab in jis['labels']:
        overview = jis['labels'][lab]['overviews']
        features = jis['labels'][lab]['features']
        cp(overview['LIME']['image'])
        cp(overview['LIME']['text'])
        cp(overview['DIME']['image-uni'])
        cp(overview['DIME']['text-uni'])
        cp(overview['DIME']['image-multi'])
        cp(overview['DIME']['text-multi'])
        for featdict in features:
            cp(featdict['forward']['image'])
            cp(featdict['forward']['text'])
            for back in featdict['backward']:
                cp(back['orig'])
                cp(back['image'])
                cp(back['text'])

#for zz in [255,1205,2305,2605,1905,3205,3255,3405,705,2655,855]:
for zz in [905,955,4155,4255,4755,5555,5605,6205,6255,4455,7505]:
    cpjson(zz)


