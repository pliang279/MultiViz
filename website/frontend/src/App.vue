<template>
  <div id="app" style="width: 100%">
    <el-container>
      <el-header id="systemHeader">MultiViz Website</el-header>
      <el-main id="systemMain">
         <el-row>
            <el-col :span="6" style="padding:2px;">
              <el-container class="viewcontainer">
                <el-header class="viewheader">Control Panel</el-header>
                <el-main>
                  <el-row>
                    <span class="textlabel">Dataset: </span>
                    <el-select v-model="selecteddataset" placeholder="Select">
                      <el-option
                        v-for="d in datasets"
                        :key="d"
                        :label="d"
                        :value="d">
                      </el-option>
                    </el-select>
                  </el-row>
                  <!-- mosei data -->
                  <div class="mosei-data" v-if="selecteddataset.toLowerCase()=='mosei'">
                    <el-row>
                    <span class="textlabel">Instance: </span>
                    <el-select v-model="selectedId" placeholder="Select">
                      <el-option
                        v-for="id in instanceIdxs"
                        :key="id"
                        :label="id"
                        :value="id">
                      </el-option>
                    </el-select>
                  </el-row>
                  <el-row>
                    <el-tag>Video</el-tag>
                    {{meta.dataset}} - {{meta.split}} - {{instanceInfo.video}}
                  </el-row>
                  <el-row>
                    <el-tag>Sentences</el-tag> 
                    {{instanceInfo.script}}
                  </el-row>
                  <el-row>
                    <el-tag style="margin-bottom: 10px;">Original Video</el-tag> 
                    <video id="moseiInstanceVideo" width="100%" controls>
                      <source id="moseiInstanceSource" type="video/mp4">
                    </video>
                  </el-row>
                  <el-row>
                      <el-tag>GT</el-tag> {{instanceInfo["correct-answer"]}} (id: {{instanceInfo["correct-answer-id"]}})
                  </el-row>
                  <el-row>
                      <el-tag>Pred.</el-tag> {{instanceInfo["pred-answer"]}} (id: {{instanceInfo["pred-id"]}})
                    </el-row>
                  </div>
                  <!-- vqa data -->
                  <div class="vqa-data" v-if="selecteddataset.toLowerCase()=='vqa'">
                    <el-row>
                      <span class="textlabel">Instance: </span>
                      <el-select v-model="sVQAInstance" placeholder="Select">
                        <el-option
                          v-for="d in VQAInstances"
                          :key="d"
                          :label="d"
                          :value="d">
                        </el-option>
                      </el-select>
                  </el-row>
                    <el-row>
                      <el-tag>Meta</el-tag>
                      {{meta.dataset}}-{{meta.split}}: instance {{meta.id}}
                    </el-row>
                    <el-row>
                      <el-tag>Question</el-tag>
                       {{instanceInfo.text}}
                    </el-row>
                    <el-row>
                      <el-tag>Image</el-tag>
                      <br/>
                      <el-image
                            :src="instanceUrl"
                            :preview-src-list="[instanceUrl]"
                            fit="contain"></el-image>
                    </el-row>
                    <el-row>
                      <el-tag>GT</el-tag> {{instanceInfo["correct-answer"]}} (id: {{instanceInfo["correct-answer-id"]}})
                    </el-row>
                    <el-row>
                      <el-tag>Pred.</el-tag> {{instanceInfo["pred-answer"]}} (id: {{instanceInfo["pred-id"]}})
                    </el-row>
                  </div>
                  <!-- mimic data -->
                  <div class="mimic-data" v-if="selecteddataset.toLowerCase()=='mimic'">
                    <el-row>
                      <span class="textlabel">Instance: </span>
                      <el-select v-model="smimicInstance" placeholder="Select">
                        <el-option
                          v-for="d in mimicInstances"
                          :key="d"
                          :label="d"
                          :value="d">
                        </el-option>
                      </el-select>
                  </el-row>
                    <el-row>
                      <el-tag>Meta</el-tag>
                      {{meta.dataset}}-{{meta.split}}: instance {{meta.id}}
                    </el-row>
                    <el-row>
                      <el-tag>Static</el-tag>
                       <br/>
                      <el-image
                            :src="staticUrl"
                            :preview-src-list="[staticUrl]"
                            fit="contain"></el-image>
                    </el-row>
                    <el-row>
                      <el-tag>Time Series</el-tag>
                      <br/>
                      <el-image
                            :src="tsUrl"
                            :preview-src-list="[tsUrl]"
                            fit="contain"></el-image>
                    </el-row>
                    <el-row>
                      <el-tag>GT</el-tag> {{instanceInfo["correct-answer"]}} (id: {{instanceInfo["correct-answer-id"]}})
                    </el-row>
                    <el-row>
                      <el-tag>Pred.</el-tag> {{instanceInfo["pred-answer"]}} (id: {{instanceInfo["pred-id"]}})
                    </el-row>
                  </div>
                  <!-- imdb -->
                  <div class="imdb-data" v-if="selecteddataset.toLowerCase()=='imdb'">
                    <el-row>
                      <span class="textlabel">Instance: </span>
                      <el-select v-model="sIMDBInstance" placeholder="Select">
                        <el-option
                          v-for="d in IMDBInstances"
                          :key="d"
                          :label="d"
                          :value="d">
                        </el-option>
                      </el-select>
                  </el-row>
                    <el-row>
                      <el-tag>Meta</el-tag>
                      {{meta.dataset}}-{{meta.split}}: instance {{meta.id}}
                    </el-row>
                    <el-row>
                      <el-tag>Question</el-tag>
                       {{instanceInfo.text}}
                    </el-row>
                    <el-row>
                      <el-tag>Image</el-tag>
                      <br/>
                      <el-image
                            :src="instanceUrl"
                            :preview-src-list="[instanceUrl]"
                            fit="contain"></el-image>
                    </el-row>
                    <el-row>
                      <el-tag>GT</el-tag> {{instanceInfo["correct-answer"]}} (id: {{instanceInfo["correct-id"]}})
                    </el-row>
                    <el-row>
                      <el-tag>Pred.</el-tag> {{instanceInfo["pred-answer"]}} (id: {{instanceInfo["pred-id"]}})
                    </el-row>
                  </div>
                  <!-- Flickr data -->
                  <div class="flickr-data" v-if="selecteddataset.toLowerCase()=='flickr'">
                    <el-row>
                      <span class="textlabel">Instance: </span>
                      <el-select v-model="sFlickrInstance" placeholder="Select">
                        <el-option
                          v-for="d in FlickrInstances"
                          :key="d"
                          :label="d"
                          :value="d">
                        </el-option>
                      </el-select>
                  </el-row>
                    <el-row>
                      <el-tag>Meta</el-tag>
                      {{meta.dataset}}-{{meta.split}}: instance {{meta.id}}
                    </el-row>
                    <el-row>
                      <el-tag>Question</el-tag>
                       {{instanceInfo.text}}
                    </el-row>
                    <el-row>
                      <el-tag>Image</el-tag>
                      <br/>
                      <el-image
                            :src="instanceUrl"
                            :preview-src-list="[instanceUrl]"
                            fit="contain"></el-image>
                    </el-row>
                    <!-- <el-row>
                      <el-tag>GT</el-tag> {{instanceInfo["correct-answer"]}} (id: {{instanceInfo["correct-id"]}})
                    </el-row> -->
                    <el-row>
                      <el-tag>Pred. logits</el-tag> {{instanceInfo["pred-logit"]}}
                    </el-row>
                  </div>
                </el-main>
              </el-container>
              <!-- <ControlPanel /> -->
            </el-col>
            <el-col :span="18" style="padding:2px;">
              <el-container class="viewcontainer">
                <el-header class="viewheader">Main View</el-header>
                <el-main>
                  <el-row>
                    <h6>{{analysisLabel}}</h6>
                  </el-row>
                  <el-col :span="6" :gutter="20">
                    <el-row>
                      <div class="grid-content bg-purple">
                        <div class="block" v-for="(img, imgidx) in images1" :key="img.caption">
                          <div class="explabel" v-show="(analysisLevel=='feature') & (imgidx==0)">Forward Explanation
                            <div class="description"> {{ descs.forward }} </div>
                          </div>
                          <div class="explabel" v-show="(analysisLevel=='feature') & (imgidx==1)">Backward Explanation
                            <div class="description"> {{ descs.backward }} </div>
                          </div>
                          <div v-if="(analysisLevel=='feature') & (imgidx==1) & (selecteddataset.toLowerCase()=='mosei')">
                            <video id="moseiOrigVideo1" width="80%" controls>
                              <source id="moseiOrigSource1" :src="img.video" type="video/mp4">
                            </video>
                            <div class="details">{{img.videoScript}}</div>
                            <div class="demonstration">Backward SLM Original Example1</div>
                            <video id="moseiVideo1" width="80%" controls>
                              <source id="moseiSource1" :src="img.featVideo" type="video/mp4">
                            </video>
                            <div class="demonstration">Backward SLM Video for Feature Analysis</div>
                          </div>
                          <div class="classdesc" v-show="(analysisLevel=='class') & (imgidx==0)"> {{ descs.overviews.LIME }} </div>
                          <!-- insert mosei original video (class level) -->
                          <video id="overviewVideo" width="80%" controls v-if="(imgidx==0) & (selecteddataset.toLowerCase()=='mosei')">
                            <source id="overviewSource" :src="img.video" type="video/mp4">
                          </video>
                          <!-- insert mosei original video (feature level) -->
                          <el-image
                            :src="img.url"
                            :preview-src-list="[img.url]"
                            fit="contain"></el-image>
                            <div class="details" v-show="(analysisLevel=='feature') & (selecteddataset.toLowerCase()=='imdb') & (imgidx==1)">{{ img.origDesc }}</div>
                            <div class="demonstration">{{ img.caption }}</div>
                            <!-- mosei -->
                        </div>
                      </div>
                    </el-row>
                  </el-col>
                  <el-col :span="6" :gutter="20">
                    <el-row>
                      <div class="grid-content bg-purple">
                        <div class="block" v-for="(img, imgidx) in images2" :key="img.caption">
                          <div class="explabel" v-show="(analysisLevel=='feature') & (imgidx==0)">
                            <div class="description"></div>
                          </div>
                          <div class="explabel" v-show="(analysisLevel=='feature') & (imgidx==1)">
                            <div class="description"></div>
                          </div>
                          <div v-if="(analysisLevel=='feature') & (imgidx==1) & (selecteddataset.toLowerCase()=='mosei')">
                            <video id="moseiOrigVideo2" width="80%" controls>
                              <source id="moseiOrigSource2" :src="img.video" type="video/mp4">
                            </video>
                            <div class="details">{{img.videoScript}}</div>
                            <div class="demonstration">Backward SLM Original Example2</div>
                            <video id="moseiVideo2" width="80%" controls>
                              <source id="moseiSource2" :src="img.featVideo" type="video/mp4">
                            </video>
                            <div class="demonstration">Backward SLM Video for Feature Analysis</div>
                          </div>
                          <div class="classdesc" v-show="(analysisLevel=='class') & (imgidx==0)"> {{ descs.overviews.DIME }} </div>
                          <div class="el-image-desc" v-if="(imgidx==0) & (selecteddataset.toLowerCase()=='mosei')">
                          Words: {{ img.words }} <br>
                          Target words: {{ img.targetwords}}
                          </div>
                          <el-image v-if="(selecteddataset.toLowerCase()!='imdb')||(imgidx!=1)||(analysisLevel=='feature')"
                            :src="img.url"
                            :preview-src-list="[img.url]"
                            fit="contain"></el-image>
                            <div class="el-image-desc" v-if="(selecteddataset.toLowerCase()=='imdb')&&(imgidx==1)&&(analysisLevel!='feature')">{{ img.url }}</div>
                            <div class="details" v-show="(analysisLevel=='feature') & (selecteddataset.toLowerCase()=='imdb') & (imgidx==1)">{{ img.origDesc }}</div>
                            <div class="demonstration">{{ img.caption }}</div>
                        </div>
                      </div>
                    </el-row>
                  </el-col>
                  <el-col :span="6" :gutter="20">
                    <el-row>
                      <div class="grid-content bg-purple">
                        <div class="block" v-for="(img, imgidx) in images3" :key="img.caption">
                          <div class="explabel" v-show="(analysisLevel=='feature') & (imgidx==0)">
                            <div class="description"></div>
                          </div>
                          <div class="explabel" v-show="(analysisLevel=='feature') & (imgidx==1)">
                            <div class="description"></div>
                          </div>
                          <div v-if="(analysisLevel=='feature') & (imgidx==1) & (selecteddataset.toLowerCase()=='mosei')">
                            <video id="moseiOrigVideo3" width="80%" controls>
                              <source id="moseiOrigSource3" :src="img.video" type="video/mp4">
                            </video>
                            <div class="details">{{img.videoScript}}</div>
                            <div class="demonstration">Backward SLM Original Example3</div>
                            <video id="moseiVideo3" width="80%" controls>
                              <source id="moseiSource3" :src="img.featVideo" type="video/mp4">
                            </video>
                            <div class="demonstration">Backward SLM Video for Feature Analysis</div>
                          </div>
                          <div class="classdesc" v-show="(analysisLevel=='class') & (imgidx==0)"></div>
                          <el-image v-if="(imgidx==0) & (selecteddataset.toLowerCase()=='mosei')"
                            :src="blankImgUrl"
                            :preview-src-list="[blankImgUrl]"
                            fit="contain"></el-image>
                          <el-image
                            :src="img.url"
                            :preview-src-list="[img.url]"
                            fit="contain"></el-image>
                            <div class="details" v-show="(analysisLevel=='feature') & (selecteddataset.toLowerCase()=='imdb') & (imgidx==1)">{{ img.origDesc }}</div>
                            <div class="demonstration">{{ img.caption }}</div>
                        </div>
                      </div>
                    </el-row>
                  </el-col>
                  <!-- node link graph -->
                  <el-col :span="6" :gutter="20">
                    <el-row>
                      <div class="grid-content bg-purple" style="display: flex; align-items: center;">
                        <svg id="network-graph"></svg>
                      </div>
                    </el-row>
                  </el-col>
                </el-main>
              </el-container>
            </el-col>
        </el-row>
      </el-main>
    </el-container>
  </div>
</template>

<script>
/* global d3 $ _ */
import pipeService from './service/pipeService.js';

export default {
  name: "app",
  components: {
  },
  data() {
    return {
      datasets:["VQA", "MOSEI", "IMDB", "MIMIC", "FLICKR"],
      selecteddataset: "VQA", 
      instances: [],
      // mosei data
      instanceIdxs: ["mosei3", "mosei9", "mosei10"],
      selectedId: "mosei3",
      selectedInstance: {},
      instanceVideoUrl: "",
      table: {},
      // image gallery
      images1: [],
      images2: [],
      images3: [],
      // top-K features
      featK: 5,
      classN: 2,
      // meta data
      meta: {},
      instanceInfo: {},
      instanceUrl: "",
      // analysis level
      analysisLabel: "Overview of class-level analysis",
      analysisLevel: "class",
      // descriptions
      descs: {},
      // selected instanve
      sInstance: "",
      // vqa instances,
      sVQAInstance: "vqa255",
      VQAInstances: ["vqa554", "vqa1205", "vqa255"],
      // imdb instances
      sIMDBInstance: "934",
      IMDBInstances: ["934", "1029", "1427"],
      // flicr instances
      sFlickrInstance: "100",
      FlickrInstances: ["100", "150", "200"],
      // mimic instances
      smimicInstance: "mimic_10",
      mimicInstances: ["mimic_10"],
      blankImgUrl: window.location.href + "results/vqa554/blank.png",
      // static and timeseries urls
      staticUrl: "",
      tsUrl: "",
    };
  },
  watch: {
    // mosei dataset
    selectedId: function(selectedId){
      this.sInstance = selectedId;
      this.drawExpl("mosei");
    },
    selecteddataset: function(selecteddataset){
      if(selecteddataset.toLowerCase()=="mosei"){
        this.sInstance = this.selectedId;
        this.drawExpl("mosei");
      }else if((selecteddataset.toLowerCase()=="vqa")){
        this.sInstance = this.sVQAInstance;
        this.drawExpl("vqa");
      }else if(selecteddataset.toLowerCase()=="imdb"){
        this.sInstance = this.sIMDBInstance;
        this.drawExpl("imdb");
      }else if(selecteddataset.toLowerCase()=="mimic"){
        this.sInstance = this.smimicInstance;
        this.drawExpl("mimic");
      }else if(selecteddataset.toLowerCase()=="flickr"){
        this.sInstance = this.sFlickrInstance;
        this.drawExpl("flickr");
      }
    },
    smimicInstance: function(smimicInstance){
      this.sInstance = smimicInstance;
      this.drawExpl("mimic");
    },
    sVQAInstance: function(sVQAInstance){
      this.sInstance = sVQAInstance;
      this.drawExpl("vqa");
    },
    sIMDBInstance: function(sIMDBInstance){
      this.sInstance = sIMDBInstance;
      this.drawExpl("imdb");
    },
    sFlickrInstance: function(sFlickrInstance){
      this.sInstance = sFlickrInstance;
      this.drawExpl("flickr");
    },
    instanceInfo: function(instanceInfo){
      if(instanceInfo.hasOwnProperty("video")){
        let video = document.getElementById('moseiInstanceVideo');
        let source = document.getElementById('moseiInstanceSource');
        source.setAttribute('src', window.location.href + `results/${this.sInstance}/${instanceInfo.video}`);
        video.load();
      }
    }
  },
  mounted: function () {
    console.log("d3: ", d3); /* eslint-disable-line */
    console.log("$: ", $); /* eslint-disable-line */
    console.log("this: ", this, this.components);
    console.log(
      "_",
      _.partition([1, 2, 3, 4], (n) => n % 2)
    ); /* eslint-disable-line */
    const _this = this;
    if(_this.selecteddataset.toLowerCase()=="mosei"){
      // _this.initializeMosei();
      _this.sInstance = _this.selectedId;
      _this.drawExpl("mosei");
    }else if(_this.selecteddataset.toLowerCase()=="vqa"){
      _this.sInstance = _this.sVQAInstance;
      _this.drawExpl("vqa");
    }else if (_this.selecteddataset.toLowerCase()=="imdb"){
      _this.sInstance = _this.sIMDBInstance;
      _this.drawExpl("imdb");
    }else if (_this.selecteddataset.toLowerCase()=="mimic"){
      _this.sInstance = _this.smimicInstance;
      _this.drawExpl("mimic");
    }else if (_this.selecteddataset.toLowerCase()=="flickr"){
      _this.sInstance = _this.sFlickrInstance;
      _this.drawExpl("flickr");
    }

  },
  methods: {
    seekVideo: function(){
      pipeService.emitSeekFromTime({ 
        start: this.selectedInstance.start, 
        end: this.selectedInstance.end });
    },
    initializeMosei: function(){
      d3.json(window.location.href + `results/${this.selectedId}/${this.selectedId}.json`).then(instance=>{
        this.selectedInstance = instance;
        this.instanceVideoUrl = window.location.href + `results/${this.selectedId}/${instance.instance.video}`
        console.log("this.instance: ", instance);
      });
    },
    setImages: function(level, type, classname, selecteddataset){
      if(level=="class"){
        console.log("type:", type);
        this.analysisLevel="class"
        this.analysisLabel = `Overview of class ${classname} analysis`;
        // `Overview of class-level analysis`;
        // ${classname} analysis`;
        // -------------------------------------------------------------
        // --- handle imdb and vqa
        // -------------------------------------------------------------
        if(selecteddataset.toLowerCase()=="vqa"){
            // arrange images in grids (pred results by default, gt on demand)
            this.images1 = [
              {
                "caption": "Unimodal Lime Image",
                "url": window.location.href + `results/${this.sInstance}/${type.LIME.image}`,
                // `results/${this.sVQAInstance}/${type.LIME.image}`,
                "desc": `${type.LIME.description}`
              },
              {
                "caption": "Unimodal Lime Text",
                "url": window.location.href + `results/${this.sInstance}/${type.LIME.text}`,
                "desc": `${type.LIME.description}`
              }
            ];
            this.images2 = [
            {
              "caption": "Unimodal Dime Image",
              "url": window.location.href + `results/${this.sInstance}/${type.DIME['image-uni']}`,
              "desc": `${type.DIME.description}`
            },
            {
              "caption": "Unimodal Dime Text",
              "url": window.location.href + `results/${this.sInstance}/${type.DIME['text-uni']}`,
              "desc": `${type.DIME.description}`
            }
            ];
            this.images3 = [
                {
                "caption": "Multimodal Dime Image",
                "url": window.location.href + `results/${this.sInstance}/${type.DIME['image-multi']}`,
                "desc": `${type.DIME.description}`
                },
                {
                "caption": "Multimodal Dime Text",
                "url": window.location.href + `results/${this.sInstance}/${type.DIME['text-multi']}`,
                "desc": `${type.DIME.description}`
                }
            ];
          this.descs.overviews = {}
          this.descs.overviews.LIME = type.LIME.description
          this.descs.overviews.DIME = type.DIME.description
        }else if(selecteddataset.toLowerCase()=="imdb"){
          // arrange images in grids (pred results by default, gt on demand)
          this.images1 = [
            {
              "caption": "Unimodal Graident Image",
              "url": window.location.href + `results/${this.sInstance}/${type.UnimodalGradient.image}`,
              "desc": `${type.UnimodalGradient.description}`,
            },
            {
              "caption": "Unimodal Gradient Text",
              "url": window.location.href + `results/${this.sInstance}/${type.UnimodalGradient.text}`,
              "desc": `${type.UnimodalGradient.description}`,
              // "origDesc": `${type.UnimodalGradient.origDesc}`,
            }
          ];
          this.descs.overviews = {}
          this.descs.overviews.LIME = type.UnimodalGradient.description;
          this.descs.overviews.DIME = type.SecondOrderGradient.description
          this.images2 = [
            {
              "caption": "Second-order Gradient Image",
              "url": window.location.href + `results/${this.sInstance}/${type.SecondOrderGradient.image}`,
              "desc": `${type.SecondOrderGradient.description}`
            },
            {
              "caption": "Second-order Gradient Text",
              "url": `${type.SecondOrderGradient.text}`,
              // window.location.href + `results/${this.sInstance}/${type.SecondOrderGradient.image}`,
              "desc": `${type.SecondOrderGradient.description}`
            }
          ];
          this.images3 = [];
        }else if(selecteddataset.toLowerCase()=="mosei"){
          // console.log("mosei text:", type.gradient.description);
          this.descs.overviews = {}
          this.descs.overviews.LIME = type.gradient.description;
          this.descs.overviews.DIME = type.SOG.description;
          this.images1=[
            {
              "caption": "Unimodal Graident Image",
              "url": window.location.href + `results/${this.sInstance}/${type.gradient.vision}`,
              // `results/${this.sVQAInstance}/${type.LIME.image}`,
              "desc": `${type.gradient.description}`,
              "video": window.location.href + `results/${this.sInstance}/${type.gradient.video}`,
              // "origDesc": `${type.UnimodalGradient.origDesc}`,
            },
            {
              "caption": "Unimodal Gradient Text",
              "url": window.location.href + `results/${this.sInstance}/${type.gradient.text}`,
              "desc": `${type.gradient.description}`,
              "video": window.location.href + `results/${this.sInstance}/${type.gradient.video}`,
              // "origDesc": `${type.UnimodalGradient.origDesc}`,
            },
            {
              "caption": "Unimodal Gradient Audio",
              "url": window.location.href + `results/${this.sInstance}/${type.gradient.audio}`,
              "desc": `${type.gradient.description}`,
              "video": window.location.href + `results/${this.sInstance}/${type.gradient.video}`,
              // "origDesc": `${type.UnimodalGradient.origDesc}`,
            },

          ];
          this.images2=[
            {
              "caption": "Second Order Gradient Image",
              "url": window.location.href + `results/${this.sInstance}/${type.SOG.vision}`,
              "desc": `${type.SOG.description}`,
              "targetwords": type.SOG["target-words"],
              "words": type.SOG["words"]
              // "video": window.location.href + `results/${this.sInstance}/${type.gradient.video}`,
              // "origDesc": `${type.UnimodalGradient.origDesc}`,
            },
            {
              "caption": "Second Order Gradient Audio",
              "url": window.location.href + `results/${this.sInstance}/${type.SOG.audio}`,
              "desc": `${type.SOG.description}`,
              "targetwords": type.SOG["target-words"],
              "words": type.SOG["words"]
              // "video": window.location.href + `results/${this.sInstance}/${type.gradient.video}`,
              // "origDesc": `${type.UnimodalGradient.origDesc}`,
            }];
          this.images3=[];
          // orig video 1
          let overviewvideo = document.getElementById('overviewVideo');
          let overviewsource = document.getElementById('overviewSource');
          if(overviewsource){
            overviewsource.setAttribute('src', window.location.href + `results/${this.sInstance}/${type.gradient.video}`);
            overviewvideo.load();
          }
        }else if(selecteddataset.toLowerCase()=="mimic"){
            this.images1 = [
              {
                "caption": "First Order Gradient (Static)",
                "url": window.location.href + `results/${this.sInstance}/${type["First Order Gradient"].static}`,
                // `results/${this.sVQAInstance}/${type.LIME.image}`,
                "desc": `${type["First Order Gradient"].description}`
              },
              {
                "caption": "First Order Gradient (Time Series)",
                "url": window.location.href + `results/${this.sInstance}/${type["First Order Gradient"].timeseries}`,
                "desc": `${type["First Order Gradient"].description}`
              }
            ];
            this.images2 = [];
            this.images3 = [];
            this.descs.overviews = {}
            this.descs.overviews.LIME = type["First Order Gradient"].description
            this.descs.overviews.DIME = ""
        }else if(selecteddataset.toLowerCase()=="flickr"){
          this.analysisLabel = "Overview analysis";
          this.images1 = [
              {
                "caption": "Unimodal Gradient Analysis",
                "url": window.location.href + `results/${this.sInstance}/${type.UnimodalGradient.image}`,
                // `results/${this.sVQAInstance}/${type.LIME.image}`,
                "desc": `${type.UnimodalGradient.description}`
              }
            ];
            let secondImgs = []
            for (const [key, value] of Object.entries(type.SecondOrderGradient)) {
              console.log(key, value);
              secondImgs.push({
                "caption": value.text,
                "url": window.location.href + `results/${this.sInstance}/${value.image}`,
                // `results/${this.sVQAInstance}/${type.LIME.image}`,
                "desc": `${value.description}`
              })
            }
            this.images2 = secondImgs;
            this.images3 = [];
            this.descs.overviews = {}
            this.descs.overviews.LIME = type.UnimodalGradient.description
            this.descs.overviews.DIME = secondImgs[0].desc
        }
        // else if(selecteddataset.toLowerCase()=="mosei"){ }
        // [type.LIME.description, type.DIME.description];
      }else if(level=="feature"){
        console.log("feature level: ", type);
        this.analysisLevel="feature"
        this.analysisLabel = `Feature ${type.id} analysis for class ${classname}`
        // `Feature ${type.id} analysis for class ${classname}`
        if(selecteddataset.toLowerCase()!="mosei"){
          this.images1 = [
            {
              "caption": "Forward SLM Image",
              "url": window.location.href +  `results/${this.sInstance}/${type.forward.image}`,
              // `results/vqa554/vqa-lxmert-sparse-554--image-lime-${type}.png`,
              "desc": `${type["forward-descriptions"]}`,
            },
            {
              "caption": "Backward SLM Original Example1",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[0].orig}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-0.png`
              "desc": `${type["backward-descriptions"]}`,
              "origDesc": selecteddataset.toLowerCase()=="imdb" ? type.backward[0].origDesc :"",
            },
            {
              "caption": "Backward SLM Example1 Image",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[0].image}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-0.png`
              "desc": `${type["backward-descriptions"]}`
            },
            {
              "caption": "Backward SLM Example1 Text",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[0].text}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-0.png`
              "desc": `${type["backward-descriptions"]}`
            }
          ];
          this.images2 = [
            {
              "caption": "Forward SLM Text",
              "url": window.location.href + `results/${this.sInstance}/${type.forward.text}`,
              // `results/vqa554/vqa-lxmert-sparse-554--text-lime-${type}.png`
              "desc": `${type["forward-descriptions"]}`
            },
            {
              "caption": "Backward SLM Original Example2",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[1].orig}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-1.png`
              "desc": `${type["backward-descriptions"]}`,
              "origDesc": selecteddataset.toLowerCase()=="imdb" ? type.backward[1].origDesc :"",
            },
            {
              "caption": "Backward SLM Example2 Image",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[1].image}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-1.png`
              "desc": `${type["backward-descriptions"]}`
            },
            {
              "caption": "Backward SLM Example2 Text",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[1].text}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-1.png`
              "desc": `${type["backward-descriptions"]}`
            }
          ];
          this.images3 = [
            {
              "caption": " ",
              "url": window.location.href + "results/vqa554/blank.png",
              "origDesc": "",
            },
            {
              "caption": "Backward SLM Original Example3",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[2].orig}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-1.png`
              "desc": `${type["backward-descriptions"]}`,
              "origDesc": selecteddataset.toLowerCase()=="imdb" ? type.backward[2].origDesc :"",
            },
            {
              "caption": "Backward SLM Example3 Image",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[2].image}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-2.png`
              "desc": `${type["backward-descriptions"]}`
            },
            {
              "caption": "Backward SLM Example3 Text",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[2].text}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-2.png`
              "desc": `${type["backward-descriptions"]}`
            }
          ];
        }else{
          this.images1 = [
            {
              "caption": "Forward SLM Image",
              "url": window.location.href +  `results/${this.sInstance}/${type.forward.vision}`,
              // `results/vqa554/vqa-lxmert-sparse-554--image-lime-${type}.png`,
              "desc": `${type["forward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.forward.video}`,
            },
            {
              "caption": "Backward SLM Example1 Image",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[0].vision}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-0.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[0]["orig_video"]}`,
              "videoScript": type.backward[0]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[0]["video"]}`,
            },
            {
              "caption": "Backward SLM Example1 Text",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[0].text}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-0.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[0]["orig_video"]}`,
              "videoScript": type.backward[0]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[0]["video"]}`,
            },
            {
              "caption": "Backward SLM Example1 Audio",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[0].audio}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-2.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[0]["orig_video"]}`,
              "videoScript": type.backward[0]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[0]["video"]}`,
            }
          ];
          this.images2 = [
            {
              "caption": "Forward SLM Text",
              "url": window.location.href + `results/${this.sInstance}/${type.forward.text}`,
              // `results/vqa554/vqa-lxmert-sparse-554--text-lime-${type}.png`
              "desc": `${type["forward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.forward.video}`,
            },
            {
              "caption": "Backward SLM Example2 Image",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[1].vision}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-1.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[1]["orig_video"]}`,
              "videoScript": type.backward[1]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[1]["video"]}`,
            },
            {
              "caption": "Backward SLM Example2 Text",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[1].text}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-1.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[1]["orig_video"]}`,
              "videoScript": type.backward[1]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[1]["video"]}`,
            },
            {
              "caption": "Backward SLM Example2 Audio",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[1].audio}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-2.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[1]["orig_video"]}`,
              "videoScript": type.backward[1]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[1]["video"]}`,
            }
          ];
          this.images3 = [
            {
              "caption": "Forward SLM Audio",
              "url": window.location.href + `results/${this.sInstance}/${type.forward.audio}`,
              // `results/vqa554/vqa-lxmert-sparse-554--text-lime-${type}.png`
              "desc": `${type["forward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.forward.video}`,
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[2]["video"]}`,
            },
            {
              "caption": "Backward SLM Example3 Image",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[2].vision}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--image-lime-${type}-2.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[2]["orig_video"]}`,
              "videoScript": type.backward[2]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[2]["video"]}`,
            },
            {
              "caption": "Backward SLM Example3 Text",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[2].text}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-2.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[2]["orig_video"]}`,
              "videoScript": type.backward[2]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[2]["video"]}`,
            },
            {
              "caption": "Backward SLM Example3 Audio",
              "url": window.location.href + `results/${this.sInstance}/${type.backward[2].audio}`,
              // `results/vqa554/vqa-lxmert-sparse-554-sampled--text-lime-${type}-2.png`
              "desc": `${type["backward-descriptions"]}`,
              "video": window.location.href +  `results/${this.sInstance}/${type.backward[2]["orig_video"]}`,
              "videoScript": type.backward[2]["orig_script"],
              "featVideo": window.location.href +  `results/${this.sInstance}/${type.backward[2]["video"]}`,
            }
          ];
          // orig video 1
          let overviewvideo = document.getElementById('overviewVideo');
          let overviewsource = document.getElementById('overviewSource');
          if(overviewsource){
            overviewsource.setAttribute('src', window.location.href +  `results/${this.sInstance}/${type.forward.video}`);
            overviewvideo.load();
          }
          // orig video 1
          let moseiorgvideo1 = document.getElementById('moseiOrigVideo1');
          let moseiorgsource1 = document.getElementById('moseiOrigSource1');
          if(moseiorgsource1){
            moseiorgsource1.setAttribute('src', window.location.href +  `results/${this.sInstance}/${type.backward[0]["orig_video"]}`);
            moseiorgvideo1.load();
          }
          // orig video 2
          let moseiorgvideo2 = document.getElementById('moseiOrigVideo2');
          let moseiorgsource2 = document.getElementById('moseiOrigSource2');
          if(moseiorgsource2){
            moseiorgsource2.setAttribute('src', window.location.href +  `results/${this.sInstance}/${type.backward[1]["orig_video"]}`);
            moseiorgvideo2.load();
          }
          // orig video 3
          let moseiorgvideo3 = document.getElementById('moseiOrigVideo3');
          let moseiorgsource3 = document.getElementById('moseiOrigSource3');
          if(moseiorgsource3){
            moseiorgsource3.setAttribute('src', window.location.href +  `results/${this.sInstance}/${type.backward[2]["orig_video"]}`);
            moseiorgvideo3.load();
          }
          // video 1
          let moseivideo1 = document.getElementById('moseiVideo1');
          let moseisource1 = document.getElementById('moseiSource1');
          if(moseisource1){
            moseisource1.setAttribute('src', window.location.href +  `results/${this.sInstance}/${type.backward[0].video}`);
            moseivideo1.load();
          }
          // orig video 2
          let moseivideo2 = document.getElementById('moseiVideo2');
          let moseisource2 = document.getElementById('moseiSource2');
          if(moseisource2){
            moseisource2.setAttribute('src', window.location.href +  `results/${this.sInstance}/${type.backward[1].video}`);
            moseivideo2.load();
          }
          // orig video 3
          let moseivideo3 = document.getElementById('moseiVideo3');
          let moseisource3 = document.getElementById('moseiSource3');
          if(moseisource3){
            moseisource3.setAttribute('src', window.location.href +  `results/${this.sInstance}/${type.backward[2].video}`);
            moseivideo3.load();
          }

        }
        this.descs.forward = type["forward-descriptions"];
        this.descs.backward = type["backward-descriptions"];
      }
    },
    drawExpl: function(sType){
      const _this = this;
      if(sType.toLowerCase()=="vqa"){
        _this.sInstance = _this.sVQAInstance;
      }else if(sType.toLowerCase()=="imdb"){
        _this.sInstance = _this.sIMDBInstance;
      }else if (sType.toLowerCase()=="mosei"){
        _this.sInstance = _this.selectedId;
        // _this.instanceUrl = window.location.href +  `results/${this.sInstance}/${_this.instanceInfo.video}`
      }else if(sType.toLowerCase()=="mimic"){
        _this.sInstance = _this.smimicInstance;
      }else if(sType.toLowerCase()=="flickr"){
        _this.sInstance = _this.sFlickrInstance;
      }
      d3.json(window.location.href + `results/${_this.sInstance}/${_this.sInstance}.json`).then((vdata)=>{
        // set meta data and instance info
        _this.meta = vdata.metadata;
        _this.instanceInfo = vdata.instance;
        _this.selectedInstance = {"url": window.location.href + `results/${_this.sInstance}/${_this.instanceInfo.video}`};
        console.log("instance Info: ", _this.instanceInfo, sType, window.location.href + `results/${_this.sInstance}/${_this.instanceInfo.video}`);
        if((sType.toLowerCase()!="mosei") & (sType.toLowerCase()!="mimic")){
          _this.instanceUrl = window.location.href + `results/${_this.sInstance}/${_this.instanceInfo.image}`;
        }else if(sType.toLowerCase()=="mimic"){
          _this.staticUrl = window.location.href + `results/${_this.sInstance}/${_this.instanceInfo.static}`;
          _this.tsUrl = window.location.href + `results/${_this.sInstance}/${_this.instanceInfo.timeseries}`;
        }
        let labels = vdata.labels;
        
        if((sType.toLowerCase()=="vqa")||(sType.toLowerCase()=="imdb")||(sType.toLowerCase()=="mosei")){
          // prepare test data
          let classN = _this.classN, featK = _this.featK, netData = {};
          let classes = Object.keys(labels);
          classes.map(cname=>{
            netData[cname] = []
            labels[cname].features.map(fdata=>{
              netData[cname].push([`feat${fdata.id}`, fdata.weight]);
            });
            netData[cname].sort((a,b)=>{return b[1] - a[1]});
          })
          console.log("vqa data: ", vdata, netData);

          // draw network graph
          let nodeH = 30, nodePadd = 10, nodesize = 10;
          let svgwidth = $(".grid-content").width(), 
          svgheight = (nodeH + nodePadd) * (Math.max(featK, classN)+2),
          gwidth = svgwidth - 2*nodePadd,
          dheight = svgheight - 2*nodePadd;
          d3.select("#network-graph").selectAll("*").remove();
          let netsvg = d3.select("#network-graph")
          // .attr("width", svgwidth)
          // .attr("height", svgheight)
          .attr("viewBox", `0 0 ${svgwidth} ${svgheight}`)
          .append("g")
          .attr("transform", `translate(${nodePadd}, ${nodePadd})`)

          let leftX = nodePadd + 10, rightX = gwidth - 10;
          // draw column labels
          netsvg.append("text").attr("class", "nodelabel")
            .attr("x", leftX)  
            .attr("y", nodePadd)  
            .attr("dx", -30)
            .text("Features");
          netsvg.append("text").attr("class", "nodelabel")
            .attr("x", rightX)  
            .attr("y", nodePadd)  
            .attr("dx", -30)
            .text("Classes");
          
          // draw links
          let linkg = netsvg.append("g").attr("class", "linkg")
          Object.entries(netData).forEach(([cname, cdata], cidx) => {
            let x0 = leftX, 
            y0 = (cidx+1) * (nodeH + nodePadd) + nodePadd;
            Object.entries(cdata).forEach(([fname, fdata], fidx) => {
              console.log([cname, cdata, fname, fdata]);
                let x1 = rightX, 
                y1 = (fidx+1) * (nodeH + nodePadd) + nodePadd;
                let strokeWidth = fdata[1]*10;
                if(sType.toLowerCase()=="imdb"){
                  strokeWidth = fdata[1];
                }
                linkg.append("line")
                  .attr("class", `node_link from_${cname} to_${fdata[0]}`)
                  .attr("x1",x1)  
                  .attr("y1",y0)  
                  .attr("x2",x0)  
                  .attr("y2",y1)  
                  .attr("stroke-width",strokeWidth); 
                if(fdata[1].toFixed(2)>0.00){
                  linkg.append("text")
                  .attr("class",`link_label from_${cname} to_${fdata[0]}`)
                  .attr("x",(x1+x0)/2)  
                  .attr("y",(y0+y1)/2)  
                  .attr("dx", -10)
                  .text(fdata[1].toFixed(2));
                }
            });
          });
          let classnames = Object.keys(netData)
          // featnames = netData[classnames[0]];
          // Object.keys(netData[classnames[0]]);
          console.log("class names", classnames);
          // draw class nodes
          classnames.map((cname, cidx)=>{
          console.log("draw class nodes: ", cname);
          let classNode = netsvg.append("g")
            .attr("class", "class_g")
            .attr("transform", `translate(${rightX}, ${ (cidx+1) * (nodeH + nodePadd) + nodePadd})`)
          classNode.append("circle")
              .attr("class", "class_node")
              .attr('cx', 0)
              .attr('cy', 0)
              .attr('r', nodesize)
              .on("mouseover", function(){
                $(".class_node").attr("class","class_node_dehighlight")
                d3.select(this).attr("class", "class_node");
                $(".node_link").addClass("node_link_dehighlight");
                $(".link_label").addClass("node_link_dehighlight");
                $(`.from_${cname}`).removeClass("node_link_dehighlight");
                // hide feature nodes
                $(".feat_g").hide();
                $(`.feat_${cname}_g`).show();
              })
              // .on("mouseout", function(){
              //   $(".class_node_dehighlight").attr("class","class_node");
              //   $(".node_link").removeClass("node_link_dehighlight");
              //   $(".link_label").removeClass("node_link_dehighlight");
              // })
              // click event on class nodes
              .on("click", function(){
                console.log("cname", cname);
                _this.setImages("class", labels[cname].overviews, `${cname}:${_this.meta.labels[cname]}`, _this.selecteddataset);
              })
              ;
            classNode.append("text")
                .attr("class","nodelabel")
                .attr("dx", -55)
                .text(`${cname}:${_this.meta.labels[cname]}`);

            // draw feature nodes
            let featnames = netData[cname];
            featnames.map((fname, fidx)=>{
              let featNode = netsvg.append("g")
                    .attr("class", `feat_g feat_${cname}_g`)
                    .attr("transform", `translate(${leftX}, ${(fidx+1) * (nodeH + nodePadd) + nodePadd})`)
                  featNode.append("circle")
                      .attr("class", `feat_node class_${cname}_feat`)
                      .attr('cx', 0)
                      .attr('cy', 0)
                      .attr('r', nodesize)
                      .on("mouseover", function(){
                        $(".feat_node").attr("class","feat_node_dehighlight")
                        d3.select(this).attr("class", "feat_node");
                        $(".node_link").addClass("node_link_dehighlight");
                        $(".link_label").addClass("node_link_dehighlight");
                        $(`.to_${fname[0]}`).removeClass("node_link_dehighlight");
                      })
                      .on("mouseout", function(){
                        $(".feat_node_dehighlight").attr("class","feat_node");
                        $(`.from_${cname}`).removeClass("node_link_dehighlight");
                        // $(".node_link").removeClass("node_link_dehighlight");
                        // $(".link_label").removeClass("node_link_dehighlight");
                      })
                      // click event on feature nodes
                      .on("click", function(){
                        let all_feats = labels[cname].features,
                        curr_feat = fname[0].replace("feat","");
                        let selected_feat = all_feats.filter(f=>{return f.id.toString()==curr_feat});
                        // console.log("current feature:", selected_feat, curr_feat);
                        _this.setImages("feature", selected_feat[0], `${cname}:${_this.meta.labels[cname]}`, _this.selecteddataset);
                      })
                      ;
                  featNode.append("text")
                      .attr("class","nodelabel")
                      .attr("dx", -50)
                      .text(fname[0]);
              });
        
        });

          // --- set default style
          // set image urls
          _this.setImages("class", labels[classes[0]].overviews, `${classes[0]}:${_this.meta.labels[classes[0]]}`, _this.selecteddataset);
          // set default highlighting
          $(".class_node").attr("class","class_node_dehighlight")
          d3.select($(".class_node_dehighlight")[0]).attr("class", "class_node");
          $(".node_link").addClass("node_link_dehighlight");
          $(".link_label").addClass("node_link_dehighlight");
          $(`.from_${classes[0]}`).removeClass("node_link_dehighlight");
          // hide feature nodes
          $(".feat_g").hide();
          $(`.feat_${classes[0]}_g`).show();
        }
        else if(sType.toLowerCase()=="mimic") {
            // set image urls
            let classes = Object.keys(labels);
            d3.select("#network-graph").selectAll("*").remove();
            console.log("mimic: ", labels[classes[0]].overviews, `${classes[0]}:${_this.meta.labels[classes[0]]}`, _this.selecteddataset);
            _this.setImages("class", labels[classes[0]].overviews, `${classes[0]}:${_this.meta.labels[classes[0]]}`, _this.selecteddataset);
        }else if(sType.toLowerCase()=="flickr"){
            let classes = Object.keys(labels);
            console.log("flickr: ", labels, classes[0], _this.meta, _this.meta.labels);
            d3.select("#network-graph").selectAll("*").remove();
            _this.setImages("class", labels[classes[0]].overviews, `${classes[0]}:${_this.meta.labels}`, _this.selecteddataset);
        }

        });

    
    }
  }
};
</script>

<style>
#app {
  font-family: "Avenir", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  width: 100%;
  margin: 0 auto;
}

#systemHeader {
  background-color: #2c3e50;
  color: white;
  text-align: start;
  font-weight: bold;
  font-size: 25px;
  line-height: 60px;
  height: 8vh;
}

#systemMain {
  padding: 5px;
  /* background-color: #E9EEF3; */
  color: #333;
  text-align: left;
  font-size: 14px;
}

.viewheader {
  font-weight: bold;
  text-align: start;
  padding: 5px;
  height: 40px !important;
  line-height: 40px;
  background: #e5e9f2;
}

.viewcontainer {
  border: 1px solid whitesmoke;
  height: 88vh;
}

.el-row {
  margin-bottom: 5px;
}

.grid-content {
  border-radius: 4px;
  min-height: 36px;
  height: 76vh;
  text-align: center;
  padding: 10px;
  /* background: #d3dce6; */
}

.el-image {
  width: 24vh;
  height: 24vh;
  /* width: 170px;
  height: 170px; */
}

video {
  height: 24vh
}

.el-image-desc {
  width: 24vh;
  height: 24vh;
  text-align: right;
  justify-content: center;
  align-items: center;
  display: flex;
  /* width: 170px;
  height: 170px; */
}

.demonstration {
  font-size: 13px;
  /* margin-top: -2vh; */
  min-height: 3vh;
}
.description {
  font-size:12px;
  font-weight: normal;
  /* min-height: 12vh; */
  /* white-space: nowrap; */
  width: 600px;
}

.classdesc {
  font-size: 12px;
  text-align: start;
  min-height: 70px;
  display: flex;
  align-items: center;
}

.explabel {
  font-size: 14px;
  font-weight: bold;
  min-height: 12vh;
  margin-top: 2vh;
  margin-bottom: -1vh;
  text-align: left;
}

.textlabel {
  font-size: 15px;
  display: inline-block;
  min-width: 70px;
}

.class_node {
  stroke: none;
  fill: #69a3b2;
}
.class_node_dehighlight {
  stroke: none;
  fill: #69a3b2;
  opacity:0.5;
}
.nodelabel {
  font-size:12px;
}
.link_label {
  font-size: 12px;
}

.feat_node {
  stroke: none;
  fill: #69b3a2;
  text-anchor: start;
}
.feat_node_dehighlight {
  stroke: none;
  fill: #69b3a2;
  fill-opacity:0.5;
}
.node_link {
  stroke: gray;
  stroke-opacity: 0.7;
}

.node_link_dehighlight {
  stroke-opacity: 0.2;
  opacity: 0.0;
}

.details {
  height: 20vh;
  overflow-y: scroll;
  margin-bottom: 10px;
  border: 1px solid lightgray;
  font-size: 12px;
}

/* video detail */
.videoPlayerContainer {
    height: 225px;
}

.my-player {
    position: absolute;
    height: 100%;
    width: 100%;
}

.labelsContainer {
    position: absolute;
    top: 0px;
    height: 100%;
    width: 100%;
    pointer-events: none;
}
</style>