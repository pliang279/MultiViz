import Vue from 'vue'
import App from './App.vue'
// ElementUI
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
Vue.use(ElementUI);
// Bootstrap
import 'bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'

import _ from 'lodash'
window._ = _
// D3
import * as d3 from 'd3'
window.d3 = d3
// JQuery
import $ from 'jquery'
window.$ = $
// fontawesome
require('../node_modules/@fortawesome/fontawesome-free/css/all.css');
// tabulator
import "../node_modules/tabulator-tables/dist/css/tabulator_simple.min.css"
// video
import 'video.js/dist/video-js.min.css'
Vue.config.productionTip = false

new Vue({
    render: h => h(App),
}).$mount('#app')