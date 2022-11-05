import Vue from 'vue'

var pipeService = new Vue({
    data: {
        VIDEOEVENT: 'video_event',
        VIDEOSEEKING: "video_seeking"
    },
    methods: {
        emitVideoEvent: function(msg) {
            this.$emit(this.VIDEOEVENT, msg)
        },
        onVideoEvent: function(callback) {
            this.$on(this.VIDEOEVENT, function(msg) {
                callback(msg)
            })
        },
        emitSeekFromTime: function(msg) {
            this.$emit(this.VIDEOSEEKING, msg)
        },
        onSeekFromTime: function(callback) {
            this.$on(this.VIDEOSEEKING, function(msg) {
                callback(msg)
            })
        },
    }
})
export default pipeService