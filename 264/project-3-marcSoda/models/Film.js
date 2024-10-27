const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const autoIncrementModelID = require('./counterModel')
const Review = require('./Review')

//Define film schema
const filmSchema = new Schema({
    FilmID: {
        type: Number,
        unique: true,
        min: 1
    },
    Title: {
        type: String,
        required: true
    },
    Body: {
        type: String,
        required: true
    },
    Date: {
        type: Date,
        required: true,
        default: Date.now
    },
    Reviews: [{
        type: Number,
        ref: 'Review'
    }]
});

//auto increment the ID
filmSchema.pre('save', function(next) {
    if (!this.isNew) {
        next();
        return;
    }
    autoIncrementModelID('films', 'FilmID', this, next)
});

const Film = mongoose.model('Film', filmSchema);
module.exports = Film;
