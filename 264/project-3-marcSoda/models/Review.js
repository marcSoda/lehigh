const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const autoIncrementModelID = require('./counterModel')

//define review schema
const reviewSchema = new Schema({
    ReviewID: {
        type: Number,
        unique: true,
        min: 1,
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
});

//autoincrement the ID
reviewSchema.pre('save', function(next) {
    if (!this.isNew) {
        next();
        return;
    }
    autoIncrementModelID('reviews', 'ReviewID', this, next)
});

const Review = mongoose.model('Review', reviewSchema);
module.exports = Review;
