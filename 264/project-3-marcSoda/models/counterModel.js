const mongoose = require('mongoose');
const Schema = mongoose.Schema;

//This serves to autoincrement the id fields in Review and Film

//seq is the highest id that has been inserted unless it was inserted through a PUT.
var counterSchema = new Schema({
    _id: {type: String, required: true},
    seq: {type: Number, default: 0}
});
counterSchema.index({ _id: 1, seq: 1 }, { unique: true })
const counterModel = mongoose.model('counter', counterSchema);

//increment the counter and set the doc id to the seq
const autoIncrementModelID = function(modelName, primaryKey, doc, next) {
    counterModel.findByIdAndUpdate(
        modelName,
        { $inc: { seq: 1 } },
        { new: true, upsert: true },
        function(error, counter) {
            if (error) return next(error);
            doc[primaryKey] = counter.seq;
            next();
        }
    );
}

module.exports = autoIncrementModelID;
