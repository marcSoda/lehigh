const express = require('express');
const Film = require('../models/Film.js');
const Review = require('../models/Review.js');
const router = express.Router();


//get all films or get all films matching the search query if it exists
router.get('/', async function(req, res, next) {
    let query = req.query.search; //get search from url
    let docs
    //get all reviews
    try { docs = await Film.find(); }
    catch (err) { next(err); }
    //if query defined, run the query and return the results. else send all films
    if (query !== undefined) {
        //result contains all films that contain the query string in the title or body field
        let result = docs.filter(function(el) {
            if (el.Title.includes(query) || el.Body.includes(query)) return el;
        });
        res.send(result);
        return;
    }
    //send all films
    res.send(docs);
});

//Post film
router.post('/', async (req, res, next) => {
    const body = req.body;
    const keys = Object.keys(body); //keys in body
    //ensure valid payload
    if (!keys.includes("Title") || !keys.includes("Body")) {
        res.status(400).send("Invalid JSON Payload");
        return;
    }
    //create film. this function is called recursively if the auto-incremented FilmID already exists because of a PUT
    async function create() {
        try {
            const doc = await Film.create({
                "Title": body.Title,
                "Body": body.Body,
            });
            res.send(doc);
        } catch (err) {
            //if there is a duplicate ID error
            if (err.code === 11000) return create()
            next(err);
        }
    }
    return create();
});

//PUT single film.
//If FilmID does not exist and JSON payload is valid, create new document
//Otherwise send 400
router.put('/:fid', async (req, res, next) => {
    const fid = req.params.fid;      //FilmID
    const body = req.body;           //body
    const keys = Object.keys(body);  //keys in body
    let upsert = false;              //wether or not film is eligible to be upserted
    //payload must include required fields or it will not upsert
    if (keys.includes("Title") && keys.includes("Body")) {
        upsert = true;
    }
    //update doc if it exists. else, upsert if upsert is set
    try {
        let raw = await Film.findOneAndUpdate({
            FilmID: fid
        }, {
            "Title": body.Title,
            "Body": body.Body,
            "Date": body.Date,
        }, {
            new: true,
            upsert: upsert,
            rawResult: true
        });
        //send 400 if FilmID not present and Payload Invalid for an upsert
        if (raw.value === null) {
            res.status(400).send("Nonexistant FilmID and Invalid JSON Payload for POST");
            return;
        }
        //send upserted/updated film
        res.send(raw.value);
    } catch(err) {
        next(err);
    }
});

//GET film if exists
router.get('/:fid', getFilm, (req, res) => {
    res.send(res.film);
});

//delete film if exists
router.delete('/:fid', getFilm, async (req, res) => {
    let film = res.film;
    let revs = film.Reviews;
    try {
        //delete all reviews associated with the film
        for (const revID of revs) {
            await Review.deleteOne({ ReviewID: Number(revID) });
        }
        await res.film.remove();
        res.send("Successfully deleted Film of ID: " + req.params.fid);
    } catch (err) {
        res.status(500).send("Error deleting Film: " + err.message);
    }
});

//Post review
router.post('/:fid/reviews', getFilm, async (req, res, next) => {
    const body = req.body;
    const keys = Object.keys(body); //keys in body
    let film = res.film;
    //ensure valid payload
    if (!keys.includes("Title") || !keys.includes("Body")) {
        res.status(400).send("Invalid JSON Payload");
        return;
    }
    //create review. this function is called recursively if the auto-incremented ReviewID already exists because of a PUT
    async function create() {
        try {
            const doc = await Review.create({
                "Title": body.Title,
                "Body": body.Body,
            });
            //add reviewID to film's review list
            film.Reviews.push(doc.ReviewID);
            film.save();
            res.send(film);
        } catch (err) {
            //if there is a duplicate ID error
            if (err.code === 11000) return create()
            next(err);
        }
    }
    return create();
});

//get all films or get all films matching the search query if it exists
router.get('/:fid/reviews', getFilm, async function(req, res, next) {
    let query = req.query.search; //get search from url
    const reviewIDs = res.film.Reviews; //list of ReviewIDs
    let revs;

    //grab all reviews
    await Review.find({ 'ReviewID': { $in: reviewIDs }
    }).then(function(reviews, err) {
        if (err) { //500 on error
            res.status(500).send("Unable to find reviews for FilmID: " + req.params.fid + " | " + err);
            return;
        }
        revs = reviews;
    });
    //if query defined, run the query and return the results. else send all reviews
    if (query !== undefined) {
        //result contains all reviews that contain the query string in the title or body field
        let result = revs.filter(function(el) {
            if (el.Title.includes(query) || el.Body.includes(query)) return el;
        });
        res.send(result);
        return;
    }
    //send all films
    res.send(revs);
});

//GET review if exists and is associated with fid
router.get('/:fid/reviews/:rid', getFilm, getReview, async (req, res) => {
    if (!res.film.Reviews.includes(res.rev.ReviewID)) {
        return res.status(400).send("FilmID: " + req.params.fid + " is not associated with ReviewID: " + req.params.rid);
    }
    res.send(res.rev);
});

//DELETE review if exists and is associated with fid
router.delete('/:fid/reviews/:rid', getFilm, getReview, async (req, res) => {
    if (!res.film.Reviews.includes(res.rev.ReviewID)) {
        return res.status(400).send("FilmID: " + req.params.fid + " is not associated with ReviewID: " + req.params.rid);
    }
    //remove the review then remove the reviewID from its associated film
    try {
        await res.rev.remove();
        res.film.Reviews = res.film.Reviews.filter(function(e) { return e !== Number(req.params.rid) })
        res.film.save();
        res.send("Successfully deleted Review of ID: " + req.params.rid);
    } catch (err) {
        res.status(500).send("Error deleting Review: " + err.message);
    }
});

//PUT review if fid and rid are associated
//If RevieID does not exist and JSON payload is valid, create new document
//Otherwise send 400
router.put('/:fid/reviews/:rid', getFilm, async (req, res, next) => {
    const fid = req.params.fid;      //FilmID
    const rid = req.params.rid;      //ReviewID
    const body = req.body;           //body
    const keys = Object.keys(body);  //keys in body

    let upsert = false;
    //payload must include required fields or it will not upsert
    if (keys.includes("Title") && keys.includes("Body")) {
        upsert = true;
    }
    if (!res.film.Reviews.includes(rid) && upsert === false) {
        return res.status(400).send("FilmID: " + req.params.fid + " is not associated with ReviewID: " + rid);
    }

    //update doc if it exists. else, upsert if upsert is set
    try {
        let raw = await Review.findOneAndUpdate({
            ReviewID: rid
        }, {
            "Title": body.Title,
            "Body": body.Body,
            "Date": body.Date,
        }, {
            new: true,
            upsert: upsert,
            rawResult: true
        });
        //send 400 if FilmID not present and Payload Invalid for an upsert
        if (raw.value === null) {
            res.status(400).send("Nonexistant ReviewID and Invalid JSON Payload for POST");
            return;
        }
        if (raw.lastErrorObject.updatedExisting === false) {
            res.film.Reviews.push(rid);
            res.film.save();
        }
        //send upserted/updated film
        res.send(raw.value);
    } catch(err) {
        next(err);
    }
});

//middleware to get a film by FilmID from req.params
async function getFilm(req, res, next) {
    let fid = req.params.fid;
    let film;
    try {
        film = await Film.findOne({ FilmID: fid });
        if (film === null) {
            return res.status(404).send("Unable to find Film with id: " + fid);
        }
    } catch (err) {
        return res.status(500).send("Error: " + err.message);
    }
    res.film = film;
    next();
}

//middleware to get a review by reviewID from req.params
async function getReview(req, res, next) {
    let rid = req.params.rid;
    let rev;
    try {
        rev = await Review.findOne({ ReviewID: rid });
        if (rev === null) {
            return res.status(404).send("Unable to find Review with id: " + rid);
        }
    } catch (err) {
        return res.status(500).send("Error: " + err.message);
    }
    res.rev = rev;
    next();
}

module.exports = router;
