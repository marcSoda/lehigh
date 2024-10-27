const express = require('express');
const router = express.Router();

let users = []; //array of all users

//get all users or get all users matching the search query if it exists
router.get('/', function(req, res, next) {
    let query = req.query.search; //get search from url
    //if query defined, run the query and return the results. else send all users
    if (query !== undefined) {
        //result contains all users that contain the query string in the bio or name field
        let result = users.filter(function(el) {
            if (el.bio.includes(query) || el.name.includes(query)) return el;
        });
        //if there are no results, send a 404
        if (result.length === 0) {
            res.status(404).send()
            return;
        }
        res.send(result);
        return;
    }
    //send all answers
    if (users.length === 0) res.status(404).send();
    res.send(users);
});

// add a user
router.post('/', function(req, res, next) {
    let body = req.body;          //json object
    let keys = Object.keys(body); //keys in body
    let usernames = users.map(function(el) { return el.username; });
    //validate json. send 404 if input invalid
    if (keys.length !== 3                        //ensure object has only three fields
        || !keys.includes("username")            //ensure username field included
        || !keys.includes("name")                //ensure bio field included
        || !keys.includes("bio")                 //ensure bio field included
        || usernames.includes(body.username)) {  //ensure user exists
        res.status(404).send();
        return;
    }
    users.push(body); //add user to users array
    res.status(201).send();
});

//get user based on username
router.get('/:username', function(req, res, next) {
    let username = req.params.username; //username from url
    let usernames = users.map(function(el) { return el.username; }); //array of usernames
    //ensure user exists. if not, send 404.
    if (!usernames.includes(username)) {
        res.status(404).send();
        return;
    }
    //get the user from the users array
    var user = users.filter(obj => { return obj.username === username }); //get user object
    res.send(user); //send user data
});

//update user based on username and json body
router.put('/:username', function(req, res, next) {
    let username = req.params.username; //username from url
    let usernames = users.map(function(el) { return el.username; }); //array of usernames
    //ensure user exists. if not, send 404.
    if (!usernames.includes(username)) {
        res.status(404).send();
        return;
    }
    let body = req.body; //json object
    let keys = Object.keys(body); //array of keys from body
    //validate json. send 404 if input invalid
    if (keys.length !== 3              //ensure object size is 3
        || !keys.includes("username")  //ensure username field exists
        || !keys.includes("name")      //ensure name field exists
        || !keys.includes("bio")) {    //ensure bio field exists
        res.status(404).send();
        return;
    }
    users = users.filter(function(el) { return el.username != username; }); //delete user from user array
    users.push(body); //update the user data
    res.status(201).send();
});

//delete user based on username
router.delete('/:username', function(req, res, next) {
    let username = req.params.username; //username from url
    let usernames = users.map(function(el) { return el.username; }); //username array
    //ensure user exists. if not, send 404.
    if (!usernames.includes(username)) {
        res.status(404).send();
        return;
    }
    users = users.filter(function(el) { return el.username != username; }); //delete user from user array
    res.status(201).send();
});


module.exports = router;
