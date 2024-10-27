/*
 * This files holds all the code to test you REST API
 */

//Run once broswer has loaded everything
window.onload = function () {

let url = "http://localhost:3000"

//Add html to the specified div in the dom.
function addHTML(text, div_id) {
    //Grab the container div
    var div = document.getElementById(div_id);
    //make a new Div element
    var newElement = document.createElement('div');
    //add text to that div
    newElement.innerHTML = text;
    //append it to the main
    div.appendChild(newElement);
}

//button event for create
document.getElementById("create")
.addEventListener("click",function(e) {
    //get values from form
    let title = document.getElementById('post_film_title').value;
    let body = document.getElementById('post_film_body').value;
    if (title.length === 0 || body.length === 0) {
        alert("Invalid title or body");
        return;
    }
    let film = { Title: title, Body: body };
    //post new film
    fetch(url + '/films', {
        method: 'POST',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(film)
    //prompt user of the post result
    }).then(res => {
        if (res.ok) {
            alert("Successful Post");
        } else {
            alert("Unsuccessful Post");
        }
    }).catch(err => alert("Post Error: " + err));

},false);

//button event for read
document.getElementById("read")
.addEventListener("click",function(e) {
    //get fid from form
    let fid = document.getElementById('get_film_fid').value;
    if (fid.length === 0) {
        alert("Invalid FilmID");
        return;
    }
    //get film
    fetch(url + '/films/' + fid)
    //prompt user of the result
    .then(res => {
        if (res.ok) {
            return res.json();
        } else {
            alert("Unsuccessful Get");
        }
    //handle data
    }).then(data => {
        if (data === undefined) return; //skip if unsuccessful post. avoids error
        else if (data.length === 0) {
            alert("No Film matching given FilmID"); //alert that there are no films if data empty
            return;
        }
        //alert user of the found film
        alert("Title: " + data.Title + "\n" + "Body: " + data.Body + "\n" + "FilmID: " + data.FilmID);
    }).catch(err => alert("Read Error: " + err));
},false);

//button event for update
document.getElementById("update")
.addEventListener("click",function(e){
    //get values from form
    let fid = document.getElementById('put_film_fid').value;
    let title = document.getElementById('put_film_title').value;
    let body = document.getElementById('put_film_body').value;
    let date = Date.parse(document.getElementById('put_film_date').value);
    let film = { FilmID: fid, Title: title, Body: body, Date: date };
    //send put request
    fetch(url + '/films/' + fid, {
        method: 'PUT',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(film)
    //alert user of fetch result
    }).then(res => {
        if (res.ok) {
            alert("Successful PUT");
        } else {
            alert("Unsuccessful PUT");
        }
    }).catch(err => alert("Put Error: " + err));
},false);

//button event for destroy
document.getElementById("destroy")
.addEventListener("click",function(e) {
    //get fid from form
    let fid = document.getElementById('delete_film_fid').value;
    fetch(url + '/films/' + fid, {
        method: 'DELETE',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
    //alert user of fetch result
    }).then(res => {
        if (res.ok) {
            alert("Successful DELETE");
        } else {
            alert("Unsuccessful DELETE");
        }
    }).catch(err => alert("Delete Error: " + err));
},false);

//button event for show all
document.getElementById("show")
.addEventListener("click",function(e) {
    fetch(url + '/films/')
        //alert user of fetch result
        .then(res => {
            if (res.ok) {
                return res.json();
            } else {
                alert("Unsuccessful GET");
            }
        //handle data
        }).then(data => {
            if (data === undefined) return; //skip if unsuccessful post. avoids error
            else if (data.length === 0) {
                alert("No Films"); //alert that there are no films if data empty
                return;
            }
            document.getElementById("all_films").innerHTML = ""; //clear div
            //populate the div for show all
            data.forEach(elm => {
                addHTML("FilmId: " + elm.FilmID + " Title: " + elm.Title + " Body: " + elm.Body + " Date: " + elm.Date, "all_films");
            });
        }).catch(err => alert("Show Error: " + err));
},false);

//button event for query
document.getElementById("query_films")
.addEventListener("click",function(e) {
    //get query from form
    let query = document.getElementById('get_films_query').value
    //query the api
    fetch(url + '/films?search=' + query)
        //alert user of the result of the fetch
        .then(res => {
            if (res.ok) {
                return res.json();
            } else {
                alert("Unsuccessful Query");
            }
        //handle data
        }).then(data => {
            if (data === undefined) return; //return if data is undefined
            else if (data.length === 0) {
                alert("No Films"); //alert that there are no films if data empty
                return;
            }
            document.getElementById("query_films_div").innerHTML = ""; //clear query div
            //populate query div
            data.forEach(elm => {
                addHTML("FilmID " + elm.FilmID + " Title: " + elm.Title + " Body: " + elm.Body + " Date: " + elm.Date, "query_films_div");
            });
        }).catch(err => alert("Query Error: " + err));
},false);

//button event for posting a review
document.getElementById("rev_post")
.addEventListener("click",function(e) {
    //get values from form
    let fid = document.getElementById('post_rev_fid').value;
    let title = document.getElementById('post_rev_title').value;
    let body = document.getElementById('post_rev_body').value;
    if (title.length === 0 || body.length === 0) {
        alert("Invalid title or body");
        return;
    }
    let rev = { Title: title, Body: body };
    //post new film
    fetch(url + '/films/' + fid + '/reviews', {
        method: 'POST',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(rev)
    //prompt user of the post result
    }).then(res => {
        if (res.ok) {
            alert("Successful Post");
        } else {
            alert("Unsuccessful Post");
        }
    }).catch(err => alert("Post Error: " + err));

},false);

//button event for reading one review
document.getElementById("rev_get_one")
.addEventListener("click",function(e) {
    let fid = document.getElementById('get_one_rev_fid').value;
    let rid = document.getElementById('get_one_rev_rid').value;
    if (fid.length === 0 || rid.length === 0) {
        alert("Invalid FilmID or ReviewID");
        return;
    }
    //get review
    fetch(url + '/films/' + fid + '/reviews/' + rid)
    //prompt user of the result
    .then(res => {
        if (res.ok) {
            return res.json();
        } else {
            alert("Unsuccessful Get");
        }
    //handle data
    }).then(data => {
        if (data === undefined) return; //skip if unsuccessful post. avoids error
        else if (data.length === 0) {
            alert("No Film matching given FilmID and ReviewID"); //alert if data empty
            return;
        }
        //alert user of the found review
        alert("Title: " + data.Title + "\nBody: " + data.Body + "\nDate: " + data.Date + "\nFilmID: " + fid + "\nReviewID: " + data.ReviewID);
    }).catch(err => alert("Read Error: " + err));
},false);

//button event for update
document.getElementById("rev_put")
.addEventListener("click",function(e){
    //get values from form
    let fid = document.getElementById('put_rev_fid').value;
    let rid = document.getElementById('put_rev_rid').value;
    let title = document.getElementById('put_rev_title').value;
    let body = document.getElementById('put_rev_body').value;
    let date = Date.parse(document.getElementById('put_rev_date').value);
    let rev = { Title: title, Body: body, Date: date };
    //send put request
    fetch(url + '/films/' + fid + '/reviews/' + rid, {
        method: 'PUT',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(rev)
    //alert user of fetch result
    }).then(res => {
        if (res.ok) {
            alert("Successful PUT");
        } else {
            alert("Unsuccessful PUT");
        }
    }).catch(err => alert("Put Error: " + err));
},false);

//button event for destroy
document.getElementById("rev_delete")
.addEventListener("click",function(e) {
    let fid = document.getElementById('delete_rev_fid').value;
    let rid = document.getElementById('delete_rev_rid').value;
    fetch(url + '/films/' + fid + "/reviews/" + rid, {
        method: 'DELETE',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
    //alert user of fetch result
    }).then(res => {
        if (res.ok) {
            alert("Successful DELETE");
        } else {
            alert("Unsuccessful DELETE");
        }
    }).catch(err => alert("Delete Error: " + err));
},false);

//button event for show all
document.getElementById("rev_get_all")
.addEventListener("click",function(e) {
    let fid = document.getElementById('get_all_rev_fid').value;
    fetch(url + '/films/' + fid + '/reviews')
        //alert user of fetch result
        .then(res => {
            if (res.ok) {
                return res.json();
            } else {
                alert("Unsuccessful GET");
            }
        //handle data
        }).then(data => {
            if (data === undefined) return; //skip if unsuccessful post. avoids error
            else if (data.length === 0) {
                alert("No Reviews"); //alert that there are no reviews if data empty
                return;
            }
            //populate the div for show all
            document.getElementById("all_film_reviews").innerHTML = ""; //clear div
            data.forEach(elm => {
                addHTML("FilmId: " + fid + " ReviewID: " + elm.ReviewID + " Title: " + elm.Title + " Body: " + elm.Body + " Date: " + elm.Date, "all_film_reviews");
            });
        }).catch(err => alert("Show Error: " + err));
},false);

//button event for query reviews
document.getElementById("query_revs")
.addEventListener("click",function(e) {
    //get data from form
    let fid = document.getElementById('get_revs_query_fid').value
    let query = document.getElementById('get_revs_query').value
    //query the api
    fetch(url + '/films/' + fid + '/reviews?search=' + query)
        //alert user of the result of the fetch
        .then(res => {
            if (res.ok) {
                return res.json();
            } else {
                alert("Unsuccessful Query");
            }
        //handle data
        }).then(data => {
            if (data === undefined) return; //return if data is undefined
            else if (data.length === 0) {
                alert("No Reviews"); //alert that there are no reviews if data empty
                return;
            }
            document.getElementById("query_revs_div").innerHTML = ""; //clear query div
            //populate query div
            data.forEach(elm => {
                addHTML("FilmID " + fid + " Title: " + elm.Title + " Body: " + elm.Body + " Date: " + elm.Date, "query_revs_div");
            });
        }).catch(err => alert("Query Error: " + err));
},false);

};
