/*
 * This files holds all the code to test you REST API
 */

//Run once broswer has loaded everything
window.onload = function () {

let url = "http://localhost:3000"

//Add html to the specified div in the dom. Used for showing all users and querying
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
    let username = document.getElementById('post_username').value;
    if (username.length === 0) {
        alert("Invalid Username");
        return;
    }
    let name = document.getElementById('post_name').value;
    let bio = document.getElementById('post_bio').value;
    let user = { username: username, name: name, bio: bio };
    //post new user
    fetch(url + '/users', {
        method: 'POST',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(user)
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
    //get username from form
    let username = document.getElementById('get_username').value;
    if (username.length === 0) {
        alert("Invalid Username");
        return;
    }
    //get user
    fetch(url + '/users/' + username)
    //prompt user of the result
    .then(res => {
        if (res.ok) {
            return res.json();
        } else {
            alert("Unsuccessful Get");
        }
    //handle data
    }).then(data => {
        console.log(data);
        if (data === undefined) return; //skip if unsuccessful post. avoids error
        else if (data.length === 0) {
            alert("No Users"); //alert that there are no users if data empty
            return;
        }
        //alert user of the found user
        alert("Username: " + data[0].username + "\n" + "Name: " + data[0].name + "\n" + "Bio: " + data[0].bio);
    }).catch(err => alert("Read Error: " + err));
},false);

//button event for update
document.getElementById("update")
.addEventListener("click",function(e){
    //get values from form
    let old_username = document.getElementById('put_old_username').value;
    let username = document.getElementById('put_username').value;
    let name = document.getElementById('put_name').value;
    let bio = document.getElementById('put_bio').value;
    let user = { username: username, name: name, bio: bio };
    //send put request
    fetch(url + '/users/' + old_username, {
        method: 'PUT',
        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(user)
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
    //get username from form
    let username = document.getElementById('delete_username').value;
    fetch(url + '/users/' + username, {
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
    //get all users
    fetch(url + '/users/')
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
                alert("No Users"); //alert that there are no users if data empty
                return;
            }
            document.getElementById("all").innerHTML = ""; //clear div
            //populate the div for show all
            data.forEach(elm => {
                addHTML("Username " + elm.username + " Name: " + elm.name + " Bio: " + elm.bio, "all");
            });
        }).catch(err => alert("Show Error: " + err));
},false);

//button event for query
document.getElementById("query")
.addEventListener("click",function(e) {
    //get query from form
    let query = document.getElementById('get_query').value
    //query the api
    fetch(url + '/users?search=' + query)
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
                alert("No Users"); //alert that there are no users if data empty
                return;
            }
            document.getElementById("query_div").innerHTML = ""; //clear query div
            //populate query div
            data.forEach(elm => {
                addHTML("Username " + elm.username + " Name: " + elm.name + " Bio: " + elm.bio, "query_div");
            });
        }).catch(err => alert("Query Error: " + err));
},false);

};
