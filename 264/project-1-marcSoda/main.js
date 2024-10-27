//Run once broswer has loaded everything
window.onload = function () {
//Function that adds new Divs to the HTML page
function addHTML(text) {
    //Grab the container div
    var start_div = document.getElementById('start');
    //make a new Div element
    var newElement = document.createElement('div');
    //add text to that div
    newElement.innerHTML = text;
    //append it to the main
    start_div.appendChild(newElement);
}

function SearchReddit(search) {
    //construct uri
    let uri = 'https://www.reddit.com/search.json?q=' + encodeURI(search);
    //query reddit for the user-provided query-string
    fetch(uri)
        .then(res => res.json())
        .then(data => {
            let objs = []; //main list of relevant data to be added to the dom later
            let authors = []; //list of pending fetches. checked against later to prevent double fetching.
            let promises = []; //list of fetches
            //for each element in the data previously fetched
            data.data.children.forEach(elm => {
                let author = elm.data.author; //get author
                if (authors.includes(author)) return; //check if author already has a pending fetch
                authors.push(author); //push author to list of pending fetches
                let title = elm.data.title; //get title
                //construct list of fetches
                promises.push(fetch('https://www.reddit.com/user/' + author + '/about.json')
                    .then(res => res.json())
                    .then(dat => objs.push({
                        RedditPost: title,
                        RedditUser: author,
                        LinkKarma: dat.data.link_karma,
                        CommentKarma: dat.data.comment_karma
                        //Handle Errors
                    })).catch(err => console.log("Could not get all info associated with user: " + author + "ERR: " + err
                )));
            });

            //wait for all fetches to be done
            Promise.all(promises).then(() => {
                //sort objects in order of descending link karma
                objs.sort((a, b) => (a.LinkKarma > b.LinkKarma) ? -1 : 1);
                //add each object to the dom
                objs.forEach(elm => {
                    addHTML("User " + elm.RedditUser + " wrote the post \"" + elm.RedditPost + "\" and has " + elm.LinkKarma + " link karma and " + elm.CommentKarma + " comment carma.");
                });
            });
      //Handle errors
        }).catch(err => console.log("Unable to get batch. ERR: " + err));
}

//gran the current form in the HTML document
var form = document.querySelector("form");

//event that listens for form submit
form.addEventListener("submit", function(event) {
  var search_text = form.elements.value.value;
  console.log("Saving value", search_text);
  //get main DIV
  var start_div = document.getElementById('start');
  //Clear main DIV
  start_div.innerHTML = '';
  addHTML("Looking up Reddit Users for search term "+search_text);
  SearchReddit(search_text);
  event.preventDefault();
});
};
