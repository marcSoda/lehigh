/*
 * This files holds all the code for Project 0.
 */

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

    Array.prototype.mapCSE264 = function(func) {
        //array of indices to skip (arguments excluding the passed function)
        let args = Object.values(arguments).slice(1);
        //for every element in the array
        for (let i = 0; i < this.length; i++) {
            //continue if index is meant to be skipped
            if (args.includes(i)) continue;
            //pass array value to function and reassign at index
            this[i] = func(this[i], i, this);
        }
        return this;
    }

    //TESTS:
    addHTML("Testing");

    //TEST 1
    const a = [1, 23, 30];
    const new_a = a.mapCSE264(function (x, y, z) {
        return x + y + z[0];
    }, 0, 2, 15);
    console.log(new_a);
    addHTML("new_a is now "+ new_a);
    addHTML("new_a should be [1, 25, 30]");

    //TEST 2
    const b = [1, 23, 22, 36];
    const new_b = b.mapCSE264(x => x + 2, 0, 2);
    console.log(new_b);
    addHTML("new_b is now "+ new_b);
    addHTML("new_b should be [1, 25, 22, 38]");

    //TEST 3
    const c = [1, 23, 22, 36];
    const new_c = c.mapCSE264(x => x + 10, 1);
    console.log(new_c);
    addHTML("new_c is now " + new_c);
    addHTML("new_c should be [11, 23, 32, 46]");

    //TEST 4
    const d = [1, 23, 22];
    const new_d = d.mapCSE264(x => x + 2, 15);
    console.log(new_d);
    addHTML("new_d is now " + new_d);
    addHTML("new_d should be [3, 25, 24]");
};
