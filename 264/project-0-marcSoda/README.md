# CSE264 Project 0: Add new method to JavaScript Array prototype
## Due: Wed, 2/16/2022 at 11:59 PM

## Marcantonio Soda, Jr

In this assignment, you will be adding a new method to the Array prototype, meaning that all Arrays will have access to this new method.
All the code you need is in this GitHub Classroom repo.

### Instructions
You will be implementing a new method to the Array Prototype called "mapCSE264".  mapCSE264 will work similarly to the regular map method for Arrays (see here https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map for more information about map and how it works).

Map takes a function as an argument, and applies it to ever element in the array, returning a new array with the results.

mapCSE264 will also take a function and apply it to ever element in an array, but will also accept other arguments which will tell mapCSE264 to skip that element if the index is passed as an argument.

For example.

```javascript

const bb = [1, 23, 22, 36];

// pass a function to mapCSE264 and index to skip
const new_a = bb.mapCSE264(x => x + 2, 0, 2);

console.log(new_a);
// expected output: Array [1, 25, 22, 38]
//notice that index 0 and 2 were skipped, the function was not applied to them

// pass a function to mapCSE264 and index to skip
const new_new_a = bb.mapCSE264(x => x + 10, 1);

console.log(new_new_a);
// expected output: Array [11, 23, 32, 46]
//now only index 1 was skipped

```

mapCSE264 can have any number of arguments after the first one (the function that will be mapped onto the array), or no arguments after, which means it will act just like the normal map.

If one of the "skip" arguments does not match anything in the array, just ignore it.

For example:
```javascript

const bb = [1, 23, 22];

// pass a function to mapCSE264 and index to skip
const new_a = bb.mapCSE264(x => x + 2, 15);

console.log(new_a);
// expected output: Array [3, 25, 24]
//there is no index 15, so nothing was skipped

```

Also notice that in normal map, the first argument (the callback function applied to each element of the array) can have three arguments.

* currentValue - The current element being processed in the array.
* index (optional) - The index of the current element being processed in the array.
* array (optional) - The array map was called upon.

for example:
```javascript

const bb = [1, 23, 25];

// pass a function to mapCSE264 and index to skip
const new_a = bb.mapCSE264(function (x, y, z)
{

//value of current element
console.log(x);

//value of current index
console.log(y);

//original array
console.log(z);

return x + y + z[0];

}, 15);

console.log(new_a);
// expected output: Array [2, 25, 28]
//there is no index 15, so nothing was skipped

```

Other things to consider:
* Use no outside javascript libraries. No jQuery, etc.
* You can use any part of ES6 Javascript.
* Use the developer tools in your browser to help with debugging
* You will need to have a local server to run javascript from a local source on your browser. If you have python installed on your computer, you can run "python -m SimpleHTTPServer” in your cloned repo directory from the command line. Then in your browser go to http://localhost:8000/ You should see the html file and be able to run the javascript code.

### Grading
* **85 Points** - Current code works as expected (examples in README work)
* **15 Points** - well commented and easy to read/follow

* If code doesn't run/compile you can get no more than a 65 (max grade). But please write comments and a README to explain the process, what you were trying to do.
