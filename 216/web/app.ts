/// <reference path="ts/EditEntryForm.ts"/>
/// <reference path="ts/NewEntryForm.ts"/>
/// <reference path="ts/ElementList.ts"/>
/// <reference path="ts/Navbar.ts"/>
/// <reference path="ts/EntryMenu.ts"/>
/// <reference path="ts/CommentList.ts"/>
/// <reference path="ts/UserPage.ts"/>

// Prevent compiler errors when using Handlebars
let Handlebars: any;

/// This constant indicates the path to our backend server
const backendUrl = location.hostname === "localhost" ? "http://localhost:4567" : "https://runtime-tremor.herokuapp.com";

// a global for the EditEntryForm of the program.  See newEntryForm for
// explanation
let editEntryForm: EditEntryForm;

// Run some configuration code when the web page loads
$(document).ready(function () {
    Navbar.refresh();
    NewEntryForm.refresh();
    ElementList.refresh();
    // EntryMenu.init();

    // Create the object that controls the "Edit Entry" form
    editEntryForm = new EditEntryForm();
    // set up initial UI state
    $("#editElement").hide();
});
