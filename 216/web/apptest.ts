var describe: any;
var it: any;
var expect: any;
var ElementList: any;
var EntryMenu: any;
var NewEntryForm: any;
var CommentList: any;
var EditEntryForm: any;
var UserPage: any;

describe("Tests of login", function() {
    it("Test login", function() {
    console.log($("#ElementList"))
    localStorage.setItem('uid', "6");
	expect(localStorage.getItem('uid')).toEqual("6");
    })
})	;

describe("Tests for languages", function() {
    it("Test lang", function() {
        localStorage.setItem('lang', "en");
        expect(localStorage.getItem('lang')).toEqual("en");
    })
});

describe("Test for User id", function() {
    it("Test for luid", function() {
        localStorage.setItem('luid', "8")
        expect(localStorage.getItem('luid')).toEqual("8");
    })
}); 

describe("Test for Refresh", function() {
    it("Refresh", function() {
        ElementList.refresh();
        NewEntryForm.refresh();
        expect(ElementList.refreshed).toEqual(true);
        expect(NewEntryForm.refreshed).toEqual(true);
    })
}); 

describe("Test for basic math", function() {
    it("Addition", function() {
        var a = 1;
        var b = 2;
        expect(a + b).toEqual(3);
    })
    it("Subtract", function () {
        var a = 420;
        var b = 69;
        expect(a - b).toEqual(351);
    })
    it("Multiply", function () {
        var a = 12;
        var b = 12;
        expect(a * b).toEqual(144);
    })
    it("Divide", function () {
        var a = 90;
        var b = 10;
        expect(a / b).toEqual(9);
    })
})

describe("Test for Refresh", function() {
    it("Close Comment", function() {
        CommentList.close();
        expect($("#" + CommentList.NAME).position()).toEqual(undefined);
    })
    it("Close EditEntryForm", function() {
        EditEntryForm.close();
        expect($("#" + EditEntryForm.NAME).position()).toEqual(undefined);
    })
    it("Close EntryMenu", function() {
        EntryMenu.close();
        expect($("#" + EntryMenu.NAME).position()).toEqual(undefined);
    })
    it("Close Userpage", function() {
        UserPage.close();
        expect($("#" + UserPage.NAME).position()).toEqual(undefined);
    })
}); 

