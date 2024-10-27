class ElementList {
    /**
     * The name of the DOM entry associated with ElementList
     */
    public static readonly NAME = "ElementList";

    public static refreshed = false;

    /**
     * Track if the Singleton has been initialized
     */
    private static isInit = false;

    /**
    * Initialize the ElementList singleton.
    * This needs to be called from any public static method, to ensure that the
    * Singleton is initialized before use.
    */
    private static init() {
	if (!ElementList.isInit) {
        ElementList.isInit = true;
        localStorage.setItem("lang", "en"); // intialize language
	}
    }

    /**
    * update() is the private method used by refresh() to update the
    * ElementList
    */
    private static update(data: any) {
	// Remove the table of data, if it exists
	$("#" + ElementList.NAME).remove();
	// Use a template to re-generate the table, and then insert it
	$("body").append(Handlebars.templates[ElementList.NAME + ".hb"](data));
    $("." + ElementList.NAME + "-menu-div").click(ElementList.clickMenu);
	// Find all of the Upvote buttons, and set their behavior
	$("." + ElementList.NAME + "-upvote-div").click(ElementList.clickUpvote);
	// Find all of the Downvote buttons, and set their behavior
	$("." + ElementList.NAME + "-downvote-div").click(ElementList.clickDownvote);
	$("." + ElementList.NAME + "-comment-div").click(ElementList.clickComment);
	$("." + ElementList.NAME + "-userName").click(ElementList.clickUser);
    }

    /**
    * refresh() is the public method for updating the ElementList
    */
    public static refresh() {
        var lang = localStorage.getItem("lang"); //THIS MUST BE REMOVED. FIND A WAY TO CHANGE THIS VARIABLE. USED IN THE PROCEEDING AJAX CALL
        // Make sure the singleton is initialized
        ElementList.refreshed = true;
        ElementList.init();
        // Issue a GET, and then pass the result to update()
        $.ajax({
            type: "GET",
            url: backendUrl + "/messages/" + lang,
            dataType: "json",
            success: ElementList.update,
        });
    }

    //entry menu spawns
    private static clickMenu() {
    let pid = $(this).data("value");
    let uid = $(this).data("user");
    localStorage.setItem("uid", uid);
    $.ajax({
        type: "GET",
        url: backendUrl + "/user/uid",
        dataType: "json",
        success: function(data: any) {
            let value = data.mData;
            console.log(value);
            localStorage.setItem("luid", value);
            return data;
        },
        error: function() {
            console.log("err");
            return "hello";
        }
    });
    EntryMenu.spawn(pid, uid);
	return false;
    }

    //user page spawns
    private static clickUser() {
	let uid = $(this).data("value");
	UserPage.spawn(uid);
    }

    /**
    * clickUpvote is the code we run in response to a click of a like button
    */
    private static clickUpvote() {
	let id = $(this).data("value");
	let upvotes = $(this).children().last();
        let downvotes = $(this).parent().children().eq(2).children().last();
	$.ajax({
	    type: "POST",
	    url: backendUrl + "/messages/" + id + "/upvote",
	    dataType: "json",
	    //Increment the number of likes ONLY if a successful response is received
	    success: function(resp: any) {
		if (resp.mStatus === 'ok') {
                    upvotes.html(resp.mData[0]);
                    downvotes.html(resp.mData[1]);
		} else { alert("upvote fail") }
	    },
	    error: function() {
		alert("upvote fail");
	    }
	});
    }

    /**
    * clickDownvote is the code we run in response to a click of a dislike button
    */
    private static clickDownvote() {
	let id = $(this).data("value");
	let downvotes = $(this).children().last();
        let upvotes = $(this).parent().children().eq(1).children().last();
	$.ajax({
	    type: "POST",
	    url: backendUrl + "/messages/" + id + "/downvote",
	    dataType: "json",
	    //Increment the number of dislikes ONLY if a successful response is received
	    success: function(resp: any) {
		if (resp.mStatus === 'ok') {
                    upvotes.html(resp.mData[0])
                    downvotes.html(resp.mData[1])
		} else { alert("downvote fail") }
	    },
	    error: function() {
		alert("downvote fail");
	    }
	});
    }

    //comment page spawns
    private static clickComment() {
	let id = $(this).data("value");
	CommentList.spawn(id);
    }
}
