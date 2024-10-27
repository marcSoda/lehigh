class CommentList {

    private static readonly NAME = "CommentList";

    /**
     * Track if the Singleton has been initialized
     */
    public static isInit = false;

    //easier way of keeping track of the postId between functions
    private static postId:number = -1;

    private static init(data: any) {
	// Remove the table of data, if it exists
	$("#" + CommentList.postId).append(Handlebars.templates[CommentList.NAME + ".hb"](data));

    $("." + CommentList.NAME + "-Delete").click(CommentList.delete);
	$("#" + CommentList.NAME + "-OK").click(CommentList.submit);
	$("#" + CommentList.NAME + "-Close").click(CommentList.close);
	CommentList.isInit = true;
    }

    public static spawn(id: number) {
        CommentList.postId = id;
	// Issue a GET, and then pass the result to update()
	$.ajax({
	    type: "GET",
            url: backendUrl + "/messages/" + id + "/comments",
	    dataType: "json",
	    //NOTE: success line is currently not running. when it does there will be hella errors because there is not a working backend
	    success: CommentList.init,
	    error: function() {
		console.log("CommentList spawn error")
	    }
	});
    }

    public static close() {
	//remove the form from the dom. unlike NewEntryForm, it is removed and added when needed
	$('.modal-backdrop').remove();
	$('body').removeClass("modal-open");
    $("#" + CommentList.NAME).remove();
    //console.log($("#" + CommentList.NAME).position());
        CommentList.postId = -1;
    }

    /**
     * Send data to submit the form only if the fields are both valid.
     * Immediately hide the form when we send data, so that the user knows that
     * their click was received.
     */
    private static submit() {
        // get the values of the two fields, force them to be strings, and check
        // that neither is empty
        let commentText = "" + $("#" + CommentList.NAME + "-commentInput").val();
        if (commentText === "") {
            window.alert("Error: Comment is not valid");
            return;
        }
        // set up an AJAX post.  When the server replies, the result will go to
        // onSubmitResponse
        $.ajax({
            type: "POST",
            url: backendUrl + "/messages/" + CommentList.postId + "/comments",
            dataType: "json",
            data: JSON.stringify({ mComment: commentText }),
            success: CommentList.onSubmitResponse
        });
    }

    /**
     * onSubmitResponse runs when the AJAX call in submitForm() returns a
     * result.
     *
     * @param data The object returned by the server
     */
    private static onSubmitResponse(data: any) {
        // If we get an "ok" message, clear the form and refresh the main
        // listing of messages
        if (data.mStatus === "ok") {
            ElementList.refresh();
        }
        // Handle explicit errors with a detailed popup message
        else if (data.mStatus === "error") {
            window.alert("The server replied with an error:\n" + data.mMessage);
        }
        // Handle other errors with a less-detailed popup message
        else {
            window.alert("Unspecified error");
        }
    }

    // delete comment on click
    private static delete() {
        let pid = $(this).data("pid");
        let cid = $(this).data("cid");
        console.log(pid);
        console.log(cid);
	$.ajax({
	    type: "DELETE",
	    url: backendUrl + "/messages/" + pid + "/comments/" + cid,
	    dataType: "json",
	    // TODO: we should really have a function that looks at the return
	    //       value and possibly prints an error message.
	    success: function() {
		ElementList.refresh();
	    }
	});
    }
}
