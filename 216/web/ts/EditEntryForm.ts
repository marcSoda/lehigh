class EditEntryForm {

    public static readonly NAME = "EditEntryForm";

    /**
     * Track if the Singleton has been initialized
     */
    private static isInit = false;

    private static init(id: number) {
    EditEntryForm.close();
    console.log(id);
	let subject = $("." + ElementList.NAME + "-subject[data-value=\"" + id + "\"]").text();
    let message = $("." + ElementList.NAME + "-message[data-value=\"" + id + "\"]").text();
    //let link = $("." + ElementList.NAME + "-file[data-value=\"" + id + "\"]").text();
	$("body").append(Handlebars.templates[EditEntryForm.NAME + ".hb"]({id: id,
									   subject: subject,
                                       message: message}));
	$("#" + EditEntryForm.NAME + "-OK").click(EditEntryForm.submitForm);
	$("#" + EditEntryForm.NAME + "-Close").click(EditEntryForm.close);
        $("#" + EditEntryForm.NAME).modal("show");
	EditEntryForm.isInit = true;
    }

    /**
     * Refresh() doesn't really have much meaning, but just like in sNavbar, we
     * have a refresh() method so that we don't have front-end code calling
     * init().
     */
    public static spawn(id: number) {
        EditEntryForm.init(id);
    }

    public static close() {
	//remove the form from the dom. unlike NewEntryForm, it is removed and added when needed
	$('.modal-backdrop').remove();
	$('body').removeClass("modal-open");
	$("#" + EditEntryForm.NAME).remove();
    }

    /**
     * Send data to submit the form only if the fields are both valid.
     * Immediately hide the form when we send data, so that the user knows that
     * their click was received.
     */
    private static submitForm() {
        // get the values of the two fields, force them to be strings, and check
        // that neither is empty
        var read = new FileReader();
        console.log("I exist");
        let id = $(this).data("value");
        
        let title = "" + $("#" + EditEntryForm.NAME + "-title").val();
        let msg = "" + $("#" + EditEntryForm.NAME + "-message").val();
        var link = "";
        var file = $('#filely').prop('files')[0];
        try {
        read.readAsBinaryString(file);
        } catch (e) {
            console.log("in ajax");
            $.ajax({
                type: "PUT",
                url: backendUrl + "/messages/" + id,
                dataType: "json",
                data: JSON.stringify({ mTitle: title, mMessage: msg, mLink: link }),
                success: EditEntryForm.onSubmitResponse
            });
        }
        link = btoa(read.result as string);
        read.onloadend = function() {
            link = "inside";
            link = btoa(read.result as string);
            $.ajax({
                type: "PUT",
                url: backendUrl + "/messages/" + id,
                dataType: "json",
                data: JSON.stringify({ mTitle: title, mMessage: msg, mLink: link }),
                success: EditEntryForm.onSubmitResponse
            });
        }
        if (title === "" || msg === "") {
            window.alert("Error: title or message is not valid");
            return;
        }
        /*let id = $(this).data("value");
        $.ajax({
            type: "PUT",
            url: backendUrl + "/messages/" + id,
            dataType: "json",
            data: JSON.stringify({ mTitle: title, mMessage: msg }),
            success: EditEntryForm.onSubmitResponse
        }); */
        EditEntryForm.close();
        // set up an AJAX post.  When the server replies, the result will go to
        // onSubmitResponse
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
}


// /**
//  * EditEntryForm encapsulates all of the code for the form for editing an entry
//  */
// class EditEntryForm {
//     /**
//      * To initialize the object, we say what method of EditEntryForm should be
//      * run in response to each of the form's buttons being clicked.
//      */
//     constructor() {
//         $("#editCancel").click(this.clearForm);
//         $("#editButton").click(this.submitForm);
//     }

//     /**
//      * init() is called from an AJAX GET, and should populate the form if and
//      * only if the GET did not have an error
//      */
//     init(data: any) {
//         if (data.mStatus === "ok") {
//             $("#editTitle").val(data.mData.mTitle);
//             $("#editMessage").val(data.mData.mContent);
//             $("#editId").val(data.mData.mId);
//             $("#editCreated").text(data.mData.mCreated);
// 	    // show the edit form
// 	    $("#addElement").hide();
// 	    $("#editElement").show();
// 	    $("#showElements").hide();
//         }
//         else if (data.mStatus === "error") {
//             window.alert("Error: " + data.mMessage);
//         }
//         else {
//             window.alert("An unspecified error occurred");
//         }
//     }

//     /**
//      * Clear the form's input fields
//      */
//     clearForm() {
//         $("#editTitle").val("");
//         $("#editMessage").val("");
//         $("#editId").val("");
//         $("#editCreated").text("");
//     }

//     /**
//      * Check if the input fields are both valid, and if so, do an AJAX call.
//      */
//     submitForm() {
//         // get the values of the two fields, force them to be strings, and check
//         // that neither is empty
//         let title = "" + $("#editTitle").val();
//         let msg = "" + $("#editMessage").val();
//         // NB: we assume that the user didn't modify the value of #editId
//         let id = "" + $("#editId").val();
//         if (title === "" || msg === "") {
//             window.alert("Error: title or message is not valid");
//             return;
//         }
//         // set up an AJAX post.  When the server replies, the result will go to
//         // onSubmitResponse
//         $.ajax({
//             type: "PUT",
//             url: backendUrl + "/messages/" + id,
//             dataType: "json",
//             data: JSON.stringify({ mTitle: title, mMessage: msg }),
//             success: editEntryForm.onSubmitResponse
//         });
//     }

//     /**
//      * onSubmitResponse runs when the AJAX call in submitForm() returns a
//      * result.
//      *
//      * @param data The object returned by the server
//      */
//     private onSubmitResponse(data: any) {
//         // If we get an "ok" message, clear the form and refresh the main
//         // listing of messages
//         if (data.mStatus === "ok") {
//             editEntryForm.clearForm();
//             mainList.refresh();
//         }
//         // Handle explicit errors with a detailed popup message
//         else if (data.mStatus === "error") {
//             window.alert("The server replied with an error:\n" + data.mMessage);
//         }
//         // Handle other errors with a less-detailed popup message
//         else {
//             window.alert("Unspecified error");
//         }
//     }
// }  //end class EditEntryForm
