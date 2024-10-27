class UserPage {

    private static readonly NAME = "UserPage";
    private static uid:string = "";

    /**
     * Track if the Singleton has been initialized
     */
    public static isInit = false

    //prepare the userPage
    private static init() {
        $("#" + ElementList.NAME).remove();
        UserPage.getUserData();
        UserPage.isInit = true;
    }

    /**
     * Refresh() doesn't really have much meaning, but just like in sNavbar, we
     * have a refresh() method so that we don't have front-end code calling
     * init().
     */
    public static spawn(id: string) {
        UserPage.uid = id;
        UserPage.init();
        $("#" + UserPage.NAME + "-update").click(UserPage.clickSubmit);
        $("#" + UserPage.NAME + "-back").click(UserPage.clickBack);
    }

    //initialize handlebars template with user data from the database
    public static fill(data: any) {
	$("body").append(Handlebars.templates[UserPage.NAME + ".hb"](data));
    }

    //get user data from the database.
    public static getUserData() {
	$.ajax({
	    type: "GET",
            url: backendUrl + "/users/" + UserPage.uid,
	    dataType: "json",
	    success: function(data: any) {
                console.log(data);
                UserPage.fill(data);
            },
            error: function() {
                console.log("getUserData error");
            }
	});
    }

    private static clickSubmit() {
        // get the values of the two fields, force them to be strings, and check
        // that neither is empty
        let bio = "" + $("#" + UserPage.NAME + "-description").val();
        if (bio === "") {
            window.alert("Error: bio field invalid");
            return;
        }
        $.ajax({
            type: "PUT",
            url: backendUrl + "/users/" + UserPage.uid,
            dataType: "json",
            data: JSON.stringify({ mDescription: bio }),
            success: function(data: any) {
                // UserPage.onSubmitResponse(data);
            },
            error: function() {
                console.log("subitForm error");
            }
        });
    }

    //close the form
    public static close() {
        UserPage.uid = "";
    }

    // return back to main page
    private static clickBack() {
        ElementList.refresh(); // refresh back to main
    }

//     /**
//      * Send data to submit the form only if the fields are both valid.
//      * Immediately hide the form when we send data, so that the user knows that
//      * their click was received.
//      */
//     private static submitForm() {
//         // get the values of the two fields, force them to be strings, and check
//         // that neither is empty
//         let email = "" + $("#" + UserPage.NAME + "-email").val();
//         let bio = "" + $("#" + UserPage.NAME + "-bio").val();
//         if (email === "" || bio === "") {
//             window.alert("Error: field(s) invalid");
//             return;
//         }
//         UserPage.close();
//         // set up an AJAX post.  When the server replies, the result will go to
//         // onSubmitResponse
// 	let id = $(this).data("value");
//         $.ajax({
//             type: "PUT",
//             url: backendUrl + "/user/" + id,
//             dataType: "json",
//             data: JSON.stringify({ mTitle: email, mMessage: bio }),
//             success: function(data: any) {
//                 console.log("succ");
//                 UserPage.onSubmitResponse(data);
//             },
//             error: function() {
//                 console.log("subitForm error");
//             }
//         });
//     }

//     /**
//      * onSubmitResponse runs when the AJAX call in submitForm() returns a
//      * result.
//      *
//      * @param data The object returned by the server
//      */
//     private static onSubmitResponse(data: any) {
//         // If we get an "ok" message, clear the form and refresh the main
//         // listing of messages
//         if (data.mStatus === "ok") {
//             ElementList.refresh();
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
}
