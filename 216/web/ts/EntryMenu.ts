class EntryMenu {
    /**
     * Track if the Singleton has been initialized
     */
    public static isInit = false;

    /**
     * The name of the DOM entry associated with EntryMenu
     */
    private static readonly NAME = "EntryMenu";

    private static init() {
	//close entry menu if a click event happens outside the menu or the menu icon
	$(document).click(function() {
	    var entryMenu = $("#EntryMenu");
	    var menuDivs = $('.ElementList-menu-div');
            const et = event?.target as EventTarget
            if ($(et).closest("#EntryMenu").length === 0) EntryMenu.close();
	});

	EntryMenu.isInit = true;
    }

    public static spawn(pid: number, uid: number) {
	if (!EntryMenu.isInit) EntryMenu.init();
	EntryMenu.close();
	//change position of menu to be where the menu icon of the corresponding element
	let position = $("." + ElementList.NAME + "-menu-div[data-value=\"" + pid + "\"]").position();
	//add EntryMenu template
	$("body").prepend(Handlebars.templates[EntryMenu.NAME + ".hb"]({id: pid,
									uid: uid,
									left: position.left,
									top: position.top,
									position: "absolute"}));

	//$("#" + EntryMenu.NAME + "-delete").style.visibility = "hidden";
	// hide buttons for user
	if (localStorage.getItem("luid") != localStorage.getItem("uid")){ // looking at other user's post
		(<HTMLElement>document.querySelector("#" + EntryMenu.NAME + "-delete")).style.display = "none";
		(<HTMLElement>document.querySelector("#" + EntryMenu.NAME + "-edit")).style.display = "none";
	}
	else { // looking at your own post
		(<HTMLElement>document.querySelector("#" + EntryMenu.NAME + "-flag")).style.display = "none";
		(<HTMLElement>document.querySelector("#" + EntryMenu.NAME + "-block")).style.display = "none";
	}
	//Set click events
	$("#" + EntryMenu.NAME + "-delete").click(EntryMenu.clickDelete);
	$("#" + EntryMenu.NAME + "-edit").click(EntryMenu.clickEdit);
	$("#" + EntryMenu.NAME + "-visit").click(EntryMenu.clickUserPage);
	$("#" + EntryMenu.NAME + "-flag").click(EntryMenu.clickFlag);
	$("#" + EntryMenu.NAME + "-block").click(EntryMenu.clickBlock);

    }

    /**
    * clickDelete is the code we run in response to a click of a delete button
    */
    private static clickDelete() {
	// for now, just print the ID that goes along with the data in the row
	// whose "delete" button was clicked
	let id = $(this).data("value");
	$.ajax({
	    type: "DELETE",
	    url: backendUrl + "/messages/" + id,
	    dataType: "json",
	    // TODO: we should really have a function that looks at the return
	    //       value and possibly prints an error message.
	    success: function() {
		ElementList.refresh();
		EntryMenu.close();
	    }
	});
    }

    /**
    * clickEdit is the code we run in response to a click of a delete button
    */
    private static clickEdit() {
	let id = $(this).data("value");
	EditEntryForm.spawn(id);
	}
	
	/**
	 * clickblock is code to send ajax response to backend to block user
	 */
	private static clickBlock() {
		console.log("clicked");
		let blockId = $(this).data("value"); // will get the blocked userid
		// uid will be grabbed by the session in backend
		console.log(blockId);
		$.ajax({
			type: "POST",
			url: backendUrl + "/users/0/block/" + blockId,
			dataType: "json",
			// TODO: we should really have a function that looks at the return
			//       value and possibly prints an error message.
			success: function() {
			ElementList.refresh();
			EntryMenu.close();
			}
		});
	}

	/**
	 * Click event that will flag a post
	 */
	private static clickFlag() {
		console.log("Flag clicked");
		let pid = $(this).data("value");
		console.log(pid);
		$.ajax({
			type: "POST",
			url: backendUrl + "/messages/" + pid + "/flag",
			dataType: "json",
			// TODO: we should really have a function that looks at the return
			//       value and possibly prints an error message.
			success: function() {
			ElementList.refresh();
			EntryMenu.close();
			}
		});
	}

    public static close() {
	$("#" + EntryMenu.NAME).remove()
	}
	
	private static clickUserPage() {
		let uid = localStorage.getItem("uid");
		if (uid != null) { // if uid exists, spawn userpage
			UserPage.spawn(uid);
		}
	}
}
