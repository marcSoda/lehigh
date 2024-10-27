/**
 * The Navbar Singleton is the navigation bar at the top of the page.  Through
 * its HTML, it is designed so that clicking the "brand" part will refresh the
 * page.  Apart from that, it has an "add" button, which forwards to
 * NewEntryForm
 */
class Navbar {
    /**
     * Track if the Singleton has been initialized
     */
    public static isInit = false;

    /**
     * The name of the DOM entry associated with Navbar
     */
    private static readonly NAME = "Navbar";

    /**
     * Initialize the Navbar Singleton by creating its element in the DOM and
     * configuring its button.  This needs to be called from any public static
     * method, to ensure that the Singleton is initialized before use.
     */
    private static init() {
        if (!Navbar.isInit) {
            $("body").prepend(Handlebars.templates[Navbar.NAME + ".hb"]());
            $("#"+Navbar.NAME+"-add-link").click(NewEntryForm.show);
            $("#"+Navbar.NAME+"-logout-link").click(Navbar.logout);
            $("#"+Navbar.NAME+"-account-link").click(Navbar.clickAccount);
            $("#"+Navbar.NAME+"-themeTog").click(Navbar.clickTheme);
            $("#"+Navbar.NAME+"-langTog").click(Navbar.clickLang);
            Navbar.isInit = true;
        }
    }

    /**
     * Refresh() doesn't really have much meaning for the navbar, but we'd
     * rather not have anyone call init(), so we'll have this as a stub that
     * can be called during front-end initialization to ensure the navbar
     * is configured.
     */
    public static refresh() {
        Navbar.init();
    }

    private static clickAccount() {
        //MUST START STORING UID IN THE DOM FOR THE BUTTON IN NAVBAR
    let uid = localStorage.getItem('luid');
    if (uid != null) {
    UserPage.spawn(uid);
    }
    }

    private static logout() {
	// $(location).attr('href', 'https://www.google.com/accounts/Logout?continue=https://appengine.google.com/_ah/logout');
        $(location).attr('href', '/logout');
    }

    // switch language of message text from en -> es or es -> en
    private static clickLang() {
        let lang: any = localStorage.getItem("lang"); // get lang that is currently being used "en" or "es".
        // toggle language
        if (lang == "es") {
            lang = "en";
        }
        else if (lang == "en") {
            lang = "es";
        }
        console.log(lang);
        localStorage.setItem("lang", lang); // set toggled language in local storage
        ElementList.refresh(); // update messages
    }

    // toggle theme in app.css
    private static clickTheme() {
        document.body.classList.toggle('light');
    }

}
