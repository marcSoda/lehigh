/// This constant indicates the path to our backend server
const backendUrl = location.hostname === "localhost" ? "http://localhost:4567" : "https://runtime-tremor.herokuapp.com";

function onSignIn(googleUser: any) {
    var profile = googleUser.getBasicProfile();
    var id_token = googleUser.getAuthResponse().id_token;
    console.log("LOGGING IN ");
    $.ajax({
        type: "POST",
        url: backendUrl + "/auth",
        dataType: "json",
        data: JSON.stringify({ mId_token: id_token }),
        success: function(resp: any) {
                console.log("success");
            $(location).attr('href',"/");
        },
        error: function() {
            alert("auth error");
        }
    });
}
