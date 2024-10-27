package edu.lehigh.cse216.masa20.backend;

// Import the Spark package, so that we can make use of the "get" function to
// create an HTTP GET route
import spark.Spark;
import spark.Request;
import spark.Response;
import java.util.Map;
import java.util.Collections;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Random;
import java.util.Base64;
import java.util.Base64.Decoder;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.lang.ExceptionInInitializerError;
import java.io.IOException;
import java.security.GeneralSecurityException;

//OAuth stuff
import com.google.api.client.googleapis.auth.oauth2.GoogleIdToken;
import com.google.api.client.googleapis.auth.oauth2.GoogleIdToken.Payload;
import com.google.api.client.googleapis.auth.oauth2.GoogleIdTokenVerifier;
import com.google.api.client.auth.oauth2.Credential;
import com.google.auth.oauth2.ServiceAccountCredentials;
import com.google.auth.http.*;

import com.google.api.client.http.HttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.*;


// google drive stuff
import com.google.api.services.drive.Drive;
import com.google.api.services.drive.DriveScopes;
import com.google.api.services.drive.model.File;
import com.google.api.services.drive.model.FileList;

// Memcache
import net.rubyeye.xmemcached.MemcachedClient;
import net.rubyeye.xmemcached.MemcachedClientBuilder;
import net.rubyeye.xmemcached.XMemcachedClientBuilder;
import net.rubyeye.xmemcached.auth.AuthInfo;
import net.rubyeye.xmemcached.command.BinaryCommandFactory;
import net.rubyeye.xmemcached.exception.MemcachedException;
import net.rubyeye.xmemcached.utils.AddrUtil;
import java.lang.InterruptedException;
import java.net.InetSocketAddress;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeoutException;

// Import Google's JSON library
import com.google.gson.*;

//For translate
import java.net.URL;
import java.io.BufferedReader;
import java.net.URLEncoder;
import java.net.HttpURLConnection;
import java.io.InputStreamReader;

public class App {
    // for google drive
    private static final String APPLICATION_NAME = "Google Drive API Java Quickstart";
    private static final JsonFactory JSON_FACTORY = JacksonFactory.getDefaultInstance();

    private static final HttpTransport transport = new NetHttpTransport();
    private static final JsonFactory jsonFactory = new JacksonFactory();
    private static final List<String> SCOPES = Arrays.asList(DriveScopes.DRIVE);
    private static final String CREDENTIALS_FILE_PATH = "/the-buzz-307823-4c18af9c5803.json";

    public static void main(String[] args) {
        // Get the port on which to listen for requests
        Spark.port(getIntFromEnv("PORT", 4567));

        //get environment variables
        Map<String, String> env = System.getenv();

        // gson provides us with a way to turn JSON into objects, and objects
        // into JSON.
        //
        // NB: it must be final, so that it can be accessed from our lambdas
        //
        // NB: Gson is thread-safe.  See
        // https://stackoverflow.com/questions/10380835/is-it-ok-to-use-gson-instance-as-a-static-field-in-a-model-bean-reuse
        final Gson gson = new Gson();

        // db holds all of the data that has been provided via HTTP
        // requests

        // get the Postgres configuration from the environment
        String databaseName = "postgres://mttuejjprtjezy:4c6467ad48c998116f9e2dda61d46b615d9cf6f236ba0281f212cbeb81ca384d@ec2-54-242-43-231.compute-1.amazonaws.com:5432/dacvh86oqpjgot";

        // Get a fully-configured connection to the database, or exit
        Database db = Database.getDatabase(databaseName);
        if (db == null)
            return;
        // db.dropPostTable();
        // db.dropUserTable();
        // db.dropCommentTable();
        // db.dropVoteTable();
        // db.dropBlockTable();
        // db.createPostTable();
        // db.createUserTable();
        // db.createCommentTable();
        // db.createVoteTable();
        // db.createBlockTable();

        // use to set protocal for memcache
        // connect to memcache
        List<InetSocketAddress> servers =
        AddrUtil.getAddresses(System.getenv("MEMCACHED_SERVERS").replace(",", " "));
        AuthInfo authInfo = AuthInfo.plain(System.getenv("MEMCACHED_USERNAME"), System.getenv("MEMCACHED_PASSWORD"));
        MemcachedClientBuilder builder = new XMemcachedClientBuilder(servers);
        // Configure SASL auth for each server
        for(InetSocketAddress server : servers) {
            builder.addAuthInfo(server, authInfo);
        }

        // Use binary protocol
        builder.setCommandFactory(new BinaryCommandFactory());
        // Connection timeout in milliseconds (default: )
        builder.setConnectTimeout(1000);
        // Reconnect to servers (default: true)
        builder.setEnableHealSession(true);
        // Delay until reconnect attempt in milliseconds (default: 2000)
        builder.setHealSessionInterval(2000);

        MemcachedClient mc = memcachier(builder);
        // Set up the location for serving static files.  If the STATIC_LOCATION
        // environment variable is set, we will serve from it.  Otherwise, serve
        // from "/web"
        String static_location_override = System.getenv("STATIC_LOCATION");
        if (static_location_override == null) {
            Spark.staticFileLocation("/web");
        } else {
            Spark.staticFiles.externalLocation(static_location_override);
        }

        Spark.before((request, response) -> {
            String path = request.pathInfo();
            String uid = request.session(true).attribute("uid");
            String sessionKey = request.session().id();
            String val = "";
            boolean isAuth = true;

            try {
                val = mc.get(sessionKey);
                if(val == null) val = "0";
            } catch (TimeoutException te) {
                System.err.println("Timeout during set or get: " + te.getMessage());
            } catch (InterruptedException ie) {
                System.err.println("Interrupt during set or get: " + ie.getMessage());
            } catch (MemcachedException me) {
                System.err.println("Memcached error during get or set: " + me.getMessage());
            } catch (ExceptionInInitializerError ex) {
                System.err.println("Blanket error");
            }

            try {
                if (uid != null && sessionKey != null && (val.equals(uid))) {
                    isAuth = true;
                } else {
                    System.err.println("Auth failed");
                    isAuth = false;
                }
            } catch (Exception e) {
                System.err.println("Auth error: " + e.toString());
            }
            if (path != null && !path.equals("/login") && !path.equals("/auth") && !isAuth) {
                response.redirect("/login.html");
            }
        });

        // Set up a route for serving the main page
        Spark.get("/", (req, res) -> {
            res.redirect("/main.html");
	    return "";
	});

        Spark.get("/logout", (request, response) -> {
            request.session().removeAttribute("uid");
            response.redirect("https://www.google.com/accounts/Logout?continue=https://appengine.google.com/_ah/logout?continue=/");
            return "";
        });

        //use cors if ENABLE_CORS = TRUE within heroku (which it should)
        String cors_enabled = env.get("ENABLE_CORS");
        if (cors_enabled.equals("TRUE")) {
            final String acceptCrossOriginRequestsFrom = "*";
            final String acceptedCrossOriginRoutes = "GET,PUT,POST,DELETE,OPTIONS";
            // final String supportedRequestHeaders = "Content-Type,Authorization,X-Requested-With,Content-Length,Accept,Origin";
            final String supportedRequestHeaders = "*";
            enableCORS(acceptCrossOriginRequestsFrom, acceptedCrossOriginRoutes, supportedRequestHeaders);
        }

        Spark.post("/auth", (request, response) -> {
            SimpleRequest req = gson.fromJson(request.body(), SimpleRequest.class);
            response.type("application/json");

            String idTokenString = req.mId_token;
            String clientId = env.get("CLIENT_ID");

            GoogleIdToken idToken = null;
            String uid = null;
            String sessionKey = null;

            GoogleIdTokenVerifier verifier = new GoogleIdTokenVerifier.Builder(transport, jsonFactory)
                .setAudience(Collections.singletonList(clientId))
                .build();
            try {
                idToken = verifier.verify(idTokenString);
            } catch (java.security.GeneralSecurityException eSecurity) {
                System.err.println("Token Verification Security Execption" + eSecurity);
            } catch (java.io.IOException eIO) {
                System.err.println("Token Verification IO Execption" + eIO);
            }
            if (idToken != null) {
                Payload payload = idToken.getPayload();
                // Get profile information from payload
                uid = payload.getSubject();
                String email = payload.getEmail();
                if (!email.contains("@lehigh.edu")) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("Error", "Wrong domain", null));
                }
                boolean emailVerified = Boolean.valueOf(payload.getEmailVerified());
                String name = (String) payload.get("name");
                String familyName = (String) payload.get("family_name");
                String givenName = (String) payload.get("given_name");

                boolean isUserExists = db.isUserExists(uid);
                if (!isUserExists) {
                    boolean succ = db.insertUser(uid, email, givenName, familyName);
                    if (!succ) {
                        response.status(500);
                        return gson.toJson(new StructuredResponse("UserInsertError", "Failed to insert the user", null));
                    }
                }

                // Auth is good
                try {
                    mc.add(request.session().id(), 5000, uid);
                } catch (TimeoutException te) {
                    System.err.println("Timeout during set or get: " + te.getMessage());
                } catch (InterruptedException ie) {
                    System.err.println("Interrupt during set or get: " + ie.getMessage());
                } catch (MemcachedException me) {
                    System.err.println("Memcached error during get or set: " + me.getMessage());
                }

                request.session().attribute("uid", uid);

            } else {
                response.status(500);
                return gson.toJson(new StructuredResponse("InvalidTokenError", "Invalid ID token", null));
            }

            response.status(200);
            return gson.toJson(new StructuredResponse("OK", null , new String[]{uid, sessionKey}));
        });


        // GET route that returns all message titles and Ids.  All we do is get
        // the data, embed it in a StructuredResponse, turn it into JSON, and
        // return it.  If there's no data, we return "[]", so there's no need
        // for error handling.
        Spark.get("/messages/:lang", (request, response) -> {
                response.type("application/json");
                String lang = request.params("lang");
                ArrayList<Database.PostData> data = db.selectAllPosts();
                String uid = request.session().attribute("uid");

                //translate to spanish
                if (lang.equals("es")) {
                    for (int i = 0; i < data.size(); i++) {
                        Database.PostData post = data.get(i);
                        post.mSubject = translate("en", "es", post.mSubject);
                        post.mMessage = translate("en", "es", post.mMessage);
                    }
                }

                if (data == null) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "failed to get all messages", null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", null, data));
                }
            });

        // GET route that returns everything for a single row in the db.
        // The ":id" suffix in the first parameter to get() becomes
        // request.params("id"), so that we can get the requested row ID.  If
        // ":id" isn't a number, Spark will reply with a status 500 Internal
        // Server Error.  Otherwise, we have an integer, and the only possible
        Spark.get("/messages/:id", (request, response) -> {
                int idx = Integer.parseInt(request.params("id"));
                response.type("application/json");
                Database.PostData data = db.selectOnePost(idx);
                if (data == null) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", idx + " not found", null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", null, data));
                }
            });

        // POST route for adding a new element to the db.  This will read
        // JSON from the body of the request, turn it into a SimpleRequest
        // object, extract the title and message, insert them, and return the
        // ID of the newly created row.
        Spark.post("/messages", (request, response) -> {
            // NB: if gson.Json fails, Spark will reply with status 500 Internal
            // Server Error
            SimpleRequest req = gson.fromJson(request.body(), SimpleRequest.class);
            response.type("application/json");
            String uid = request.session().attribute("uid");
            String lfile = req.mLink;

            // put optional link into google-drive if user entered something
            if (req.mLink != null && req.mLink != "") {
                // intialize data structures
                NetHttpTransport HTTP_TRANSPORT = null;
                HttpRequestInitializer requestInitializer = null;
                Drive service = null;
                File fileMetadata = null;
                File file = null;

            try {
                // Build a new authorized API client service.
                HTTP_TRANSPORT = GoogleNetHttpTransport.newTrustedTransport();
            } catch (java.security.GeneralSecurityException eSecurity) {
                System.err.println("HTTP_TRANSPORT Security Execption" + eSecurity);
            } catch (java.io.IOException eIO) {
                System.err.println("HTTP_TRANSPORT IO Execption" + eIO);
            }

            try {
                InputStream in = App.class.getResourceAsStream(CREDENTIALS_FILE_PATH);
                requestInitializer = new HttpCredentialsAdapter(ServiceAccountCredentials.fromStream(in)
                    .createScoped(SCOPES));
            } catch (java.io.IOException eIO) {
                System.err.println("requestInitializer exception" + eIO);
            }

            service = new Drive.Builder(HTTP_TRANSPORT, JSON_FACTORY, requestInitializer)
                    .setApplicationName(APPLICATION_NAME)
                    .build();

            try {
                System.err.println("drive service quota" + service.about().get().setFields("user, storageQuota").execute());
                byte[] imageBytes = Base64.getDecoder().decode(lfile);
                FileOutputStream imageOutFile = new FileOutputStream("decode.png");
                imageOutFile.write(imageBytes);
                imageOutFile.close();

                // upload file data
                fileMetadata = new File();
                fileMetadata.setName("red-dot.png");
                java.io.File filePath = new java.io.File("decode.png");
                FileContent mediaContent = new FileContent("image/png", filePath);
                file = service.files().create(fileMetadata, mediaContent).setFields("id").execute();

                // add file to memcachier
                    try {
                    mc.add(file.getId(), 5000, lfile);
                } catch (TimeoutException te) {
                    System.err.println("Timeout during set or get: " +
                                       te.getMessage());
                  } catch (InterruptedException ie) {
                    System.err.println("Interrupt during set or get: " +
                                       ie.getMessage());
                  } catch (MemcachedException me) {
                    System.err.println("Memcached error during get or set: " +
                                       me.getMessage());
                  }
                } catch (java.io.IOException eIO) {
                    System.err.println("Google Drive File Error" + eIO);
                }
                }
                int newId = db.insertPost(uid, req.mTitle, req.mMessage, req.mLink);

                if (newId == -1) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "error performing insertion", null));
                } else {
                response.status(200);
                    return gson.toJson(new StructuredResponse("ok", "" + newId, null));
            }
            });

        // POST route for altering upvote
        Spark.post("/messages/:id/upvote", (request, response) -> {
                response.type("application/json");

                int postId = Integer.parseInt(request.params("id"));
                String uid = request.session().attribute("uid");

                int[] numVotes = db.vote(postId, uid, Database.VoteType.UPVOTE);

                if (numVotes[0] < 0 || numVotes[1] < 0) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "upvote error on post " + postId, null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", null, numVotes));
                }
            });

        // POST route for altering downvote
        Spark.post("/messages/:id/downvote", (request, response) -> {
                response.type("application/json");

                int postId = Integer.parseInt(request.params("id"));
                String uid = request.session().attribute("uid");

                int[] numVotes = db.vote(postId, uid, Database.VoteType.DOWNVOTE);

                if (numVotes == null) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "upvote error on post " + postId, null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", null, numVotes));
                }
            });


        // PUT route for updating a row in the db. This is almost
        // exactly the same as POST
        Spark.put("/messages/:id", (request, response) -> {
            // If we can't get an ID or can't parse the JSON, Spark will send
            // a status 500
            int idx = Integer.parseInt(request.params("id"));
            SimpleRequest req = gson.fromJson(request.body(), SimpleRequest.class);
            response.type("application/json");
            String lfile = req.mLink;

            // put optional link into google-drive if user entered something
            if (req.mLink != null && req.mLink != "") {
                // intialize data structures
                NetHttpTransport HTTP_TRANSPORT = null;
                HttpRequestInitializer requestInitializer = null;
                Drive service = null;
                File fileMetadata = null;
                File file = null;

            try {
                // Build a new authorized API client service.
                HTTP_TRANSPORT = GoogleNetHttpTransport.newTrustedTransport();
            } catch (java.security.GeneralSecurityException eSecurity) {
                System.err.println("HTTP_TRANSPORT Security Execption" + eSecurity);
            } catch (java.io.IOException eIO) {
                System.err.println("HTTP_TRANSPORT IO Execption" + eIO);
            }

            try {
                InputStream in = App.class.getResourceAsStream(CREDENTIALS_FILE_PATH);
                requestInitializer = new HttpCredentialsAdapter(ServiceAccountCredentials.fromStream(in)
                    .createScoped(SCOPES));
            } catch (java.io.IOException eIO) {
                System.err.println("requestInitializer exception" + eIO);
            }

            service = new Drive.Builder(HTTP_TRANSPORT, JSON_FACTORY, requestInitializer)
                    .setApplicationName(APPLICATION_NAME)
                    .build();

            try {
                System.err.println("drive service quota" + service.about().get().setFields("user, storageQuota").execute());
                byte[] imageBytes = Base64.getDecoder().decode(lfile);
                FileOutputStream imageOutFile = new FileOutputStream("decode.png");
                imageOutFile.write(imageBytes);
                imageOutFile.close();

                // upload file data
                fileMetadata = new File();
                fileMetadata.setName("red-dot.png");
                java.io.File filePath = new java.io.File("decode.png");
                FileContent mediaContent = new FileContent("image/png", filePath);
                file = service.files().create(fileMetadata, mediaContent).setFields("id").execute();

                // add file to memcachier
                    try {
                    mc.add(file.getId(), 5000, lfile);
                } catch (TimeoutException te) {
                    System.err.println("Timeout during set or get: " +
                                       te.getMessage());
                  } catch (InterruptedException ie) {
                    System.err.println("Interrupt during set or get: " +
                                       ie.getMessage());
                  } catch (MemcachedException me) {
                    System.err.println("Memcached error during get or set: " +
                                       me.getMessage());
                  }
                } catch (java.io.IOException eIO) {
                    System.err.println("Google Drive File Error" + eIO);
                }
                }
                Database.PostData result = db.updateOnePost(idx, req.mTitle, req.mMessage, lfile);
                if (result == null) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "unable to update row " + idx, null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", null, result));
                }
            });

        // DELETE route for removing a row from the db
        Spark.delete("/messages/:id", (request, response) -> {
                // If we can't get an ID, Spark will send a status 500
                int idx = Integer.parseInt(request.params("id"));
                response.type("application/json");
                // NB: we won't concern ourselves too much with the quality of the
                //     message sent on a successful delete
                boolean result = db.deleteOnePost(idx);
                if (!result) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "unable to delete row " + idx, null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", null, null));
                }
            });

        // GET all comments for message `:id`
        Spark.get("/messages/:id/comments", (request, response) -> {
                response.type("application/json");

                int postId = Integer.parseInt(request.params("id"));
                ArrayList<Database.CommentData> data = db.selectPostComments(postId);
                if (data == null) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "failed to get comments for " + postId, null));
                } else {
                    return gson.toJson(new StructuredResponse("ok", null, data));
                }
            });

        // POST new comment for message `:id`
        Spark.post("/messages/:id/comments", (request, response) -> {
                response.type("application/json");

                CommentRequest req = gson.fromJson(request.body(), CommentRequest.class);
                String uid = request.session().attribute("uid");
                int postId = Integer.parseInt(request.params("id"));

                int cid = db.insertComment(postId, uid, req.mComment);
                if (cid == -1) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "error performing insertion", null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", "" + cid, null));
                }
            });

        // PUT for updating comment `:cid` on message `:id`
        // `:id` is not used
        Spark.put("/messages/:id/comments/:cid", (request, response) -> {
                response.type("application/json");

                int commentId = Integer.parseInt(request.params("cid"));
                CommentRequest req = gson.fromJson(request.body(), CommentRequest.class);
                Database.CommentData result = db.updateComment(commentId, req.mComment);
                if (result == null) {
                    response.status(500);
                    return gson.toJson(new StructuredResponse("error", "unable to update comment " + commentId, null));
                } else {
                    response.status(200);
                    return gson.toJson(new StructuredResponse("ok", null, result));
                }
            });

        // DELETE comment `:cid` on message `:pid`
        // `:pid is not used, but included for the sake of consistency
        Spark.delete("/messages/:pid/comments/:cid", (request, response) -> {
            response.type("application/json");

            int commentId = Integer.parseInt(request.params("cid"));
            int pid = Integer.parseInt(request.params("pid"));
            boolean isSuccess = db.deleteComment(commentId, pid);

            if (!isSuccess) {
                response.status(500);
                return gson.toJson(new StructuredResponse("error", "unable to delete comment " + commentId, null));
            } else {
                response.status(200);
                return gson.toJson(new StructuredResponse("ok", null, null));
            }
        });

        //get user page data
        Spark.get("/users/:uid", (request, response) -> {
            response.type("application/json");
            String uid = request.params("uid");
            Database.UserPageData data = db.selectUserPageData(uid);
            if (data == null) {
                response.status(500);
                return gson.toJson(new StructuredResponse("error", "unable to get user page data " + uid, null));
            } else {
                response.status(200);
                return gson.toJson(new StructuredResponse("ok", null, data));
            }
        });

        //update user page data
        Spark.put("/users/:uid", (request, response) -> {
            response.type("application/json");

            String uid = request.params("uid");
            String description = gson.fromJson(request.body(), SimpleRequest.class).mDescription;

            boolean isSuccess = db.updateUserDescription(uid, description);

            if (!isSuccess) {
                response.status(500);
                return gson.toJson(new StructuredResponse("error", "unable to update description", null));
            } else {
                response.status(200);
                return gson.toJson(new StructuredResponse("ok", null, null));
            }
        });

        //block a user based on blockerUid and blockerUid
        //note: :uid is not grabbed from the url, but rather from the session. it is included for the sake of consistency
        Spark.post("/users/:uid/block/:blockedUid", (request, response) -> {
            response.type("application/json");
            String blockerUid = request.session().attribute("uid");
            String blockedUid = request.params("blockedUid");
            if (blockerUid.equals(blockedUid)) {
                System.out.println("HERE");
                response.status(500);
                return gson.toJson(new StructuredResponse("error", "you cannot block yourself", null));
            }
            boolean isSuccess = db.blockUser(blockerUid, blockedUid);
            if (!isSuccess) {
                response.status(500);
                return gson.toJson(new StructuredResponse("error", "unable to block user", null));
            } else {
                response.status(200);
                return gson.toJson(new StructuredResponse("ok", null, null));
            }
        });

        //flag a post
        Spark.post("/messages/:pid/flag", (request, response) -> {
            response.type("application/json");
            int pid = Integer.parseInt(request.params("pid"));
            if (!db.flagPost(pid)) {
                response.status(500);
                return gson.toJson(new StructuredResponse("error", "unable to flag post", null));
            } else {
                response.status(200);
                return gson.toJson(new StructuredResponse("ok", null, null));
            }
        });

        //get session UID
        Spark.get("/user/uid", (request, response) -> {
            response.type("application/json");
            String uid = request.session().attribute("uid");
            if (uid == null) {
                response.status(500);
                return gson.toJson(new StructuredResponse("error", "unable to get user page data " + uid, null));
            } else {
                response.status(200);
                return gson.toJson(new StructuredResponse("ok", null, uid));
            }
        });
    }

    /**
     * Get an integer environment varible if it exists, and otherwise return the
     * default value.
     *
     * @envar      The name of the environment variable to get.
     * @defaultVal The integer value to use as the default if envar isn't found
     *
     * @returns The best answer we could come up with for a value for envar
     */
    static int getIntFromEnv(String envar, int defaultVal) {
        ProcessBuilder processBuilder = new ProcessBuilder();
        if (processBuilder.environment().get(envar) != null) {
            return Integer.parseInt(processBuilder.environment().get(envar));
        }
        return defaultVal;
    }

    //Query the Google script that Marc made for free google translate.
    private static String translate(String langFrom, String langTo, String text) throws IOException {
        String urlStr = "https://script.google.com/macros/s/AKfycbzawLZdeoOx5TmU6PjoMOnpY1oL5qSu_YlR_XYSyiEJyP28AS4NMYYqvAJr3c1ZCXc/exec" +
                "?q=" + URLEncoder.encode(text, "UTF-8") +
                "&target=" + langTo +
                "&source=" + langFrom;
        URL url = new URL(urlStr);
        StringBuilder response = new StringBuilder();
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestProperty("User-Agent", "Mozilla/5.0");
        BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
        String inputLine;
        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();
        return response.toString();
    }


    private static boolean isAuth(Request request, MemcachedClientBuilder builder) {
        String uid = request.session(true).attribute("uid");
        String sessionKey = request.session().id();
        String val = "";
        try {
            if (uid != null && sessionKey != null && (val.equals(uid))) {
                return true;
            } else {
                System.err.println("Auth failed");
                return false;
            }
        } catch (Exception e) {
            System.err.println("Auth error: " + e.toString());
        }
        return false;
    }

    /**
     * SET up CORS headers for the OPTIONS verb, and for every response that the
     * server sends.  This only needs to be called once.
     *
     * @param origin The server that is allowed to send requests to this server
     * @param methods The allowed HTTP verbs from the above origin
     * @param headers The headers that can be sent with a request from the above
     *                origin
     */
    private static void enableCORS(String origin, String methods, String headers) {
        // Create an OPTIONS route that reports the allowed CORS headers and methods
        Spark.options("/*", (request, response) -> {
                String accessControlRequestHeaders = request.headers("Access-Control-Request-Headers");
                if (accessControlRequestHeaders != null) {
                    response.header("Access-Control-Allow-Headers", accessControlRequestHeaders);
                }
                String accessControlRequestMethod = request.headers("Access-Control-Request-Method");
                if (accessControlRequestMethod != null) {
                    response.header("Access-Control-Allow-Methods", accessControlRequestMethod);
                }
                return "OK";
            });

        // 'before' is a decorator, which will run before any
        // get/post/put/delete.  In our case, it will put three extra CORS
        // headers into the response
        Spark.before((request, response) -> {
                response.header("Access-Control-Allow-Origin", origin);
                response.header("Access-Control-Request-Method", methods);
                response.header("Access-Control-Allow-Headers", headers);
            });
    }

    /*
    * Variable to create final object memcahched
    */
    private static MemcachedClient memcachier(MemcachedClientBuilder builder) {
        try {
            MemcachedClient mc = builder.build();
            return mc;
        } catch (IOException ioe) {
            System.err.println("Couldn't create a connection to Memcached server: " +
                            ioe.getMessage());
        }
        return null;
    }
}
