package edu.lehigh.cse216.masa20.backend;

/**
 * SimpleRequest provides a format for clients to present title and message
 * strings to the server.
 *
 * NB: since this will be created from JSON, all fields must be public, and we
 *     do not need a constructor.
 */
public class SimpleRequest {
    // public String mUid; taken out because uid is handled by the session cookie now
    public String mTitle;
    public String mMessage;
    public String mId_token;
    public String mLink;
    public String mDescription;
}
