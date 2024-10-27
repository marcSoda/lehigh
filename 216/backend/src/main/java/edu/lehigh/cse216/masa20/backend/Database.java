package edu.lehigh.cse216.masa20.backend;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;

public class Database {
    //State of the database connection. Should be null when not connected
    private Connection mConnection;

    //Postgresql statements to be called in subsequent functions.
    private PreparedStatement mSelectAllPosts;
    private PreparedStatement mSelectOnePost;
    private PreparedStatement mSelectOneUser;
    private PreparedStatement mDeleteOnePost;

    private PreparedStatement mSelectAllUsers;
    private PreparedStatement mSelectAllUserPosts;
    private PreparedStatement mUpdateUserDescription;

    private PreparedStatement mInsertPost;
    private PreparedStatement mInsertUser;
    private PreparedStatement mUpdateOnePost;

    private PreparedStatement mSelectPostComments;
    private PreparedStatement mInsertComment;
    private PreparedStatement mUpdateComment;
    private PreparedStatement mDeleteComment;

    private PreparedStatement mSelectPostVotesForUid;
    private PreparedStatement mSelectPostVotesCountForType;
    private PreparedStatement mInsertVote;
    private PreparedStatement mRemoveVote;
    private PreparedStatement mIncPostUpvote;
    private PreparedStatement mDecPostUpvote;
    private PreparedStatement mIncPostDownvote;
    private PreparedStatement mDecPostDownvote;
    private PreparedStatement mIncPostComment;
    private PreparedStatement mDecPostComment;

    private PreparedStatement mBlockUser;
    private PreparedStatement mSelectBlockedUids;
    private PreparedStatement mFlagPost;

    private PreparedStatement mCreatePostTable;
    private PreparedStatement mDropPostTable;
    private PreparedStatement mCreateUserTable;
    private PreparedStatement mDropUserTable;
    private PreparedStatement mCreateVoteTable;
    private PreparedStatement mDropVoteTable;
    private PreparedStatement mCreateCommentTable;
    private PreparedStatement mDropCommentTable;
    private PreparedStatement mCreateBlockTable;
    private PreparedStatement mDropBlockTable;

    //Contains all fields of a single post.
    public static class PostData {
        int mPostId;
        String mSubject;
        String mMessage;
        String mTimestamp;
        int mUpvotes; //number of upvotes
        int mDownvotes; //number of downvotes
        String mUid; //NOTE: uid is not in the post table, but is needed in frontend
        String mName; //NOTE: name is not in the post table, but is needed in frontend
        int mComments; //number of comments
        String mLink; // Added by Jake
        int mFlag; //for flagging inappropriate content. 0 when unflagged, 1 when flagged

        public PostData(int postId, String subject, String message, String timestamp, int upvotes, int downvotes, String uid, String name, int comments, String link, int flag) {
            mPostId = postId;
            mSubject = subject;
            mMessage = message;
            mTimestamp = timestamp;
            mUpvotes = upvotes;
            mDownvotes = downvotes;
            mUid = uid;
            mName = name;
            mComments = comments;
            mLink = link;
            mFlag = flag;
        }
    }

    //Contains all fields for a single user.
    public static class UserData {
        String mEmail;
        String mName;
        String mDescription;
        String mUid;

        public UserData(String uid, String email, String name, String description) {
            mUid = uid;
            mEmail=email;
            mName=name;
            mDescription=description;
        }
    }

    //All data that goes on a user's page
    public static class UserPageData {
        UserData mUserData;
        ArrayList<PostData> mUserPosts;

        public UserPageData(UserData userData, ArrayList<PostData> userPosts) {
            mUserData = userData;
            mUserPosts = userPosts;
        }
    }

    //Contains all fields for a single comment.
    public static class CommentData {
        int mCommentId;
        int mPostId;
        String mUid;
        String mComment;
        String mName;

        public CommentData(int commentId, int postId, String uid, String comment, String name) {
            mCommentId = commentId;
            mPostId = postId;
            mUid = uid;
            mComment = comment;
            mName = name;
        }
    }

    //enum for abstracting votetypes. helps eliminate duplicate code for voting
    public enum VoteType {
        UPVOTE(1), DOWNVOTE(2);

        int value;

        private VoteType(int value) {
            this.value = value;
        }

        public static VoteType fromValue(int value) {
            for (VoteType vt : VoteType.values()) {
                if (vt.value == value) {
                    return vt;
                }
            }
            return null;
        }
    }

    //tracks wether a vote exists and the type
    //necessary because returning a tuple in java is ridiculus
    public static class VoteQuery {
        boolean isExists;
        VoteType type;

        public VoteQuery(boolean isExists, VoteType type) {
            this.isExists = isExists;
            this.type = type;
        }
    }

    private Database() {
    }

    /**
     * Get a fully-configured connection to the database
     *
     * @param db_url the name of the database on heroku
     *
     * @return A Database object, or null if we cannot connect properly
     */
    static Database getDatabase(String db_url) {
        Database db = new Database();

        // Give the Database object a connection, fail if we cannot get one
        try {
            Class.forName("org.postgresql.Driver");
            URI dbUri = new URI(db_url);
            String username = dbUri.getUserInfo().split(":")[0];
            String password = dbUri.getUserInfo().split(":")[1];
            String dbUrl = "jdbc:postgresql://" + dbUri.getHost() + ':' + dbUri.getPort() + dbUri.getPath() + "?sslmode=require";
            Connection conn = DriverManager.getConnection(dbUrl, username, password);
            if (conn == null) {
                System.err.println("Error: DriverManager.getConnection() returned a null object");
                return null;
            }
            db.mConnection = conn;
        } catch (SQLException e) {
            System.err.println("Error: DriverManager.getConnection() threw a SQLException");
            e.printStackTrace();
            return null;
        } catch (ClassNotFoundException cnfe) {
            System.out.println("Unable to find postgresql driver");
            return null;
        } catch (URISyntaxException s) {
            System.out.println("URI Syntax Error");
            return null;
        }

        // Attempt to create all of our prepared statements.  If any of these
        // fail, the whole getDatabase() call should fail
        try {
            // Note: no "IF NOT EXISTS" or "IF EXISTS" checks on table
            // creation/deletion, so multiple executions will cause an exception
            db.mCreatePostTable = db.mConnection.prepareStatement(
                                                                  "CREATE TABLE postData (" +
                                                                  "postId SERIAL PRIMARY KEY," +
                                                                  "uid VARCHAR(50)," +
                                                                  "subject VARCHAR(50) NOT NULL," +
                                                                  "message VARCHAR(500) NOT NULL," +
                                                                  "timestamp VARCHAR(50) NOT NULL," +
                                                                  "upvotes INTEGER DEFAULT 0," +
                                                                  "downvotes INTEGER DEFAULT 0," +
                                                                  "comments INTEGER DEFAULT 0," +
                                                                  "link VARCHAR(10485760), " +
                                                                  "flag INTEGER DEFAULT 0)");
            db.mCreateUserTable = db.mConnection.prepareStatement(
                                                                  "CREATE TABLE userData (" +
                                                                  "uid VARCHAR(50) PRIMARY KEY," +
                                                                  "email VARCHAR(50) NOT NULL," +
                                                                  "firstName VARCHAR(50)," +
                                                                  "lastName VARCHAR(50) NOT NULL," +
                                                                  "description VARCHAR(500))");
            db.mCreateCommentTable = db.mConnection.prepareStatement(
                                                                     "CREATE TABLE comments (" +
                                                                     "commentId SERIAL PRIMARY KEY," +
                                                                     "postId INTEGER," +
                                                                     "uid VARCHAR(50)," +
                                                                     "comment VARCHAR(200))");
            db.mCreateVoteTable = db.mConnection.prepareStatement(
                                                                    "CREATE TABLE votes (" +
                                                                    "postId INTEGER, " +
                                                                    "uid VARCHAR(50), " +
                                                                    "type INTEGER, " +
                                                                    "PRIMARY KEY(postId, uid))");
            db.mCreateBlockTable = db.mConnection.prepareStatement("CREATE TABLE blocked (" +
                                                                   "blockerUid VARCHAR(50), " +
                                                                   "blockedUid VARCHAR(50), " +
                                                                   "PRIMARY KEY(blockerUid, blockedUid))");
            db.mDropPostTable = db.mConnection.prepareStatement(
                                                                "DROP TABLE postData");
            db.mDropUserTable = db.mConnection.prepareStatement(
                                                                "DROP TABLE userData");
            db.mDropVoteTable = db.mConnection.prepareStatement(
                                                                  "DROP TABLE votes");
            db.mDropCommentTable = db.mConnection.prepareStatement(
                                                                   "DROP TABLE comments");
            db.mDropBlockTable = db.mConnection.prepareStatement(
                                                                   "DROP TABLE blocked");

            db.mDeleteOnePost = db.mConnection.prepareStatement(
                                                                "DELETE FROM postData " +
                                                                "WHERE postId = ?");
            db.mInsertPost = db.mConnection.prepareStatement(
                                                             "INSERT INTO postData (postId, uid, subject, message, timestamp, link)" +
                                                             "VALUES (default, ?, ?, ?, NOW(), ?)", PreparedStatement.RETURN_GENERATED_KEYS); //RETURN_GENERATED_KEYS to be able to get the id in insertRow()
            db.mSelectAllPosts = db.mConnection.prepareStatement(
                                    "SELECT postId, postData.uid, subject, message, timestamp, upvotes, downvotes, firstName, lastName, comments, link, flag " +
                                    "FROM postData " +
                                    "INNER JOIN userData " +
                                    "ON userData.uid = postData.uid " +
                                    "WHERE userData.uid NOT IN " +
                                        "(SELECT blockedUid FROM blocked)");
            db.mSelectAllUsers = db.mConnection.prepareStatement(
                                                                 "SELECT * FROM userData");
            db.mSelectAllUserPosts = db.mConnection.prepareStatement(
                                                                     "SELECT * FROM postData " +
                                                                     "WHERE uid = ?");
            db.mUpdateUserDescription = db.mConnection.prepareStatement(
                                                                       "UPDATE userData " +
                                                                       "SET description = ? " +
                                                                       "WHERE uid = ?");

	    db.mSelectOneUser = db.mConnection.prepareStatement(
								"SELECT * FROM userData " +
								"WHERE uid = ?");
            db.mSelectOnePost = db.mConnection.prepareStatement(
                                                                "SELECT * from postData " +
                                                                "WHERE postId=?");
            db.mUpdateOnePost = db.mConnection.prepareStatement(
                                                                "UPDATE postData " +
                                                                "SET subject = ?, message = ?, link = ?" +
                                                                "WHERE postId = ?", PreparedStatement.RETURN_GENERATED_KEYS);
            db.mInsertUser = db.mConnection.prepareStatement(
                                                             "INSERT INTO userData (uid, email, firstName, lastName)" +
                                                             "VALUES (?, ?, ?, ?)");

            // Comments
            db.mSelectPostComments = db.mConnection.prepareStatement(
                                                                     "SELECT commentId, postId, comments.uid, comment, firstName, lastName " +
                                                                     "FROM comments " +
                                                                     "INNER JOIN userData " +
                                                                     "ON userData.uid = comments.uid " +
                                                                     "WHERE postId = ?");
            db.mInsertComment = db.mConnection.prepareStatement(
                                                                "INSERT INTO comments " +
                                                                "VALUES (default, ?, ?, ?)",
                                                                PreparedStatement.RETURN_GENERATED_KEYS);
            db.mUpdateComment = db.mConnection.prepareStatement(
                                                                "UPDATE comments " +
                                                                "SET comment = ? " +
                                                                "WHERE commentId = ?",
                                                                PreparedStatement.RETURN_GENERATED_KEYS);
            db.mDeleteComment = db.mConnection.prepareStatement(
                                                                "DELETE FROM comments " +
                                                                "WHERE commentId = ?");

            // Votes
            db.mSelectPostVotesForUid = db.mConnection.prepareStatement(
                                                                        "SELECT * FROM votes " +
                                                                        "WHERE postId = ? " +
                                                                        "AND uid = ?");
            db.mSelectPostVotesCountForType = db.mConnection.prepareStatement(
                                                                              "SELECT COUNT(*) FROM votes " +
                                                                              "WHERE postId = ? " +
                                                                              "AND type = ?");
            db.mInsertVote = db.mConnection.prepareStatement(
                                                             "INSERT INTO votes " +
                                                             "VALUES (?, ?, ?)");
            db.mRemoveVote = db.mConnection.prepareStatement(
                                                             "DELETE FROM votes " +
                                                             "WHERE postId = ? " +
                                                             "AND uid = ?");
            db.mIncPostUpvote = db.mConnection.prepareStatement(
                                                                "UPDATE postData " +
                                                                "SET upvotes = upvotes + 1 " +
                                                                "WHERE postId = ?");
            db.mDecPostUpvote = db.mConnection.prepareStatement(
                                                                "UPDATE postData " +
                                                                "SET upvotes = upvotes - 1 " +
                                                                "WHERE postId = ?");
            db.mIncPostDownvote = db.mConnection.prepareStatement(
                                                                  "UPDATE postData " +
                                                                  "SET downvotes = downvotes + 1 " +
                                                                  "WHERE postId = ?");
            db.mDecPostDownvote = db.mConnection.prepareStatement(
                                                                  "UPDATE postData " +
                                                                  "SET downvotes = downvotes - 1 " +
                                                                  "WHERE postId = ?");
            db.mIncPostComment = db.mConnection.prepareStatement(
                                                                  "UPDATE postData " +
                                                                  "SET comments = comments + 1 " +
                                                                  "WHERE postId = ?");
            db.mDecPostComment = db.mConnection.prepareStatement(
                                                                  "UPDATE postData " +
                                                                  "SET comments = comments - 1 " +
                                                                  "WHERE postId = ?");
            db.mBlockUser = db.mConnection.prepareStatement("INSERT INTO blocked (blockerUid, blockedUid) " +
                                                            "VALUES (?, ?)");
            db.mFlagPost = db.mConnection.prepareStatement("UPDATE postData " +
                                                           "SET flag = ? " +
                                                           "WHERE postId = ?");

        } catch (SQLException e) {
            System.err.println("Error creating prepared statement");
            e.printStackTrace();
            db.disconnect();
            return null;
        }
        return db;
    }

    /**
     * Close the current connection to the database, if one exists.
     *
     * NB: The connection will always be null after this call, even if an
     *     error occurred during the closing operation.
     *
     * @return True if the connection was cleanly closed, false otherwise
     */
    boolean disconnect() {
        if (mConnection == null) {
            System.err.println("Unable to close connection: Connection was null");
            return false;
        }
        try {
            mConnection.close();
        } catch (SQLException e) {
            System.err.println("Error: Connection.close() threw a SQLException");
            e.printStackTrace();
            mConnection = null;
            return false;
        }
        mConnection = null;
        return true;
    }

    /**
     * Insert a row into the database
     * @param uid The user ID associated with this post.
     * @param subject The heading of this post.
     * @param message The content of this post.
     * @return The postId of the inserted row.
     */
    int insertPost(String uid, String subject, String message, String link) {
        int newPostId = -1;
        try {
            mInsertPost.setString(1, uid);
            mInsertPost.setString(2, subject);
            mInsertPost.setString(3, message);
            mInsertPost.setString(4, link);
            mInsertPost.executeUpdate();
            //get the postId
            ResultSet rs = mInsertPost.getGeneratedKeys();
            if (rs.next()) newPostId = rs.getInt(1);

        } catch (SQLException e) {
            e.printStackTrace();
        }

        return newPostId;
    }

    //Insert a user into the db
    boolean insertUser(String uid, String email, String firstName, String lastName) {
        try {
            mInsertUser.setString(1, uid);
            mInsertUser.setString(2, email);
            mInsertUser.setString(3, firstName);
            mInsertUser.setString(4, lastName);
            mInsertUser.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    //Get all posts from db.
    ArrayList<PostData> selectAllPosts() {
        ArrayList<PostData> res = new ArrayList<PostData>();
        try {
            ResultSet rs = mSelectAllPosts.executeQuery();
            while (rs.next()) {
                res.add(new PostData(rs.getInt("postId"),
                                     rs.getString("subject"),
                                     rs.getString("message"),
                                     rs.getString("timestamp"),
                                     rs.getInt("upvotes"),
                                     rs.getInt("downvotes"),
                                     rs.getString("uid"),
                                     rs.getString("firstName") + " " + rs.getString("lastName"),
                                     rs.getInt("comments"),
                                     rs.getString("link"),
                                     rs.getInt("flag")));
            }
            rs.close();
            return res;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    //Get all users from db. (Not sure why we would ever need this... I don't even know if it works.)
    ArrayList<UserData> selectAllUsers() {
        ArrayList<UserData> res = new ArrayList<UserData>();
        try {
            ResultSet rs = mSelectAllUsers.executeQuery();
            while (rs.next()) {
                res.add(new UserData(rs.getString("uid"),
                                     rs.getString("email"),
                                     rs.getString("name"),
                                     rs.getString("description")));
            }
            rs.close();
            return res;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    //Get all user data for a user page
    UserPageData selectUserPageData(String uid) {
        UserData userData = null;
        ArrayList<PostData> allPostData = new ArrayList<PostData>();
	try {
            mSelectOneUser.setString(1, uid);
	    ResultSet rs = mSelectOneUser.executeQuery();
            if (rs.next()) {
                userData = new UserData(rs.getString("uid"),
                                        rs.getString("email"),
                                        rs.getString("firstName") + " " + rs.getString("lastName"),
                                        rs.getString("description"));
            }
	} catch (SQLException e) {
	    e.printStackTrace();
	    return null;
	}

        try {
            mSelectAllUserPosts.setString(1, uid);
            ResultSet rs = mSelectAllUserPosts.executeQuery();
            while (rs.next()) {
                allPostData.add(new PostData(rs.getInt("postId"),
                                     rs.getString("subject"),
                                     rs.getString("message"),
                                     rs.getString("timestamp"),
                                     rs.getInt("upvotes"),
                                     rs.getInt("downvotes"),
                                     rs.getString("uid"),
                                     userData.mName,
                                     rs.getInt("comments"),
                                     rs.getString("link"),
                                     rs.getInt("flag")));
            }
            rs.close();
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }

        return new UserPageData(userData, allPostData);
    }

    boolean updateUserDescription(String uid, String description) {
        if (!isUserExists(uid)) return false;

        try {
            mUpdateUserDescription.setString(1, description);
            mUpdateUserDescription.setString(2, uid);
            mUpdateUserDescription.executeUpdate();
        } catch (Exception e) {
            System.out.println("insertUserData error:" + e);
            return false;
        }

        return true;
    }

    //Get all post data for one post.
    PostData selectOnePost(int postId) {
        PostData res = null;
        try {
            mSelectOnePost.setInt(1, postId);
            ResultSet rs = mSelectOnePost.executeQuery();
            if (rs.next()) {
                res = new PostData(rs.getInt("postId"),
                                   rs.getString("subject"),
                                   rs.getString("message"),
                                   rs.getString("timestamp"),
                                   rs.getInt("upvotes"),
                                   rs.getInt("downvotes"),
                                   null,
                                   null,
                                   rs.getInt("comments"),
                                   rs.getString("link"),
                                   rs.getInt("flag"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return res;
    }

    //check if user exists in db
    boolean isUserExists(String uid) {
        try {
            mSelectOneUser.setString(1, uid);
            ResultSet rs = mSelectOneUser.executeQuery();
            if (!rs.isBeforeFirst()) {
                return false;
            }
            return true;
        } catch (SQLException e) {
            e.printStackTrace();
            return false;
        }
    }

    //Delete a post
    boolean deleteOnePost(int postId) {
        int res = 0;
        try {
            mDeleteOnePost.setInt(1, postId);
            res = mDeleteOnePost.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        //res is the number of altered rows. return false if the row was no deleted
        if (res > 0) return true;
        return false;
    }

    //PUT post
    PostData updateOnePost(int postId, String subject, String message, String link) {
        PostData row = null;
        try {
            mUpdateOnePost.setString(1, subject);
            mUpdateOnePost.setString(2, message);
            mUpdateOnePost.setString(3, link);
            mUpdateOnePost.setInt(4, postId);
            mUpdateOnePost.executeUpdate();
            ResultSet rs = mUpdateOnePost.getGeneratedKeys();
            if (rs.next()) {
                row = new PostData (rs.getInt("postId"),
                                    rs.getString("subject"),
                                    rs.getString("message"),
                                    rs.getString("timestamp"),
                                    rs.getInt("upvotes"),
                                    rs.getInt("downvotes"),
                                    null,
                                    null,
                                    rs.getInt("comments"),
                                    rs.getString("link"),
                                    rs.getInt("flag"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return row;
    }

    // Get all comments associated with a single post
    ArrayList<CommentData> selectPostComments(int postId) {
        ArrayList<CommentData> data = new ArrayList<CommentData>();

        try {
            mSelectPostComments.setInt(1, postId);
            ResultSet rs = mSelectPostComments.executeQuery();
            while (rs.next()) {
                data.add(new CommentData(rs.getInt("commentId"),
                                         rs.getInt("postId"),
                                         rs.getString("uid"),
                                         rs.getString("comment"),
                                         rs.getString("firstName") + " " + rs.getString("lastName")));
            }
            rs.close();
            return data;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Create new comment
    int insertComment(int postId, String uid, String comment) {
        int commentId = -1;

        try {
            mInsertComment.setInt(1, postId);
            mInsertComment.setString(2, uid);
            mInsertComment.setString(3, comment);
            mInsertComment.executeUpdate();
            ResultSet rs = mInsertComment.getGeneratedKeys();
            if (rs.next()) commentId = rs.getInt(1);

            mIncPostComment.setInt(1, postId);
            mIncPostComment.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }

        return commentId;
    }

    // Update comment
    CommentData updateComment(int commentId, String comment) {
        CommentData row = null;

        try {
            mUpdateComment.setString(1, comment);
            mUpdateComment.setInt(2, commentId);
            mUpdateComment.executeUpdate();
            ResultSet rs = mUpdateComment.getGeneratedKeys();
            if (rs.next()) {
                row = new CommentData(rs.getInt("commentId"),
                                      rs.getInt("postId"),
                                      rs.getString("uid"),
                                      rs.getString("comment"),
                                      null);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        return row;
    }

    // Delete a comment
    boolean deleteComment(int commentId, int postId) {
        int res = 0;

        try {
            mDeleteComment.setInt(1, commentId);
            res = mDeleteComment.executeUpdate();
            mDecPostComment.setInt(1, postId);
            mDecPostComment.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }

        // res is the number of altered rows. return false if the row was no deleted
        if (res > 0) return true;
        return false;
    }

    // Insert vote into table. Fails on dups.
    void insertVote(int postId, String uid, VoteType type) {
        try {
            mInsertVote.setInt(1, postId);
            mInsertVote.setString(2, uid);
            mInsertVote.setInt(3, type.value);
            mInsertVote.executeUpdate();

            if (type == VoteType.UPVOTE) {
                mIncPostUpvote.setInt(1, postId);
                mIncPostUpvote.executeUpdate();
            } else if (type == VoteType.DOWNVOTE) {
                mIncPostDownvote.setInt(1, postId);
                mIncPostDownvote.executeUpdate();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    // Remove vote
    void removeVote(int postId, String uid) {
        try {
            mRemoveVote.setInt(1, postId);
            mRemoveVote.setString(2, uid);
            mRemoveVote.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    //handles voting based on postId, uid, and vote type (to reduce duplicate code)
    int[] vote(int postId, String uid, VoteType type) {
        try {
            VoteQuery ve = checkForVote(postId, uid);

            if (ve.isExists) {
                removeVote(postId, uid);
                // Only adjust the post table if the types match
                if (ve.type == type) {
                    if (type == VoteType.UPVOTE) {
                        mDecPostUpvote.setInt(1, postId);
                        mDecPostUpvote.executeUpdate();
                    } else if (type == VoteType.DOWNVOTE) {
                        mDecPostDownvote.setInt(1, postId);
                        mDecPostDownvote.executeUpdate();
                    }
                    return voteCounts(postId);
                }
                if (type == VoteType.UPVOTE) {
                    mDecPostDownvote.setInt(1, postId);
                    mDecPostDownvote.executeUpdate();
                } else if (type == VoteType.DOWNVOTE) {
                    mDecPostUpvote.setInt(1, postId);
                    mDecPostUpvote.executeUpdate();
                }
            }

            insertVote(postId, uid, type);

            return voteCounts(postId);
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Find vote count
    int[] voteCounts(int postId) throws SQLException {
        int[] counts = new int[2];
        mSelectPostVotesCountForType.setInt(1, postId);
        mSelectPostVotesCountForType.setInt(2, VoteType.UPVOTE.value);
        ResultSet rs = mSelectPostVotesCountForType.executeQuery();
        if (rs.isBeforeFirst()) {
            rs.next();
            counts[0] = rs.getInt(1);
        } else return null;

        mSelectPostVotesCountForType.setInt(2, VoteType.DOWNVOTE.value);
        rs = mSelectPostVotesCountForType.executeQuery();
        if (rs.isBeforeFirst()) {
            rs.next();
            counts[1] = rs.getInt(1);
        } else return null;

        return counts;
    }

    // Check that that vote for (postId, uid) exists.
    VoteQuery checkForVote(int postId, String uid) throws SQLException {
        mSelectPostVotesForUid.setInt(1, postId);
        mSelectPostVotesForUid.setString(2, uid);

        ResultSet rs = mSelectPostVotesForUid.executeQuery();
        if (rs.isBeforeFirst()) {
            rs.next();
            VoteType type = VoteType.fromValue(rs.getInt("type"));
            return new VoteQuery(true, type);
        }

        return new VoteQuery(false, null);
    }

    //Block a user based on blockerUid and blockedUid
    boolean blockUser(String blockerUid, String blockedUid) {
        try {
            mBlockUser.setString(1, blockerUid);
            mBlockUser.setString(2, blockedUid);
            mBlockUser.executeUpdate();
        } catch (SQLException e) {
            System.err.println("ERROR BLOCKING USER");
            e.printStackTrace();
            return false;
        }
        return true;

    }

    boolean flagPost(int postId) {
        try {
            mFlagPost.setInt(1, 1);
            mFlagPost.setInt(2, postId);
            mFlagPost.executeUpdate();
        } catch (SQLException e) {
            System.err.println("ERROR FLAGGING POST");
            e.printStackTrace();
            return false;
        }
        return true;
    }

    void createPostTable() {
        try {
            mCreatePostTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void createUserTable() {
        try {
            mCreateUserTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void createVoteTable() {
        try {
            mCreateVoteTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void createCommentTable() {
        try {
            mCreateCommentTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void createBlockTable() {
        try {
            mCreateBlockTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void dropPostTable() {
        try {
            mDropPostTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void dropUserTable() {
        try {
            mDropUserTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void dropVoteTable() {
        try {
            mDropVoteTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void dropCommentTable() {
        try {
            mDropCommentTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void dropBlockTable() {
        try {
            mDropBlockTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
