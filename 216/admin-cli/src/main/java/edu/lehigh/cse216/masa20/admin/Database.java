package edu.lehigh.cse216.masa20.admin;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;

public class Database {
    /**
     * The connection to the database.  When there is no connection, it should
     * be null.  Otherwise, there is a valid open connection
     */
    private Connection mConnection;

    /**
     * A prepared statement for getting all data in the database
     */
    private PreparedStatement mSelectAll;

    /**
     * A prepared statement for getting one row from the database
     */
    private PreparedStatement mSelectOne;

    /**
     * A prepared statement for deleting a row from the database
     */
    private PreparedStatement mDeleteOne;

    /**
     * A prepared statement for inserting into the database
     */
    private PreparedStatement mInsertOne;
    private PreparedStatement mInsertUser;
    private PreparedStatement mInsertLike;
    private PreparedStatement mInsertComments;


    /**
     * A prepared statement for updating a single row in the database
     */
    private PreparedStatement mUpdateOne;
    private PreparedStatement mUpvote;
    private PreparedStatement mDownvote;

    /**
     * A prepared statement for creating the table in our database
     */
    private PreparedStatement mSelectAllUser;

    private PreparedStatement mCreateTable;

    private PreparedStatement mCreateUserTable;
    private PreparedStatement mDropUserTable;
    
    

    private PreparedStatement mCreateLikeTable;
    private PreparedStatement mDropLikeTable;

    private PreparedStatement mCreateCommentsTable;
    private PreparedStatement mDropCommentsTable;
    private PreparedStatement mSelectAllLike;
    private PreparedStatement mSelectAllComments;
    

    /**
     * A prepared statement for dropping the table in our database
     */
    private PreparedStatement mDropTable;

    /**
     * RowData is like a struct in C: we use it to hold data, and we allow
     * direct access to its fields.  In the context of this Database, RowData
     * represents the data we'd see in a row.
     *
     * We make RowData a static class of Database because we don't really want
     * to encourage users to think of RowData as being anything other than an
     * abstract representation of a row of the database.  RowData and the
     * Database are tightly coupled: if one changes, the other should too.
     */
    public static class RowData {
        
        /**
         * The ID of this row of the database
         */
        int mId;
        /**
         * The subject stored in this row
         */
        String mSubject;
        /**
         * The message stored in this row
         */
        String mMessage;
	/**
	 * creation time and date of row
	 */
	String mTimestamp;
	int mUpvotes;
	int mDownvotes;
        /**
         * Construct a RowData object by providing values for its fields
         */
        public RowData(int id, String subject, String message, String timestamp, int upvotes, int downvotes) {
            mId = id;
            mSubject = subject;
            mMessage = message;
	    mTimestamp = timestamp;
	    mUpvotes = upvotes;
	    mDownvotes = downvotes;
        }
    }
    public static class UserRowData {
        String mEmail;
        String mName;
        String mDescription;
        int muserid;

    public UserRowData(int userid, String email, String name, String description) {
        muserid = userid;
        mEmail=email;
        mName=name;
        mDescription=description;
        
    }
}

public static class LikeRowData {
    int mUpvotes;
	int mDownvotes;
    String mSubject;
    int mlideid;

public LikeRowData(int likeid, String email, int up, int down) {
    mlideid = likeid;
    mSubject=email;
    mUpvotes=up;
    mDownvotes=down;
    
}
}

public static class CommentRowData {
    String mEmail;
    String mMessage;
    int mcommentid;

public CommentRowData(int commentid, String email, String description) {
    mcommentid = commentid;
    mEmail=email;
    mMessage=description;
    
}
}
    /**
     * The Database constructor is private: we only create Database objects
     * through the getDatabase() method.
     */
    private Database() {
    }

    /**
     * Get a fully-configured connection to the database
     *
     * @param db_url the name of the database on heroku
     *
     * @return A Database object, or null if we cannot connect properly
     */
    // static Database getDatabase(String ip, String port, String db_url, String user, String pass) {
    static Database getDatabase(String db_url) {
        // Create an un-configured Database object
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
            // NB: we can easily get ourselves in trouble here by typing the
            //     SQL incorrectly.  We really should have things like "tblData"
            //     as constants, and then build the strings for the statements
            //     from those constants.

            // Note: no "IF NOT EXISTS" or "IF EXISTS" checks on table
            // creation/deletion, so multiple executions will cause an exception
            db.mCreateTable = db.mConnection.prepareStatement(
                    "CREATE TABLE tblData (id SERIAL PRIMARY KEY, subject VARCHAR(50) "
                    + "NOT NULL, message VARCHAR(500) NOT NULL, timestamp VARCHAR(50) NOT NULL, upvotes INTEGER DEFAULT 0, downvotes INTEGER DEFAULT 0)");
            db.mDropTable = db.mConnection.prepareStatement("DROP TABLE tblData");

            // Standard CRUD operations
            db.mDeleteOne = db.mConnection.prepareStatement("DELETE FROM tblData WHERE id = ?");
            
            db.mInsertOne = db.mConnection.prepareStatement("INSERT INTO tblData VALUES (default, ?, ?, NOW())", PreparedStatement.RETURN_GENERATED_KEYS); //RETURN_GENERATED_KEYS to be able to get the id in insertRow()
            /////////// SelectAll
            db.mSelectAll = db.mConnection.prepareStatement("SELECT * FROM tblData");
            db.mSelectAllUser = db.mConnection.prepareStatement("SELECT * FROM tblUserData");
            db.mSelectAllLike = db.mConnection.prepareStatement("SELECT * FROM tblLikeData");
            db.mSelectAllComments = db.mConnection.prepareStatement("SELECT * FROM tblCommentsData");

            //////
            db.mSelectOne = db.mConnection.prepareStatement("SELECT * from tblData WHERE id=?");
            db.mUpdateOne = db.mConnection.prepareStatement("UPDATE tblData SET subject = ?, message = ? WHERE id = ?", PreparedStatement.RETURN_GENERATED_KEYS);
	        db.mUpvote = db.mConnection.prepareStatement("UPDATE tblData SET upvotes = upvotes + 1 WHERE id = ?");
	        db.mDownvote = db.mConnection.prepareStatement("UPDATE tblData SET downvotes = downvotes + 1 WHERE id = ?");

            ////USER
            db.mCreateUserTable = db.mConnection.prepareStatement(
                    "CREATE TABLE tblUserData (userid SERIAL PRIMARY KEY, email VARCHAR(20) NOT NULL, name VARCHAR(50) "
                    + "NOT NULL, description VARCHAR(500) NOT NULL)");
            db.mInsertUser = db.mConnection.prepareStatement("INSERT INTO tblUserData VALUES (default, ?, ?, ?) RETURNING userid"); //RETURN_GENERATED_KEYS to be able to get the id in insertRow()

            db.mDropUserTable = db.mConnection.prepareStatement("DROP TABLE tblUserData");
            
            ///LIKE
            db.mCreateLikeTable = db.mConnection.prepareStatement(
                    "CREATE TABLE tblLikeData (likeid SERIAL PRIMARY KEY, subject VARCHAR(50) "
                    + "NOT NULL, upvotes INTEGER DEFAULT 0, downvotes INTEGER DEFAULT 0)");
            db.mInsertLike = db.mConnection.prepareStatement("INSERT INTO tblLikeData VALUES (default, ?, ?, ?) RETURNING likeid"); //RETURN_GENERATED_KEYS to be able to get the id in insertRow()

            db.mDropLikeTable = db.mConnection.prepareStatement("DROP TABLE tblLikeData");
            
            ///COMMENTS
            db.mCreateCommentsTable = db.mConnection.prepareStatement(
                    "CREATE TABLE tblCommentsData (commentid SERIAL PRIMARY KEY, email VARCHAR(20) NOT NULL, message VARCHAR(500) NOT NULL)");
            db.mInsertComments = db.mConnection.prepareStatement("INSERT INTO tblCommentsData VALUES (default, ?, ?) RETURNING commentid"); //RETURN_GENERATED_KEYS to be able to get the id in insertRow()

            db.mDropCommentsTable = db.mConnection.prepareStatement("DROP TABLE tblCommentsData");


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
     *
     * @param subject The subject for this new row
     * @param message The message body for this new row
     *
     * @return The id of the inserted row.
     */
    int insertRow(String subject, String message) {
	int newId = -1;
        try {
            mInsertOne.setString(1, subject);
            mInsertOne.setString(2, message);
	    mInsertOne.executeUpdate();
	    //get the id
	    ResultSet rs = mInsertOne.getGeneratedKeys();
	    if (rs.next()) newId = rs.getInt(1);

        } catch (SQLException e) {
            e.printStackTrace();
        }

        return newId;
    }


    int insertUserRow(String email, String name, String description) {
        int newId = -1;
            try {
                mInsertUser.setString(1, email);
                mInsertUser.setString(2, name);
                mInsertUser.setString(3, description);
                //mInsertUser.executeUpdate();
            //get the id
            ResultSet rs = mInsertUser.executeQuery();
            //ResultSet rs = mInsertUser.getGeneratedKeys();
            if (rs.next()) newId = rs.getInt(1);
    
            } catch (SQLException e) {
                e.printStackTrace();
            }
    
            return newId;
        }



        int insertLikeRow(String subject, Integer up, Integer down) {
            int newId = -1;
                try {
                    mInsertLike.setString(1, subject);
                    mInsertLike.setInt(2, up);
                    mInsertLike.setInt(3, down);
                    //mInsertLike.executeUpdate();
                //get the id
                ResultSet rs = mInsertLike.executeQuery();
                //ResultSet rs = mInsertUser.getGeneratedKeys();
                if (rs.next()) newId = rs.getInt(1);
        
                } catch (SQLException e) {
                    e.printStackTrace();
                }
        
                return newId;
            }
        int insertCommentRow(String email, String message) {
            int newId = -1;
                try {
                    mInsertComments.setString(1, email);
                    mInsertComments.setString(2, message);
                    //mInsertComments.executeUpdate();
                //get the id
                ResultSet rs = mInsertComments.executeQuery();
                //ResultSet rs = mInsertUser.getGeneratedKeys();
                if (rs.next()) newId = rs.getInt(1);
        
                } catch (SQLException e) {
                    e.printStackTrace();
                }
        
                return newId;
            }

    /**
     * Query the database for a list of all subjects and their IDs
     *
     * @return All rows, as an ArrayList
     */
    ArrayList<RowData> selectAll() {
        ArrayList<RowData> res = new ArrayList<RowData>();
        try {
            ResultSet rs = mSelectAll.executeQuery();
            while (rs.next()) {
                res.add(new RowData(rs.getInt("id"),
				    rs.getString("subject"),
				    rs.getString("message"),
				    rs.getString("timestamp"),
				    rs.getInt("upvotes"),
				    rs.getInt("downvotes")));
            }
            rs.close();
            return res;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }
    ArrayList<UserRowData> selectAllUser() {
        ArrayList<UserRowData> res = new ArrayList<UserRowData>();
        try {
            ResultSet rs = mSelectAllUser.executeQuery();
            while (rs.next()) {
                res.add(new UserRowData(rs.getInt("userid"),
				    rs.getString("email"),
				    rs.getString("name"),
				    rs.getString("description")
				   
				   ));
            }
            rs.close();
            return res;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    ArrayList<LikeRowData> selectAllLike() {
        ArrayList<LikeRowData> res = new ArrayList<LikeRowData>();
        try {
            ResultSet rs = mSelectAllLike.executeQuery();
            while (rs.next()) {
                res.add(new LikeRowData(rs.getInt("likeid"),
				    rs.getString("subject"),
				    rs.getInt("upvotes"),
				    rs.getInt("downvotes")
				   
				   ));
            }
            rs.close();
            return res;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    ArrayList<CommentRowData> selectAllComments() {
        ArrayList<CommentRowData> res = new ArrayList<CommentRowData>();
        try {
            ResultSet rs = mSelectAllComments.executeQuery();
            while (rs.next()) {
                res.add(new CommentRowData(rs.getInt("commentid"),
				    rs.getString("email"),
				    rs.getString("message")
				    
				   
				   ));
            }
            rs.close();
            return res;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Get all data for a specific row, by ID
     *
     * @param id The id of the row being requested
     *
     * @return The data for the requested row, or null if the ID was invalid
     */
    RowData selectOne(int id) {
        RowData res = null;
        try {
            mSelectOne.setInt(1, id);
            ResultSet rs = mSelectOne.executeQuery();
            if (rs.next()) {
                res = new RowData(rs.getInt("id"),
				  rs.getString("subject"),
				  rs.getString("message"),
				  rs.getString("timestamp"),
				  rs.getInt("upvotes"),
				  rs.getInt("downvotes"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return res;
    }

    /**
     * Delete a row by ID
     *
     * @param id The id of the row to delete
     *
     * @return number of rows deleted if success. -1 if error.
     */
    int deleteRow(int id) {
        int res = 0;
        try {
            mDeleteOne.setInt(1, id);
            res = mDeleteOne.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
	//res is the number of altered rows. return false if the row was no deleted
	if (res > 0) return res;
	return -1;
    }

    /**
     * Update the message for a row in the database
     *
     * @param id The id of the row to update
     * @param subject The new subject
     * @param message The new message contents
     *
     * @return The number of rows updated, if 0 then -1
     */
    int updateOne(int id, String subject, String message) {
    int res = -1;
    RowData row = null;
        try {
            mUpdateOne.setString(1, subject);
            mUpdateOne.setString(2, message);
            mUpdateOne.setInt(3, id);
            res = mUpdateOne.executeUpdate();
	    ResultSet rs = mUpdateOne.getGeneratedKeys();
	    if (rs.next()) {
		row = new RowData (rs.getInt("id"),
				   rs.getString("subject"),
				   rs.getString("message"),
				   null,
				   -1,
				   -1);
	    }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return res;
    }

    /**
     * Increment upvote for a row.
     * Method made by Jake F
     *
     * @param id The id of the row to update
     *
     * @return true on success false on fail
     */
    boolean upvote(int id) {
        try {
	    mUpvote.setInt(1, id);
	    mUpvote.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
	    return false;
        }
	return true;
    }

    /**
     * Increment downvote for a row.
     * Method made by Jake F
     *
     * @param id The id of the row to update
     *
     * @return true on success false on fail
     */
    boolean downvote(int id) {
        try {
	    mDownvote.setInt(1, id);
	    mDownvote.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
	    return false;
        }
	return true;
    }

    /**
     * Create tblData.  If it already exists, this will print an error
     */
    void createTable() {
        try {
            mCreateTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void createuserTable() {
        try {
            mCreateUserTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void createLikeTable() {
        try {
            mCreateLikeTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void CreateCommentsTable() {
        try {
            mCreateCommentsTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    /**
     * Remove tblData from the database.  If it does not exist, this will print
     * an error.
     */
    void dropTable() {
        try {
            mDropTable.execute();
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
    void dropLikesTable() {
        try {
            mDropLikeTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    void dropCommentsTable() {
        try {
            mDropCommentsTable.execute();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }


}

