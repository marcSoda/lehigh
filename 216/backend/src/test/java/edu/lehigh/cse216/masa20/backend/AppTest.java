package edu.lehigh.cse216.masa20.backend;

import java.util.ArrayList;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/*
 * Unit test for simple App.
 */
public class AppTest
    extends TestCase
{
    /*
     * Create the test case
     *
     * @param testName name of the test case
     */
    public AppTest( String testName )
    {
        super( testName );
    }

    /*
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( AppTest.class );
    }

    /**
     * Unit test to test row data
     */
    public void testTest()
    {
        assertTrue(true);
    }

    /**
     * Unit test done by Jake F to test RowData constructor
     */
    public void testPostDataConstructor()
    {
        Database.PostData test = new Database.PostData(1, "subject", "this is a message", "20:30", 6, 2, "1", "jaku", 12, "https://herkou.com", 0);
        assertTrue(test.mPostId == 1);
        assertTrue(test.mSubject.equals("subject"));
        assertTrue(test.mMessage.equals("this is a message"));
        assertTrue(test.mTimestamp.equals("20:30"));
        assertTrue(test.mUpvotes == 6);
        assertTrue(test.mDownvotes == 2);
        assertTrue(test.mUid.equals("1"));
        assertTrue(test.mName.equals("jaku"));
        assertTrue(test.mLink.equals("https://herkou.com"));
    }

    public void testUserDataConstructor()
    {
        Database.UserData test = new Database.UserData("1", "jmf223", "jake", "hi");
        assertTrue(test.mUid.equals("1"));
        assertTrue(test.mEmail.equals("jmf223"));
        assertTrue(test.mName.equals("jake"));
        assertTrue(test.mDescription.equals("hi"));
    }

    public void testUserPageDataConstructor()
    {
        ArrayList<Database.PostData> posts = new ArrayList<Database.PostData>();
        Database.PostData post = new Database.PostData(1, "subject", "this is a message", "20:30", 6, 2, "1", "jaku", 12, "https://herkou.com", 0);
        posts.add(post);
        Database.UserData user = new Database.UserData("1", "jmf223", "jake", "hi");
        Database.UserPageData test = new Database.UserPageData(user, posts);
        assertTrue(test.mUserData.equals(user));
        assertTrue(test.mUserPosts.equals(posts));
    }

    public void testCommentDataConstructor()
    {
        Database.CommentData test = new Database.CommentData(69, 420, "69420", "haha", "funny number");
        assertTrue(test.mCommentId == 69);
        assertTrue(test.mPostId == 420);
        assertTrue(test.mUid.equals("69420"));
        assertTrue(test.mComment.equals("haha"));
        assertTrue(test.mName.equals("funny number"));
    }

    //A test to test the VoteQuery class and VoteType enum
    public void testVote() {
        Database.VoteQuery vq = new Database.VoteQuery(true, Database.VoteType.UPVOTE);
        assertTrue(vq.isExists == true);
        assertTrue(vq.type == Database.VoteType.UPVOTE);
    }
}
