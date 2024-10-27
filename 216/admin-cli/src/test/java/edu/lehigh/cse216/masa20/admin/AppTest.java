package edu.lehigh.cse216.masa20.admin;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class AppTest 
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public AppTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( AppTest.class );
    }

    /**
     * Rigourous Test :-)
     */
    public void testApp()
    {
        assertTrue( true );
    }

    /**
     * Unit test done by Jake F to test RowData constructor
     */
    public void testRowDataConstructor()
    {
        Database.RowData test = new Database.RowData(1, "subject", "this is a message", "20:30", 6, 2);
        assertTrue(test.mId == 1);
        assertTrue(test.mSubject.equals("subject"));
        assertTrue(test.mMessage.equals("this is a message"));
        assertTrue(test.mTimestamp.equals("20:30"));
        assertTrue(test.mUpvotes == 6);
        assertTrue(test.mDownvotes == 2);

        Database.UserRowData test2 = new Database.UserRowData(2, "anv223@lehigh.edu", "this is a name", "This is description");
        assertTrue(test2.muserid == 2);
        assertTrue(test2.mEmail.equals("anv223@lehigh.edu"));
        assertTrue(test2.mName.equals("this is a name"));
        assertTrue(test2.mDescription.equals("This is description"));
        
        
        Database.LikeRowData test3 = new Database.LikeRowData(3, "email", 3, 2);
        assertTrue(test3.mlideid == 3);
        assertTrue(test3.mSubject.equals("email"));
        assertTrue(test3.mUpvotes == 3);
        assertTrue(test3.mDownvotes ==2);
        

        Database.CommentRowData test4 = new Database.CommentRowData(1, "email", "description");
        assertTrue(test4.mcommentid == 1);
        assertTrue(test4.mEmail.equals("email"));
        assertTrue(test4.mMessage.equals("description"));
        


    }

}
