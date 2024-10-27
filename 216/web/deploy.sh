# deploy script for the web front-end

# This file is responsible for preprocessing all TypeScript files, making sure
# all dependencies are up-to-date, and copying all necessary files into the
# web deploy directory.

# This is the resource folder where maven expects to find our files
TARGETFOLDER=../backend/src/main/resources

# This is the folder that we used with the Spark.staticFileLocation command
WEBFOLDERNAME=web

# step 1: make sure we have someplace to put everything.  We will delete the
#         old folder tree, and then make it from scratch
rm -rf $TARGETFOLDER
mkdir $TARGETFOLDER
mkdir $TARGETFOLDER/$WEBFOLDERNAME

# there are many more steps to be done.  For now, we will just copy an HTML file
cp main.html login.html spec_runner.html $TARGETFOLDER/$WEBFOLDERNAME
cp the-buzz-307823-4c18af9c5803.json $TARGETFOLDER

# step 2: update our npm dependencies
# npm update

# step 3: copy jQuery, Handlebars, and Bootstrap files
cp node_modules/jquery/dist/jquery.min.js $TARGETFOLDER/$WEBFOLDERNAME
cp node_modules/handlebars/dist/handlebars.min.js $TARGETFOLDER/$WEBFOLDERNAME
cp node_modules/bootstrap/dist/js/bootstrap.min.js $TARGETFOLDER/$WEBFOLDERNAME
cp node_modules/bootstrap/dist/css/bootstrap.min.css $TARGETFOLDER/$WEBFOLDERNAME
cp -R node_modules/bootstrap/dist/fonts $TARGETFOLDER/$WEBFOLDERNAME

# step 4: compile TypeScript files
node_modules/.bin/tsc app.ts --strict --outFile $TARGETFOLDER/$WEBFOLDERNAME/app.js
node_modules/.bin/tsc apptest.ts --strict --outFile $TARGETFOLDER/$WEBFOLDERNAME/apptest.js
node_modules/.bin/tsc login.ts --strict --outFile $TARGETFOLDER/$WEBFOLDERNAME/login.js

# step 5: copy css files
cat app.css login.css css/ElementList.css css/EditEntryForm.css css/NewEntryForm.css css/Navbar.css css/EntryMenu.css css/CommentList.css css/UserPage.css > $TARGETFOLDER/$WEBFOLDERNAME/app.css
cp favicon.ico $TARGETFOLDER/$WEBFOLDERNAME

# step 6: compile handlebars templates to the deploy folder
node_modules/handlebars/bin/handlebars hb/ElementList.hb >> $TARGETFOLDER/$WEBFOLDERNAME/templates.js
node_modules/handlebars/bin/handlebars hb/EditEntryForm.hb >> $TARGETFOLDER/$WEBFOLDERNAME/templates.js
node_modules/handlebars/bin/handlebars hb/NewEntryForm.hb >> $TARGETFOLDER/$WEBFOLDERNAME/templates.js
node_modules/handlebars/bin/handlebars hb/Navbar.hb hb/EntryMenu.hb hb/CommentList.hb hb/UserPage.hb >> $TARGETFOLDER/$WEBFOLDERNAME/templates.js

# step 7: add Jasmine Test files
cp node_modules/jasmine-core/lib/jasmine-core/jasmine.css $TARGETFOLDER/$WEBFOLDERNAME
cp node_modules/jasmine-core/lib/jasmine-core/jasmine.js $TARGETFOLDER/$WEBFOLDERNAME
cp node_modules/jasmine-core/lib/jasmine-core/boot.js $TARGETFOLDER/$WEBFOLDERNAME
cp node_modules/jasmine-core/lib/jasmine-core/jasmine-html.js $TARGETFOLDER/$WEBFOLDERNAME