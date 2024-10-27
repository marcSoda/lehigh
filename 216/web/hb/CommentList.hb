<div id="CommentList">
    {{#each mData}}
        <div id="{{this.mCommentId}}" class="CommentList-comment" data-value="{{mCommentId}}">
            <a class="CommentList-commentUser" data-value="{{mUid}}">{{this.mName}}</a>
            <h1 class="CommentList-commentText">{{this.mComment}}</h1>
            <button type="button" class="btn-default CommentList-Delete" data-pid="{{mPostId}}" data-cid="{{mCommentId}}">Delete</button>
        </div>
    {{/each}}
    <textarea id="CommentList-commentInput"></textarea>
    <div class="footer">
        <button type="button" class="btn btn-default" id="CommentList-OK">Ok</button>
        <button type="button" class="btn btn-default" id="CommentList-Close">Close</button>
    </div>
</div>