<div class="panel panel-default" id="ElementList">
    <div class="panel-heading">
        <h3 class="panel-title"><b>Your Feed</b></h3>
    </div>
    <div id="contents">
        {{#each mData}}
        <div id="{{this.mPostId}}" class="ElementList-element">
            <div class="ElementList-heading">
              <a class="ElementList-userName" data-value="{{this.mUid}}" href="#"><b>{{this.mName}}</b></a>
                <a class="ElementList-menu-div" data-value="{{this.mPostId}}" data-user="{{this.mUid}}" href="#">
                    <svg class="ElementList-dots-svg ElementList-svg"><use href="#dots-symbol"></use></svg>
                </a>
            </div>
            <h1 class="ElementList-subject" data-value="{{this.mPostId}}"><b>{{this.mSubject}}</b></h1>
            <h1 class="ElementList-message" data-value="{{this.mPostId}}">{{this.mMessage}}</h1>
            <h1 class="ElementList-file" data-value="{{this.mPostId}}" data-link="{{mLink}}">
                <img src="data:image/png;base64,{{this.mLink}}" onerror="this.onerror=null;this.src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAD0lEQVQYV2NkQAOMpAsAAADuAAUtpgPEAAAAAElFTkSuQmCC';" class="w3-border w3-padding"/>
            </h1>
            <div class="ElementList-feedbackDiv">
                <a class="ElementList-comment-div" data-value="{{this.mPostId}}" href="#">
                    <svg class="ElementList-comment-svg ElementList-svg"><use href="#comment-symbol"></use></svg>
                    <h1 class="ElementList-numComments">{{this.mComments}}</h4>
                </a>
                <a class="ElementList-upvote-div" data-value="{{this.mPostId}}" href="#">
                    <svg class="ElementList-upvote-svg ElementList-svg"><use href="#upvote-symbol"></use></svg>
                    <h1 class="ElementList-upvotes">{{this.mUpvotes}}</h4>
                </a>
                <a class="ElementList-downvote-div" data-value="{{this.mPostId}}" href="#">
                    <svg class="ElementList-downvote-svg ElementList-svg"><use href="#downvote-symbol"></use></svg>
                    <h1 class="ElementList-downvotes">{{this.mDownvotes}}</h4>
                </a>
            </div>
        </div>
        {{/each}}
    </div>
</div>