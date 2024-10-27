<div id="EditEntryForm" class=" modal fade" role="dialog">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h4 class="modal-title">Add a Edit Entry</h4>
            </div>
            <div class="modal-body">
                <label for="EditEntryForm-title">Title</label>
                <input class="form-control" type="text" id="EditEntryForm-title" value="{{this.subject}}" />
                <label for="EditEntryForm-message">Message</label>
                <textarea class="form-control" id="EditEntryForm-message">{{this.message}}</textarea>
                <label for="EditEntryForm-file">file</label>
                <input type="file" class="file-input" id="filely" accept="image/png"></input> 
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-default" id="EditEntryForm-OK" data-value="{{this.id}}">OK</button>
                <button type="button" class="btn btn-default" id="EditEntryForm-Close" data-value="{{this.id}}">Close</button>
            </div>
        </div>
    </div>
</div>