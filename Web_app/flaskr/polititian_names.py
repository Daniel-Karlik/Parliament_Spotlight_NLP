from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.db import get_db

bp = Blueprint('politician_names', __name__)


@bp.route('/politician_names')
def politician_names():
    db = get_db()
    posts = db.execute(
        'SELECT p.id, title, body, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('blog/politician_names.html', posts=posts)

# The check_author argument is defined so that the function can be used to get a post without checking the author.
# This would be useful if you wrote a view to show an individual post on a page, where the user doesn’t matter because they’re not modifying the post.



# Unlike the views you’ve written so far, the update function takes an argument, id. That corresponds to the <int:id> in the route.
# A real URL will look like /1/update. Flask will capture the 1, ensure it’s an int, and pass it as the id argument.
# If you don’t specify int: and instead do <id>, it will be a string.
# To generate a URL to the update page, url_for() needs to be passed the id so it knows what to fill in: url_for('blog.update', id=post['id']).
# This is also in the index.html file above.

# The create and update views look very similar.
# The main difference is that the update view uses a post object and an UPDATE query instead of an INSERT.
# With some clever refactoring, you could use one view and template for both actions, but for the tutorial it’s clearer to keep them separate.

