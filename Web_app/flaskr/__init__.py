import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app

    #__name__ is the name of the current Python module. The app needs to know where it’s located. In this case it's __init__.py file
    # instance_relative_config=True tells the app that configuration files are relative to the instance folder.
    # The instance folder is located outside the flaskr package and can hold local data that shouldn’t be committed to version control,
    # such as configuration secrets and the database file.
    app = Flask(__name__, instance_relative_config=True)

    # Sets some default configuration that the app will use:
    app.config.from_mapping(
        # SECRET_KEY is used by Flask and extensions to keep data safe.
        # It’s set to 'dev' to provide a convenient value during development, but it should be overridden with a random value when deploying.
        SECRET_KEY='dev',

        # DATABASE is the path where the SQLite database file will be saved.
        # It’s under app.instance_path, which is the path that Flask has chosen for the instance folder.
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),

    )

    if test_config is None:
        # overrides the default configuration with values taken from the config.py file in the instance folder if it exists.
        # For example, when deploying, this can be used to set a real SECRET_KEY
        app.config.from_pyfile('config.py', silent=True)
    else:
        # test_config can also be passed to the factory, and will be used instead of the instance configuration.
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        # Flask doesn’t create the instance folder automatically, but it needs to be created because your project will create the SQLite database file there.
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # creates a simple route so you can see the application working before getting into the rest of the tutorial.
    # It creates a connection between the URL /hello and a function that returns a response, the string 'Hello, World!' in this case.
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    # Import and call this function from the factory.
    from . import db
    db.init_app(app)

    # Import and register the blueprint from the factory using app.register_blueprint().
    # Place the new code at the end of the factory function before returning the app.

    # The authentication blueprint will have views to register new users and to log in and log out.
    from . import auth
    app.register_blueprint(auth.bp)

    # Unlike the auth blueprint, the blog blueprint does not have a url_prefix.
    # So the index view will be at /, the create view at /create, and so on.
    # The blog is the main feature of Flaskr, so it makes sense that the blog index will be the main index.

    # However, the endpoint for the index view defined below will be blog.index. Some of the authentication views referred to a plain index endpoint.
    # app.add_url_rule() associates the endpoint name 'index' with the / url so that url_for('index') or url_for('blog.index') will both work, generating the same / URL either way.
    from . import blog
    app.register_blueprint(blog.bp)
    app.add_url_rule('/', endpoint='index')

    from . import polititian_names
    app.register_blueprint(polititian_names.bp)


    return app