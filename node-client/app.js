var createError = require('http-errors');
var express = require('express');
var app = express();
global.config = require('./config/config')[app.get('env')];
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const passport = require('passport');
require('./config/passport')(passport);
const session = require('express-session');
const MySQLStore = require('express-mysql-session')(session);

var db;
(async function () {
    db = await require('./db/db');

    let [results, tableDef] = await db.execute('SELECT domain_string from endpoint where id=1;');
    if (results.length > 0) {
        global.nnEndpoint = results[0]['domain_string'];
    }
})();

var indexRouter = require('./routes/index');
var authedRouter = require('./routes/authed');
var usersRouter = require('./routes/users');

console.log(`using environment: ${app.get('env')}`);

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));

app.use(session({
    secret: global.config.secret,
    store: new MySQLStore(
        {

            host: global.config.host,
            user: global.config.user,
            password: global.config.password,
            database: global.config.database
        }),
    resave: false,
    saveUninitialized: false
}));

app.use(express.json());
app.use(express.urlencoded({extended: false}));

app.use(passport.initialize());
app.use(passport.session());


app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/authed', authedRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function (req, res, next) {
    next(createError(404));
});

// error handler
app.use(function (err, req, res, next) {
    // set locals, only providing error in development
    res.locals.message = err.message;
    res.locals.error = err;

    // render the error page
    res.status(err.status || 500);
    res.render('error');
});

module.exports = app;
