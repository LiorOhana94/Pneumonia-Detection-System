var createError = require('http-errors');
var express = require('express');
var db = require('./db/db'); 
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const passport = require('passport');
require('./config/passport')(passport);
const session = require('express-session');
const MySQLStore = require('express-mysql-session')(session);

var indexRouter = require('./routes/index');
var authedRouter = require('./routes/authed');
var usersRouter = require('./routes/users');

var app = express();

console.log(`using environment: ${app.get('env')}`);

global.config = require('./config/config')[app.get('env')];

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
app.use(express.urlencoded({ extended: false }));

app.use(passport.initialize());
app.use(passport.session());


app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/authed', authedRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
