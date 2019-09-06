var express = require('express');
var router = express.Router();
var passport = require('passport');

function validateUser(req, res, next) {
  if(req.user){
    res.redirect('/authed');
  }
  else{
    res.redirect('/login');
  }
}

/* GET home page. */
router.get('/', validateUser, function(req, res, next) {
  res.redirect('login');
});

router.get('/login', function(req, res, next) {
  res.render('login');
});

router.post('/login', passport.authenticate('local', {
  successRedirect: '/authed/',
  failureRedirect: '/login'
}));


module.exports = router;
