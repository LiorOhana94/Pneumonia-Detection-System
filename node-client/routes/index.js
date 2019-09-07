const express = require('express');
const router = express.Router();
const passport = require('passport');
const multer = require('multer');
const fs = require('fs');
const path = require('path');


const upload = multer({
    dest: "./temp/"
});

var db;
(async function () {
    db = await require('../db/db');
})();

function validateUser(req, res, next) {
    if (req.user) {
        res.redirect('/authed');
    } else {
        res.redirect('/login');
    }
}

/* GET home page. */
router.get('/', validateUser, function (req, res, next) {
    res.redirect('login');
});

router.get('/login', function (req, res, next) {
    res.render('login');
});

router.post('/upload-map/:hash', async function (req, res, next) {
        const hash = req.params.hash;
        const [results, tableDef] = await db.execute(`Select * from upload_hashes where hash="${hash}" and used=0;`);
        if (results.length !== 0) {
            await db.execute(`update upload_hashes set used=1 where hash="${hash}"`);
            next();
        } else {
            res.send(201);
        }
    },
    upload.single('map'),
    async function (req, res, next) {
        try {

            if (!req.file) {
                res.status(405);
            }

            const tempPath = req.file.path;
            const targetPath = path.join(__dirname, `../public/maps/${req.file.originalname}`);
            fs.rename(tempPath, targetPath, err => {
                if (err) {
                    console.log(err);
                    return;
                }
                // update db with correct map name
                res.send(200);
            });

        } catch (e) {
            console.log(e);
        }
    });

router.post('/login', passport.authenticate('local', {
    successRedirect: '/authed/',
    failureRedirect: '/login'
}));


module.exports = router;
