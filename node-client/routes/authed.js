const express = require('express');
const router = express.Router();
const uuid = require('uuid/v4');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const diagnosisHandler = require('../activations-downloader');

const upload = multer({
    dest: "./temp/"
});


var db;
(async function () {
    db = await require('../db/db');
})();

function validateUser(req, res, next) {
    if (req.user) {
        next();
    } else {
        res.redirect('/login');
    }
}


router.use(validateUser);

router.get('/send', function (req, res, next) {
    res.render('send');
})


router.get('/', function (req, res, next) {
    res.render('index');
});

router.get('/aboutPage', function (req, res, next) {
    res.render('about');
});

router.get('/uploadPage', function (req, res, next) {
    res.render('upload');
});

router.post('/doesIdExist', async function (req, res, next) {
    console.log(req.body);
    const [results, tableDef] = await db.execute(`Select * from patients where patient_id="${req.body.id}";`);
    if (results.length === 0) {
        res.status(200).send(false);
    } else {
        res.status(200).send(true);
    }
});

router.post('/getScan', async function (req, res, next) {
    console.log(req.body);
    const [results, tableDef] = await db.execute(`Select * from scans where id="${req.body.scanId}";`);
    if (results.length === 0) {
        res.status(200).send({found: false});
    } else {
        res.status(200).send({
            found: true,
            scanData: results[0]
        });
    }
});

router.post('/addPatient', async function (req, res, next) {
    console.log(req.body);
    const [results, tableDef] = await db.execute(`INSERT INTO patients (patient_id, first_name, last_name, age, phone_number) VALUES (${req.body.id}, "${req.body.firstName}", "${req.body.lastName}", "${req.body.age}", "${req.body.phone}");`);
    res.status(200).send(true);
});

router.post('/dignoseImage', function (req, res, next) {
    console.log(req.body);
    /*
    1 save image
    2 update db
    3 send to flask
    */
});

router.get('/searchPage', function (req, res, next) {
    res.render('search');
});

module.exports = router;


router.post(
    "/diagnoseScan",
    upload.single("scan"),
    async function (req, res, next) {
        try {

            if (!req.file) {
                res.status(405);
            }

            const tempPath = req.file.path;
            const ext = path.extname(req.file.originalname).toLowerCase();
            const filename = uuid();
            const targetPath = path.join(__dirname, `../public/scans/${filename}${ext}`);
            if (ext === ".png" || ext === '.jpeg' || ext === '.jpg') {
                fs.rename(tempPath, targetPath, err => {
                    if (err) {
                        console.log(err);
                        return;
                    }
                    // file saved! update database with scan file name...
                    req.filename = filename + ext;
                    req.scanUuid = filename;
                    next();
                });
            } else {
                fs.unlink(tempPath, err => {
                    if (err) {
                        console.log(err);
                        return;
                    }

                    res
                        .status(403)
                        .contentType("text/plain")
                        .end("Only .png / .jpeg / .jpg files are allowed!");
                });

            }
        } catch (e) {
            console.log(e);
        }
    },
    async function (req, res, next) {
        fetch(global.config.nnEndpoint + '/predict', {
            method: 'post',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                scan: `${global.config.endpoint}/scans/${req.filename}`,
                scanGuid: req.scanUuid
            })
        }).then(nnRes => {
            let json = nnRes.json();
            json.then((data) => {
                console.log("Response from NN: ", data);
                res.send(data);
                if (data.heatmap_guid) {
                    diagnosisHandler.emit('scanComplete', data.heatmap_guid);
                }
            })
        })
    }
);