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

router.get('/t', function (req, res, next) {
    res.render('diagnosisReview', {
        data: {
            "heatmap_guid": "62f1b8cc-3a43-4180-bf18-114b6d6250ac",
            "result_index": 1,
            "result_prob": 0.5173879861831665,
            "result_text": "pneumonia",
            "scan_id": 746,
            "patient_id": 8485969,
            "date": Date(11 / 03 / 2019)
        }
    });
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

router.post('/updateFinalDiagnosis', async function (req, res, next) {
    console.log(req.body);
    const [results, tableDef] = await db.execute(`UPDATE scans SET final_diagnosis_id=${req.body.finalDiagnosis} WHERE scan_id=${req.body.scanID};`);
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


router.post("/diagnoseScan",
    function (req, res, next) {
        next();
    },
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
                fs.rename(tempPath, targetPath, async err => {
                    if (err) {
                        console.log(err);
                        return;
                    }
                    let [results, tableDef] = await db.execute(`SELECT id FROM patients WHERE patient_id=${req.body.patientId}`);

                    if (results.length === 0) {
                        console.log("could not find patient");
                        res.status(405);
                        return;
                    }
                    req.patientId = results[0].id;

                    results = await db.execute(`INSERT INTO scans (patient_id, file_name) VALUES(${results[0].id}, '${filename + ext}')`);
                    req.filename = filename + ext;
                    req.scanId = results[0].insertId;
                    req.scanDate = new Date();
                    req.scanGuid = filename;
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
        fetch(global.nnEndpoint + '/predict', {
            method: 'post',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                scan: `${global.config.endpoint}/scans/${req.filename}`,
                scanGuid: req.scanGuid
            })
        }).then(nnRes => {
            let json = nnRes.json();
            json.then((data) => {
                req.nnResponse = data;
                console.log("Response from NN: ", data);
                next();
                if (data.heatmap_guid) {
                    diagnosisHandler.emit('mapGenerated', data.heatmap_guid);
                }
            })
        })
    },

    async function (req, res, next) {
        await db.execute(`UPDATE scans set system_diagnosis_id=${req.nnResponse['result_text'] === 'pneumonia'? 1 : 2} where id=${req.scanId}`);

        let locals = {
            data: {
                "heatmap_guid": req.nnResponse['heatmap_guid'],
                "result_index": req.nnResponse['result_index'],
                "result_prob": req.nnResponse['result_prob'],
                "result_text": req.nnResponse['result_text'],
                "scan_id": req.scanId,
                "patient_id": req.patientId,
                "date": req.scanDate
            }
        };
        console.log(locals);
        res.render('diagnosisReview', locals);
    }
);

router.get('/updateEndpoint', function (req, res, next) {
    if (req.user.id > 2) {
        res.send(401);
    } else {
        next();
    }
}, function (req, res, next) {
    res.render('updateEndpoint', {success: false});
});

router.post('/updateEndpoint', function (req, res, next) {
    if (req.user.id > 2) {
        res.send(401);
    } else {
        next();
    }
}, async function (req, res, next) {
    let nnEndpoint = 'http://' + req.body.endpoint + ':8080';

    await db.execute(`UPDATE endpoint SET domain_string="${nnEndpoint}" where id=1;`);
    global.nnEndpoint = nnEndpoint;

    res.render('updateEndpoint', {success: true});
})
